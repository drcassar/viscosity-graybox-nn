import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.model_selection import train_test_split
from chemparse import parse_formula
from mendeleev import element

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


class ViscosityModelGraybox1(pl.LightningModule):
    def __init__(
            self,
            dataframe,
            composition_cols,
            holdout_size=0.2,
            dataloader_num_workers=1,
    ):
        super().__init__()

        self.data = dataframe
        self.composition_cols = composition_cols
        self.holdout_size = holdout_size
        self.dl_num_workers = dataloader_num_workers
        self.loss_fun = F.mse_loss

        self.feature_names = [
            'abs|atomic_number|std', 'abs|atomic_radius_rahm|max',
            'abs|atomic_radius_rahm|mean', 'abs|atomic_radius_rahm|std',
            'abs|atomic_weight|std', 'abs|boiling_point|sum',
            'abs|boiling_point|mean', 'abs|boiling_point|std',
            'abs|covalent_radius_pyykko|std',
            'abs|covalent_radius_pyykko_double|mean', 'abs|density|std',
            'abs|dipole_polarizability|sum', 'abs|electrons|std',
            'abs|en_ghosh|sum', 'abs|heat_of_formation|sum',
            'abs|heat_of_formation|min', 'abs|heat_of_formation|mean',
            'abs|heat_of_formation|std', 'abs|lattice_constant|mean',
            'abs|lattice_constant|std', 'abs|mass_number|std',
            'abs|melting_point|sum', 'abs|melting_point|mean',
            'abs|melting_point|std', 'abs|mendeleev_number|min',
            'abs|neutrons|std', 'abs|protons|std', 'abs|vdw_radius_alvarez|std',
            'abs|vdw_radius_uff|min', 'abs|vdw_radius_uff|max',
            'abs|vdw_radius_uff|mean',
            'wei|atomic_number|sum', 'wei|atomic_radius|max',
            'wei|atomic_radius_rahm|sum', 'wei|atomic_radius_rahm|max',
            'wei|atomic_volume|sum', 'wei|atomic_volume|max',
            'wei|atomic_volume|std', 'wei|atomic_weight|sum',
            'wei|boiling_point|sum', 'wei|boiling_point|max',
            'wei|boiling_point|mean', 'wei|boiling_point|std', 'wei|c6_gb|sum',
            'wei|c6_gb|min', 'wei|c6_gb|max', 'wei|c6_gb|std',
            'wei|density|sum', 'wei|density|max', 'wei|density|mean',
            'wei|density|std', 'wei|dipole_polarizability|min',
            'wei|dipole_polarizability|max', 'wei|dipole_polarizability|std',
            'wei|electrons|sum', 'wei|en_ghosh|max', 'wei|en_pauling|max',
            'wei|en_pauling|std', 'wei|glawe_number|max',
            'wei|heat_of_formation|sum', 'wei|heat_of_formation|std',
            'wei|lattice_constant|sum', 'wei|lattice_constant|max',
            'wei|lattice_constant|std', 'wei|mass_number|sum',
            'wei|melting_point|sum', 'wei|melting_point|min',
            'wei|melting_point|max', 'wei|melting_point|mean',
            'wei|melting_point|std', 'wei|mendeleev_number|sum',
            'wei|mendeleev_number|max', 'wei|neutrons|sum', 'wei|neutrons|max',
            'wei|neutrons|std', 'wei|pettifor_number|sum',
            'wei|pettifor_number|max', 'wei|protons|sum', 'wei|vdw_radius|sum',
            'wei|vdw_radius|max', 'wei|vdw_radius_alvarez|sum',
            'wei|vdw_radius_alvarez|max', 'wei|vdw_radius_batsanov|max',
            'wei|vdw_radius_mm3|max', 'wei|vdw_radius_uff|sum',
            'wei|vdw_radius_uff|max', 'wei|vdw_radius_uff|mean',
        ]

        attribute_list = []
        for f in self.feature_names:
            _, attribute, _ = f.split('|')
            attribute_list.append(attribute)
        self.attributes = list(sorted(set(attribute_list)))

        self.linear = torch.nn.Linear(len(self.feature_names), 128)
        self.drop = torch.nn.Dropout(0.03143042658495489)
        self.relu = nn.ReLU()
        self.output_layer = nn.Sequential(
            nn.Linear(128, 3),
            nn.Identity(),
        )

        elements = []
        all_attributes = []
        for i in range(1,119):
            el = element(i)
            elements.append(el.symbol)
            element_features = []
            for attr in self.attributes:
                element_features.append(getattr(el, attr, np.nan))
            all_attributes.append(element_features)

        self.chemical_attributes = pd.DataFrame(
            np.array(all_attributes).T,
            columns=elements,
            index=self.attributes
        ).dropna(axis=1)

    def gen_atomic_df(self, composition_df):
        compound_lst = composition_df.columns.tolist()
        all_elements = self.chemical_attributes.columns.tolist()

        element_guide = np.zeros((len(all_elements), len(compound_lst)))
        for j in range(len(compound_lst)):
            c = compound_lst[j]
            cdic = parse_formula(c)
            for el in cdic:
                i = all_elements.index(el)
                element_guide[i,j] += cdic[el]

        atomic_df = np.zeros((len(composition_df), len(all_elements)))
        for i in range(len(compound_lst)):
            c = compound_lst[i]
            cdic = parse_formula(c)
            for el in cdic:
                j = all_elements.index(el)
                atomic_df[:,j] += composition_df[c].values*element_guide[j,i]

        atomic_df = pd.DataFrame(
            atomic_df,
            columns=all_elements,
            index=composition_df.index,
        )
        atomic_df = atomic_df.div(atomic_df.sum(axis=1), axis=0)

        return atomic_df

    def featurizer(self, composition_df):
        atomic_df = self.gen_atomic_df(composition_df)

        wei_data = atomic_df.copy()
        logic = wei_data > 0
        wei_data[~logic] = pd.NA

        abs_data = atomic_df.copy().astype(bool)
        abs_data[abs_data == False] = pd.NA

        data = pd.DataFrame([], index=atomic_df.index)

        for attr in self.attributes:
            attr_vec = self.chemical_attributes.loc[attr]
            wei = attr_vec * wei_data
            ab = attr_vec * abs_data

            for feat in self.feature_names:
                abs_or_wei, feat_, method = feat.split('|')
                if attr == feat_:
                    if abs_or_wei == 'abs':
                        data[f'abs|{attr}|{method}'] = \
                            getattr(ab, method)(axis=1, skipna=True)
                    else:
                        data[f'wei|{attr}|{method}'] = \
                            getattr(wei, method)(axis=1, skipna=True)

        return data

    def prepare_data(self):
        if self.holdout_size:
            train_val, self.test_idx = train_test_split(
                self.data.index,
                test_size=self.holdout_size,
                random_state=42,
                shuffle=True,
            )
        else:
            train_val = self.data.index
            self.test_idx = self.data.index

        self.train_idx, self.val_idx = train_test_split(
            train_val,
            test_size=0.1,
            random_state=72,
            shuffle=True,
        )

        composition_df = self.data[self.composition_cols]
        self.features = self.featurizer(composition_df)

        feat_train = self.features.loc[self.train_idx]
        feat_train_tensor = torch.from_numpy(feat_train.values).float()
        self.x_train_mean = feat_train_tensor.mean(0, keepdim=True)
        self.x_train_std = feat_train_tensor.std(0, unbiased=False, keepdim=True)

        self.features['temperature'] = self.data['temperature']
        self.target = self.data['log_viscosity']

    def x_scaler(self, x):
        x -= self.x_train_mean
        x /= self.x_train_std
        return x

    def forward(self, x):
        feats = x[:, :-1]
        T = x[:, -1]

        xf = self.x_scaler(feats)
        xf = self.linear(xf)
        xf = self.drop(xf)
        xf = self.relu(xf)
        xf = self.output_layer(xf)

        log_eta_inf = torch.mul(xf[:, 0], -10)
        K = torch.mul(xf[:, 1], 1000)
        C = torch.mul(xf[:, 2], 1000)

        log_viscosity = log_eta_inf + K / T + (C / T).exp()

        return log_viscosity

    def _get_params(self, feats):
        xf = torch.from_numpy(feats.values).float()
        xf = self.x_scaler(xf)
        xf = self.linear(xf)
        xf = self.relu(xf)
        xf = self.output_layer(xf)

        log_eta_inf = torch.mul(xf[:, 0], -10).cpu().detach().numpy()
        K = torch.mul(xf[:, 1], 1000).cpu().detach().numpy()
        C = torch.mul(xf[:, 2], 1000).cpu().detach().numpy()

        if len(log_eta_inf) == 1:
            log_eta_inf = log_eta_inf[0]
        if len(K) == 1:
            K = K[0]
        if len(C) == 1:
            C = C[0]

        return log_eta_inf, K, C
        
    def _get_extra_params(self, log_eta_inf, K, C):
        def log_viscosity_fun(T):
            log_viscosity = log_eta_inf + K / T + np.exp(C / T)
            return log_viscosity

        def fun_T12(T):
            return log_viscosity_fun(T) - 12

        T12 = brentq(fun_T12, 200, 5000)
        m = (C/T12+1)*(12-log_eta_inf)  

        return log_viscosity_fun, T12, m
        
    def get_params_from_string(self, composition_string):
        composition_df = pd.DataFrame([1,], columns=[composition_string])
        feats = self.featurizer(composition_df)
        log_eta_inf, K, C = self._get_params(feats)
        log_viscosity_fun, T12, m = self._get_extra_params(log_eta_inf, K, C)
        return log_viscosity_fun, T12, m, log_eta_inf, K, C

    def get_params_from_dict(self, composition_dict):
        composition_df = pd.DataFrame.from_dict(
            composition_dict,
            orient='index',
        ).T
        feats = self.featurizer(composition_df)
        log_eta_inf, K, C = self._get_params(feats)
        log_viscosity_fun, T12, m = self._get_extra_params(log_eta_inf, K, C)
        return log_viscosity_fun, T12, m, log_eta_inf, K, C

    def get_params_from_df(self, composition_df, extra_params=True):
        feats = self.featurizer(composition_df)
        log_eta_inf, K, C = self._get_params(feats)

        params_df = pd.DataFrame(
            [log_eta_inf, K, C],
            columns=['log_eta_inf', 'K', 'C'],
        )

        if extra_params:
            T12_lst = []
            m_lst = []
            for _, row in params_df.iterrows():
                log_eta_inf, K, C = row
                _, T12, m = self._get_extra_params(log_eta_inf, K, C)
                T12_lst.append(T12)
                m_lst.append(m)

            params_df['T12'] = T12_lst
            params_df['m'] = m_lst

        return params_df

    def eval_from_string(self, composition_string, T):
        log_viscosity_fun, _, _, _, _, _ = \
            self.get_params_from_string(composition_string)
        log_viscosity = log_viscosity_fun(T)
        return log_viscosity

    def eval_from_dict(self, composition_dict, T):
        log_viscosity_fun, _, _, _, _, _ = \
            self.get_params_from_dict(composition_dict)
        log_viscosity = log_viscosity_fun(T)
        return log_viscosity

    def eval_from_df(self, dataframe, composition_cols):
        '''
        Note: data must contain a 'temperature' column.

        '''
        composition_df = dataframe[composition_cols]
        features = self.featurizer(composition_df)
        T = torch.from_numpy(dataframe['temperature'].values).float()
        x = torch.from_numpy(features.values).float()
        xf = self.x_scaler(x)
        xf = self.linear(xf)
        xf = self.relu(xf)
        xf = self.output_layer(xf)
        log_eta_inf = torch.mul(xf[:, 0], -10)
        K = torch.mul(xf[:, 1], 1000)
        C = torch.mul(xf[:, 2], 1000)
        log_viscosity = log_eta_inf + K / T + (C / T).exp()
        log_viscosity = log_viscosity.cpu().detach().numpy()
        return log_viscosity

    def configure_optimizers(self):
        return SGD(
            self.parameters(),
            lr=1.0605501618656703e-05,
            momentum=0.9752040845183261,
        )

    def train_dataloader(self):
        feat_train = self.features.loc[self.train_idx]
        target_train = self.target.loc[self.train_idx]

        x_train = torch.from_numpy(feat_train.values).float()
        y_train = torch.from_numpy(target_train.values).float()

        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds,
                              batch_size=256,
                              shuffle=True,
                              num_workers=self.dl_num_workers)

        return train_dl

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        return {'loss': loss}

    def val_dataloader(self):
        feat_val = self.features.loc[self.val_idx]
        target_val = self.target.loc[self.val_idx]

        x_val = torch.from_numpy(feat_val.values).float()
        y_val = torch.from_numpy(target_val.values).float()

        val_ds = TensorDataset(x_val, y_val)
        val_dl = DataLoader(val_ds,
                            batch_size=256 * 2,
                            num_workers=self.dl_num_workers)

        return val_dl

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def test_dataloader(self):
        feat_test = self.features.loc[self.test_idx]
        target_test = self.target.loc[self.test_idx]

        x_test = torch.from_numpy(feat_test.values).float()
        y_test = torch.from_numpy(target_test.values).float()

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds,
                             batch_size=256 * 2,
                             num_workers=self.dl_num_workers)

        return test_dl

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        loss = self.loss_fun(y_hat, y)
        return {'test_loss': loss, 'mse_loss': mse_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_mse_loss = torch.stack([x['mse_loss'] for x in outputs]).mean()
        self.test_loss = float(avg_loss)
        return {'test_loss': avg_loss}

    def clear_data(self):
        delattr(self, 'data')
        delattr(self, 'composition_cols')
        delattr(self, 'features')
        delattr(self, 'target')
        delattr(self, 'train_idx')
        delattr(self, 'val_idx')
        delattr(self, 'test_idx')


class ViscosityModelGraybox2(pl.LightningModule):
    def __init__(
            self,
            dataframe,
            composition_cols,
            holdout_size=0.2,
            dataloader_num_workers=1,
    ):
        super().__init__()

        self.data = dataframe
        self.composition_cols = composition_cols
        self.holdout_size = holdout_size
        self.dl_num_workers = dataloader_num_workers
        self.loss_fun = F.mse_loss

        self.feature_names = [
            'abs|atomic_number|std', 'abs|atomic_radius_rahm|max',
            'abs|atomic_radius_rahm|mean', 'abs|atomic_radius_rahm|std',
            'abs|atomic_weight|std', 'abs|boiling_point|sum',
            'abs|boiling_point|mean', 'abs|boiling_point|std',
            'abs|covalent_radius_pyykko|std',
            'abs|covalent_radius_pyykko_double|mean', 'abs|density|std',
            'abs|dipole_polarizability|sum', 'abs|electrons|std',
            'abs|en_ghosh|sum', 'abs|heat_of_formation|sum',
            'abs|heat_of_formation|min', 'abs|heat_of_formation|mean',
            'abs|heat_of_formation|std', 'abs|lattice_constant|mean',
            'abs|lattice_constant|std', 'abs|mass_number|std',
            'abs|melting_point|sum', 'abs|melting_point|mean',
            'abs|melting_point|std', 'abs|mendeleev_number|min',
            'abs|neutrons|std', 'abs|protons|std', 'abs|vdw_radius_alvarez|std',
            'abs|vdw_radius_uff|min', 'abs|vdw_radius_uff|max',
            'abs|vdw_radius_uff|mean',
            'wei|atomic_number|sum', 'wei|atomic_radius|max',
            'wei|atomic_radius_rahm|sum', 'wei|atomic_radius_rahm|max',
            'wei|atomic_volume|sum', 'wei|atomic_volume|max',
            'wei|atomic_volume|std', 'wei|atomic_weight|sum',
            'wei|boiling_point|sum', 'wei|boiling_point|max',
            'wei|boiling_point|mean', 'wei|boiling_point|std', 'wei|c6_gb|sum',
            'wei|c6_gb|min', 'wei|c6_gb|max', 'wei|c6_gb|std',
            'wei|density|sum', 'wei|density|max', 'wei|density|mean',
            'wei|density|std', 'wei|dipole_polarizability|min',
            'wei|dipole_polarizability|max', 'wei|dipole_polarizability|std',
            'wei|electrons|sum', 'wei|en_ghosh|max', 'wei|en_pauling|max',
            'wei|en_pauling|std', 'wei|glawe_number|max',
            'wei|heat_of_formation|sum', 'wei|heat_of_formation|std',
            'wei|lattice_constant|sum', 'wei|lattice_constant|max',
            'wei|lattice_constant|std', 'wei|mass_number|sum',
            'wei|melting_point|sum', 'wei|melting_point|min',
            'wei|melting_point|max', 'wei|melting_point|mean',
            'wei|melting_point|std', 'wei|mendeleev_number|sum',
            'wei|mendeleev_number|max', 'wei|neutrons|sum', 'wei|neutrons|max',
            'wei|neutrons|std', 'wei|pettifor_number|sum',
            'wei|pettifor_number|max', 'wei|protons|sum', 'wei|vdw_radius|sum',
            'wei|vdw_radius|max', 'wei|vdw_radius_alvarez|sum',
            'wei|vdw_radius_alvarez|max', 'wei|vdw_radius_batsanov|max',
            'wei|vdw_radius_mm3|max', 'wei|vdw_radius_uff|sum',
            'wei|vdw_radius_uff|max', 'wei|vdw_radius_uff|mean',
        ]

        attribute_list = []
        for f in self.feature_names:
            _, attribute, _ = f.split('|')
            attribute_list.append(attribute)
        self.attributes = list(sorted(set(attribute_list)))

        self.linear = torch.nn.Linear(len(self.feature_names) + 1, 128)
        self.drop = torch.nn.Dropout(0.03143042658495489)
        self.relu = nn.ReLU()
        self.output_layer = nn.Sequential(
            nn.Linear(128, 3),
            nn.Identity(),
        )

        elements = []
        all_attributes = []
        for i in range(1,119):
            el = element(i)
            elements.append(el.symbol)
            element_features = []
            for attr in self.attributes:
                element_features.append(getattr(el, attr, np.nan))
            all_attributes.append(element_features)

        self.chemical_attributes = pd.DataFrame(
            np.array(all_attributes).T,
            columns=elements,
            index=self.attributes
        ).dropna(axis=1)

    def gen_atomic_df(self, composition_df):
        compound_lst = composition_df.columns.tolist()
        all_elements = self.chemical_attributes.columns.tolist()

        element_guide = np.zeros((len(all_elements), len(compound_lst)))
        for j in range(len(compound_lst)):
            c = compound_lst[j]
            cdic = parse_formula(c)
            for el in cdic:
                i = all_elements.index(el)
                element_guide[i,j] += cdic[el]

        atomic_df = np.zeros((len(composition_df), len(all_elements)))
        for i in range(len(compound_lst)):
            c = compound_lst[i]
            cdic = parse_formula(c)
            for el in cdic:
                j = all_elements.index(el)
                atomic_df[:,j] += composition_df[c].values*element_guide[j,i]

        atomic_df = pd.DataFrame(
            atomic_df,
            columns=all_elements,
            index=composition_df.index,
        )
        atomic_df = atomic_df.div(atomic_df.sum(axis=1), axis=0)

        return atomic_df

    def featurizer(self, composition_df):
        atomic_df = self.gen_atomic_df(composition_df)

        wei_data = atomic_df.copy()
        logic = wei_data > 0
        wei_data[~logic] = pd.NA

        abs_data = atomic_df.copy().astype(bool)
        abs_data[abs_data == False] = pd.NA

        data = pd.DataFrame([], index=atomic_df.index)

        for attr in self.attributes:
            attr_vec = self.chemical_attributes.loc[attr]
            wei = attr_vec * wei_data
            ab = attr_vec * abs_data

            for feat in self.feature_names:
                abs_or_wei, feat_, method = feat.split('|')
                if attr == feat_:
                    if abs_or_wei == 'abs':
                        data[f'abs|{attr}|{method}'] = \
                            getattr(ab, method)(axis=1, skipna=True)
                    else:
                        data[f'wei|{attr}|{method}'] = \
                            getattr(wei, method)(axis=1, skipna=True)

        return data

    def prepare_data(self):
        if self.holdout_size:
            train_val, self.test_idx = train_test_split(
                self.data.index,
                test_size=self.holdout_size,
                random_state=42,
                shuffle=True,
            )
        else:
            train_val = self.data.index
            self.test_idx = self.data.index

        self.train_idx, self.val_idx = train_test_split(
            train_val,
            test_size=0.1,
            random_state=72,
            shuffle=True,
        )

        composition_df = self.data[self.composition_cols]
        self.features = self.featurizer(composition_df)
        self.features['temperature'] = self.data['temperature']

        feat_train = self.features.loc[self.train_idx]
        feat_train_tensor = torch.from_numpy(feat_train.values).float()
        self.x_train_mean = feat_train_tensor.mean(0, keepdim=True)
        self.x_train_std = feat_train_tensor.std(0, unbiased=False, keepdim=True)

        self.target = self.data['log_viscosity']

    def x_scaler(self, x):
        x -= self.x_train_mean
        x /= self.x_train_std
        return x

    def forward(self, x):
        T = x[:, -1].clone().detach()

        xf = self.x_scaler(x)
        xf = self.linear(xf)
        xf = self.drop(xf)
        xf = self.relu(xf)
        xf = self.output_layer(xf)

        log_eta_inf = torch.mul(xf[:, 0], -10)
        K = torch.mul(xf[:, 1], 1000)
        C = torch.mul(xf[:, 2], 1000)

        log_viscosity = log_eta_inf + K / T + (C / T).exp()

        return log_viscosity

    def eval_from_df(self, dataframe, composition_cols):
        '''
        Note: data must contain a 'temperature' column.

        '''
        composition_df = dataframe[composition_cols]
        features = self.featurizer(composition_df)
        features['temperature'] = dataframe['temperature']
        x = torch.from_numpy(features.values).float()
        T = x[:, -1].clone().detach()
        xf = self.x_scaler(x)
        xf = self.linear(xf)
        xf = self.relu(xf)
        xf = self.output_layer(xf)
        log_eta_inf = torch.mul(xf[:, 0], -10)
        K = torch.mul(xf[:, 1], 1000)
        C = torch.mul(xf[:, 2], 1000)
        log_viscosity = log_eta_inf + K / T + (C / T).exp()
        log_viscosity = log_viscosity.cpu().detach().numpy()
        return log_viscosity

    def eval_from_dict(self, composition_dict, T):
        composition_cols = list(composition_dict.keys())
        dataframe = pd.DataFrame.from_dict({**composition_dict, 'T':T})
        log_viscosity = self.eval_from_df(dataframe, composition_cols)
        return log_viscosity

    def eval_from_string(self, composition_string, T):
        log_viscosity = self.eval_from_dict(
            {composition_string:1},
            T,
        )
        return log_viscosity

    def configure_optimizers(self):
        return SGD(
            self.parameters(),
            lr=1.0605501618656703e-05,
            momentum=0.9752040845183261,
        )

    def train_dataloader(self):
        feat_train = self.features.loc[self.train_idx]
        target_train = self.target.loc[self.train_idx]

        x_train = torch.from_numpy(feat_train.values).float()
        y_train = torch.from_numpy(target_train.values).float()

        train_ds = TensorDataset(x_train, y_train)
        train_dl = DataLoader(train_ds,
                              batch_size=256,
                              shuffle=True,
                              num_workers=self.dl_num_workers)

        return train_dl

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        return {'loss': loss}

    def val_dataloader(self):
        feat_val = self.features.loc[self.val_idx]
        target_val = self.target.loc[self.val_idx]

        x_val = torch.from_numpy(feat_val.values).float()
        y_val = torch.from_numpy(target_val.values).float()

        val_ds = TensorDataset(x_val, y_val)
        val_dl = DataLoader(val_ds,
                            batch_size=256 * 2,
                            num_workers=self.dl_num_workers)

        return val_dl

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fun(y_hat, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def test_dataloader(self):
        feat_test = self.features.loc[self.test_idx]
        target_test = self.target.loc[self.test_idx]

        x_test = torch.from_numpy(feat_test.values).float()
        y_test = torch.from_numpy(target_test.values).float()

        test_ds = TensorDataset(x_test, y_test)
        test_dl = DataLoader(test_ds,
                             batch_size=256 * 2,
                             num_workers=self.dl_num_workers)

        return test_dl

    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        mse_loss = F.mse_loss(y_hat, y)
        loss = self.loss_fun(y_hat, y)
        return {'test_loss': loss, 'mse_loss': mse_loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_mse_loss = torch.stack([x['mse_loss'] for x in outputs]).mean()
        self.test_loss = float(avg_loss)
        return {'test_loss': avg_loss, 'test_mse_loss': avg_mse_loss}

    def clear_data(self):
        delattr(self, 'data')
        delattr(self, 'composition_cols')
        delattr(self, 'features')
        delattr(self, 'target')
        delattr(self, 'train_idx')
        delattr(self, 'val_idx')
        delattr(self, 'test_idx')


def train_model(model_class, patience, data, composition_cols, holdout,
                n_dataloaders, max_epochs, save_path=False):

    pl.seed_everything(61455)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=patience,
        verbose=False,
        mode='min',
    )

    model = model_class(
        data,
        composition_cols,
        holdout,
        n_dataloaders,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        early_stop_callback=early_stop_callback,
        progress_bar_refresh_rate=10,
        deterministic=True,
    )

    trainer.fit(model)
    trainer.test()

    if save_path:

        state_dict = model.state_dict()

        model_ = model_class(
            data,
            composition_cols,
            holdout,
            n_dataloaders,
        )

        model_.load_state_dict(state_dict)
        model_.prepare_data()
        model_.clear_data()
        torch.save(model_, save_path)

    return model


def load_model(model_class, path, data=False, composition_cols=False,
               holdout=False, n_dataloaders=False):

    model = torch.load(path)

    if data is not False:

        state_dict = model.state_dict()

        model = model_class(
            data,
            composition_cols,
            holdout,
            n_dataloaders,
        )

        model.load_state_dict(state_dict)
        model.prepare_data()

    return model
