#!/usr/bin/env python3

from models import ViscosityModelGraybox2 as Model
from models import train_model
from data import get_data


PLOT_2DHISTOGRAM = True
COMPUTE_METRICS = True

exp_num = 2

compounds = [
    'SiO2', 'Al2O3', 'Na2O', 'CaO', 'MgO', 'B2O3', 'K2O', 'BaO', 'SrO', 'SnO2',
    'ZrO2', 'TiO2', 'ZnO', 'Li2O', 'P2O5', 'PbO', 'Fe2O3', 'SO3', 'Sb2O3',
    'As2O3', 'FeO', 'La2O3', 'MnO', 'GeO2', 'Y2O3', 'Cr2O3', 'Bi2O3', 'CdO',
]

data = get_data(compounds, round_comp_decimal=3, round_temperature_decimal=0)
patience = 13
holdout_size = 0.2
dataloader_num_workers = 4
max_epochs = 200

# Reserving a holdout dataset
h_path = rf'./model_files/experiment_{exp_num:02d}_model_with_holdout.pt'
model_h = train_model(Model, patience, data, compounds, holdout_size,
                      dataloader_num_workers, max_epochs, save_path=h_path)

# Training the final model with all the data
f_path = rf'./model_files/experiment_{exp_num:02d}_model_final.pt'
model_f = train_model(Model, patience, data, compounds, False,
                      dataloader_num_workers, max_epochs, save_path=f_path)

if COMPUTE_METRICS:
    from functools import partial
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.metrics import median_absolute_error as MedAE
    from sklearn.metrics import mean_absolute_error as MAE

    def R2oneParameter(y_true, y_pred):
        return 1 - sum((y_true - y_pred)**2) / sum(y_true**2)

    metrics = ['R2', 'RMSE', 'MAE', 'MedAE']
    metrics_fun = [R2oneParameter, partial(MSE, squared=False), MAE, MedAE]

    test_data = data.loc[model_h.test_idx]

    y = test_data['log_viscosity'].values
    x = test_data.drop('log_viscosity', axis=1)
    y_hat = model_h.eval_from_df(x, compounds)

    print('Metrics for the holdout dataset:')
    for met, fun in zip(metrics, metrics_fun):
        print(rf'{met} = {fun(y, y_hat):.3f}')


if PLOT_2DHISTOGRAM:
    import numpy as np
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from sklearn.metrics import mean_squared_error as MSE
    from sklearn.metrics import median_absolute_error as MedAE
    from sklearn.metrics import mean_absolute_error as MAE

    def R2oneParameter(y_true, y_pred):
        return 1 - sum((y_true - y_pred)**2) / sum(y_true**2)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': 'DejaVu Serif',
        'axes.formatter.limits': [-2, 5],
        'axes.formatter.useoffset': False,
        'axes.formatter.use_mathtext': True,
        'mathtext.fontset': 'dejavuserif',
    })

    fig, axe = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(5, 5),
        dpi=150,
    )

    s = 0.1  # size of the square
    xedges = np.arange(
        round(min(min(y), min(y_hat)), 1),
        round(max(max(y), max(y_hat)), 1),
        s
    )
    yedges = xedges

    H, xedges, yedges = np.histogram2d(y, y_hat, bins=(xedges, yedges))
    H = H.T  
    X, Y = np.meshgrid(xedges, yedges)
    cm = axe.pcolormesh(X, Y, H, cmap='magma_r', norm=LogNorm())
    cb = fig.colorbar(cm, ax=axe, fraction=0.046, pad=0.04)
    cb.set_label('Density')
    axe.plot([min(y)-0.2, max(y)+0.2], [min(y)-0.2,max(y)+0.2], ls='--', alpha=0.7)
    axe.set_xlim((min(y)-0.2), max(y)+0.2)

    axe.set_xlabel(r'Reported $\log_{10}(\eta)$')
    axe.set_ylabel(r'Predicted $\log_{10}(\eta)$')

    textbox = (
        f"$R^2=$ {R2oneParameter(y, y_hat):.3f}\n" 
        f"RMSE = {MSE(y, y_hat, squared=False):.2f}\n"
        f"MAE = {MAE(y, y_hat):.2f}\n"
        f"MedAE = {MedAE(y, y_hat):.2f}"
        )

    if textbox:
        props = dict(boxstyle='round', facecolor='white')
        axe.text(0.95,
                0.05,
                textbox,
                transform=axe.transAxes,
                fontsize=10,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=props)

    ### Inset

    size = '33%'

    ax2 = inset_axes(
        axe,
        width=size,
        height=size,
        loc=2,
    )

    ax2.yaxis.set_label_position('right')
    ax2.set_xlabel('Residual')
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.yaxis.tick_right()

    cols = np.linspace(-5, 5, 100)
    residuals = y - y_hat
    n, bins, patches = ax2.hist(residuals,
                                cols,
                                fc='k',
                                weights=None,
                                align='mid',
                                log=False,
                                ec='k')

    ax2.set_xlim([-2, 2])
    ax2.xaxis.set_ticks([-2, -1, 0, 1, 2])

    fig.savefig(
        rf'./plots/2D_histogram_experiment_{exp_num:02d}.pdf',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )

    plt.close(fig)
