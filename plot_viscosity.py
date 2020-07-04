#!/usr/bin/env python3

import torch
import numpy as np
from matplotlib import pyplot as plt
from glasspy.data.viscosity import viscosityFromString


plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': 'DejaVu Serif',
    'axes.formatter.limits': [-2, 5],
    'axes.formatter.useoffset': False,
    'axes.formatter.use_mathtext': True,
    'mathtext.fontset': 'dejavuserif',
})

path1 = rf'./model_files/experiment_01_model_final.pt'
model1 = torch.load(path1)

path2 = rf'./model_files/experiment_02_model_final.pt'
model2 = torch.load(path2)

compositions = [
    '(SiO2)2(CaO)1(MgO)1',
    '(SiO2)7(Na2O)2(CoO)1',
    '(SiO2)75(Rb2O)25',
]

for comp in compositions:

    fig, axe = plt.subplots(
        ncols=1,
        nrows=1,
        figsize=(4, 4),
        dpi=150,
    )

    axe.set_ylabel('$\log_{10}(\eta)$  [$\eta$ in Pa.s]')
    axe.set_xlabel('$1/T$  [K]')

    data_comp = viscosityFromString(comp)

    T = np.linspace(
        min(data_comp['temperature']),
        max(data_comp['temperature']),
    )

    y1 = model1.eval_from_string(comp, T)
    y2 = model2.eval_from_string(comp, T)

    axe.plot(
        1 / T,
        y1,
        c='tab:orange',
        label='Gray-box 1',
    )

    axe.plot(
        1 / T,
        y2,
        c='tab:blue',
        label='Gray-box 2',
    )

    axe.plot(
        1 / data_comp['temperature'],
        data_comp['log_viscosity'],
        marker='o',
        ls='none',
        markeredgecolor='black',
        c='silver',
        label='Experimental data'
    )

    axe.legend(loc=2)

    fig.savefig(
        rf'./plots/viscosity_plot_{comp}.pdf',
        dpi=150,
        bbox_inches='tight',
        pad_inches=2e-2,
    )

    plt.close(fig)
