import torch
import ternary
from pathlib import Path
from matplotlib import pyplot as plt
from ternary.helpers import simplex_iterator


### Config

round_fragility = True

scale = 100     # takes some time to compute, reduce to 10 to see what happens
round_base = 5  # round fragility to the nearest multiple of this number
fig_save_path = Path(r'./plots/')

# Choose the ternanies that you want to compute
ternaries = [
    ('Na2O', 'SiO2', 'CaO'),
    # ('Na2O', 'SiO2', 'Al2O3'),
]

# Translator to beautify the plot labels
chemtrans = {
    'SiO2': r'$\mathrm{SiO_2}$',
    'Na2O': r'$\mathrm{Na_2O}$',
    'Li2O': r'$\mathrm{Li_2O}$',
    'CaO': r'$\mathrm{CaO}$',
}


### Code

path = Path(r'./model_files/experiment_01_model_final.pt')
model = torch.load(path)


def gen_heatmap(comp1, comp2, comp3, scale):

    def computeValue(comp1, comp2, comp3, quant1, quant2, quant3):
        glass = {
            comp1: quant1,
            comp2: quant2,
            comp3: quant3,
        }
        _, _, m, _, _, _ = model.get_params_from_dict(glass)
        return m

    d = dict()
    for (i, j, k) in simplex_iterator(scale):
        d[(i, j, k)] = computeValue(comp1, comp2, comp3, i, j, k,)
    return d


for comp1, comp2, comp3 in ternaries:

    data = gen_heatmap(comp1, comp2, comp3, scale)

    if round_fragility:
        data = {key: round_base * round(data[key]/round_base) for key in data}

    fig, tax = ternary.figure(scale=scale)

    tax.heatmap(
        data,
        style="hexagonal",
        use_rgba=False,
        colorbar=True,
        cbarlabel='Fragility',
        cmap='viridis_r',
    )

    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.boundary()

    tax.left_axis_label(
        chemtrans.get(comp3, comp3) + ' mol%',
        fontsize=10,
        offset=0.14,
    )
    tax.right_axis_label(
        chemtrans.get(comp2, comp2) + ' mol%',
        fontsize=10,
        offset=0.14,
    )
    tax.bottom_axis_label(
        chemtrans.get(comp1, comp1) + ' mol%',
        fontsize=10,
        offset=0.14,
    )

    tax.ticks(axis='lbr', linewidth=1, multiple=scale/10, offset=0.03)

    fig.savefig(fig_save_path / rf'ternary_fragility_{comp1}_{comp2}_{comp3}.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=2e-2)

    plt.close(fig)
