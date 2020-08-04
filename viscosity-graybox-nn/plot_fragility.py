import torch
import ternary
from ternary.helpers import simplex_iterator
from matplotlib import pyplot as plt
from pathlib import Path

from models import ViscosityModelGraybox1 as Model
from models import load_model
from data import get_data


compounds = [
    'SiO2', 'Al2O3', 'Na2O', 'CaO', 'MgO', 'B2O3', 'K2O', 'BaO', 'SrO', 'SnO2',
    'ZrO2', 'TiO2', 'ZnO', 'Li2O', 'P2O5', 'PbO', 'Fe2O3', 'SO3', 'Sb2O3',
    'As2O3', 'FeO', 'La2O3', 'MnO', 'GeO2', 'Y2O3', 'Cr2O3', 'Bi2O3', 'CdO',
]

data = get_data(compounds, round_comp_decimal=3, round_temperature_decimal=0)

path = Path(r'./model_files/experiment_01_model_final.pt')
model = torch.load(path1)


def generate_heatmap_data(comp1, comp2, comp3, precision=5):

    def computeValue(comp1, comp2, comp3, quant1, quant2, quant3):
        glass = {
            comp1: quant1,
            comp2: quant2,
            comp3: quant3,
        }
        _, _, m, _, _, _ = model.get_params_from_dict(glass)
        return m

    d = dict()
    for (i, j, k) in simplex_iterator(precision):
        d[(i, j, k)] = computeValue(comp1, comp2, comp3, i, j, k,)
    return d


precision = 100
round_base = 5

ternaries = [
    ('Na2O', 'SiO2', 'CaO'),
]

chemtrans = {
    'SiO2': r'$\mathrm{SiO_2}$',
    'Na2O': r'$\mathrm{Na_2O}$',
    'Li2O': r'$\mathrm{Li_2O}$',
    'CaO': r'$\mathrm{CaO}$',
}

for comp1, comp2, comp3 in ternaries:

    data = generate_heatmap_data(comp1, comp2, comp3, precision)

    for key in data:
        data[key] = round_base * round(data[key]/round_base)

    fig, tax = ternary.figure(scale=precision)

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
        chemtrans[comp3] + ' ' + 'mol%',
        fontsize=10,
        offset=0.14,
    )
    tax.right_axis_label(
        chemtrans[comp2] + ' ' + 'mol%',
        fontsize=10,
        offset=0.14,
    )
    tax.bottom_axis_label(
        chemtrans[comp1] + ' ' + 'mol%',
        fontsize=10,
        offset=0.14,
    )

    tax.ticks(axis='lbr', linewidth=1, multiple=10, offset=0.03)

    fig.savefig(rf'./plots/ternary_fragility_{comp1}_{comp2}_{comp3}.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=2e-2)

    plt.close(fig)
