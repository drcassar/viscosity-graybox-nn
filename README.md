# Gray-box Neural Network Viscosity Model
[![DOI](https://zenodo.org/badge/277125509.svg)](https://zenodo.org/badge/latestdoi/277125509)
[![arXiv](https://img.shields.io/badge/arXiv-2007.03719-b31b1b.svg)](https://arxiv.org/abs/2007.03719)

This repository has a gray-box Neural Network to predict the viscosity of liquids. It was trained using PyTorch/PyTorch-lightning on the SciGlass database. For more information, please see the pre-print "Reproducible gray-box neural network for predicting the fragility index and the temperature-dependency of viscosity".

## How to use
Python 3.6+ is required to run the code. The recommended procedure is to create a new virtual environment and install the necessary modules by running

``` sh
pip install -r requirements.txt
```

### Training the models
To train the Gray-box 1 and Gray-box 2 models, run the scripts "experiment_01.py" and "experiment_02.py". The model files will be saved in the folder "model_files". This step is not necessary, as trained model files are already available. However, you should rebuild the model files if you change the model classes or if you want to reproduce this work.

### Viscosity plot
To generate viscosity plots, run the script "plot_viscosity.py". It is pre-configured to create the plots shown in the pre-print, but this can easily be changed to produce new plots for other liquids. The script exports the results in the "plots" folder.

### Fragility ternary plot
To generate the fragility ternary plot, run the script "plot_fragility.py". It is pre-configured to create the plot for the system SiO2-Na2O-CaO, which is discussed in the original publication. 

## Issues and how to contribute
If you find bugs or have questions, please open an issue. PRs are most welcome.

## How to cite
D.R. Cassar, Reproducible gray-box neural network for predicting the fragility index and the temperature-dependency of viscosity, ArXiv:2007.03719. (2020). http://arxiv.org/abs/2007.03719.
  
## License
[GPL](LICENSE)

Gray-box Neural Network Viscosity Model. Copyright (C) 2020 Daniel Roberto Cassar

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
