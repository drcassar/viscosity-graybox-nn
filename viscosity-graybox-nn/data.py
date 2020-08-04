from glasspy.data.viscosity import loadViscosityTable


def get_data(compounds, round_comp_decimal=5, round_temperature_decimal=2):

    # Loading the data
    data = loadViscosityTable(True)
    data = data.drop(['author', 'year'], axis=1)
    chemical_columns = data.columns.values[:-2]
    data[chemical_columns] = data[chemical_columns].round(round_comp_decimal)

    # Removing duplicates
    data['temperature'] = data['temperature'].round(round_temperature_decimal)
    grouped = data.groupby(list(chemical_columns) + ['temperature'], sort=False)
    data = grouped.median().reset_index()

    # Removing columns with no information
    nonzero_cols_bool = data.sum(axis=0).astype(bool)
    zero_cols = data.columns.values[~nonzero_cols_bool]
    data = data.drop(zero_cols, axis=1)

    # Filtering the dataset
    for c in set(data.columns) - set(compounds) - set(['temperature', 'log_viscosity']):
        data = data[data[c] == 0]
    data = data.reindex(compounds + ['temperature', 'log_viscosity'], axis=1)

    # Removing log10(viscosity) greater than 12
    logic = data['log_viscosity'] > 12
    data = data[~logic].copy()

    return data


