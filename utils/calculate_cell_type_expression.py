import pandas as pd


def calc_cell_type_expr(df_sc, save_dir):
    data_all = []
    cell_type = df_sc.iloc[:, 0].unique()
    t = df_sc.iloc[:, 1:].to_numpy()
    df_sc.iloc[:, 1:] = t / t.sum(axis=1, keepdims=True)
    for cell_type_i in cell_type:
        expr_i = df_sc.loc[df_sc['cell_type'] == cell_type_i].iloc[:, 1:].mean(axis=0)
        data_all.append(expr_i)
    df = pd.concat(data_all, axis=1).T
    df.index = cell_type
    df.to_csv(save_dir + 'sc_based_cell_type_expression.csv')

    return df