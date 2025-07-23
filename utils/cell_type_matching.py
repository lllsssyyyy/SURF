import numpy as np
import pandas as pd
from scipy.stats import pearsonr


def cell_type_matching(df_sc, df_beta):
    gt_cell_type_num = df_sc.shape[0]
    pred_cell_type_num = df_beta.shape[0]
    cell_type_gt = df_sc.index.tolist()

    # calculate correlation between gt and pred
    overlapping_genes = list(set(df_sc.columns.tolist()) & set(df_beta.columns.tolist()))
    cell_type_correlation = np.zeros([gt_cell_type_num, pred_cell_type_num])
    df_sc = df_sc.loc[:, overlapping_genes]
    df_beta = df_beta.loc[:, overlapping_genes]
    df_sc /= (df_sc.to_numpy().sum(axis=1, keepdims=True))
    df_beta /= (df_beta.to_numpy().sum(axis=1, keepdims=True))
    df_sc /= (df_sc.sum(axis=0) + 1e-6)
    df_beta /= (df_beta.sum(axis=0) + 1e-6)
    for i in range(gt_cell_type_num):
        sc_expr = df_sc.iloc[i, :].to_numpy()
        for j in range(pred_cell_type_num):
            pred_expr = df_beta.iloc[j, :].to_numpy()
            cell_type_correlation[i, j] = pearsonr(sc_expr, pred_expr)[0]
    df_corr = pd.DataFrame(cell_type_correlation, index=cell_type_gt, columns=['X' + str(k) for k in range(pred_cell_type_num)])

    # match gt and pred
    cell_type_score_copy = cell_type_correlation.copy()
    df_matching_results = pd.DataFrame(index=cell_type_gt, columns=['cell_type_index', 'corr'])
    for i in range(gt_cell_type_num):
        max_value = np.max(cell_type_score_copy)
        if max_value == -100:
            break
        max_indices = np.transpose(np.where(cell_type_score_copy == max_value))
        for row_index, col_index in max_indices:
            df_matching_results.iloc[row_index, 0] = col_index
            df_matching_results.iloc[row_index, 1] = max_value
            cell_type_score_copy[row_index] = -100
            cell_type_score_copy[:, col_index] = -100

    return df_matching_results, df_corr