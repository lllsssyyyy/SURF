import pandas as pd
import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def replace_top_k(column):
    sorted_column = column.sort_values(ascending=False).reset_index(drop=True)
    if len(sorted_column) > 10:
        eleventh_value = sorted_column.iloc[10]  
        top_10_indices = column.nlargest(10).index
        column.loc[top_10_indices] = eleventh_value
    return column


def calculate_metagenes(df_data, df_beta, topk1=5, topk2=10):
    df_marker = pd.DataFrame()
    df_logFC = pd.DataFrame()
    top_k1_marker_all = []
    for i in range(len(df_beta)):
        num = i
        df_beta_i = df_beta.iloc[i, :]
        sel = list(set(range(df_beta.shape[0])) - set([i]))
        df_sel = df_beta.iloc[sel, :]
        fc = df_beta.iloc[num, :] / df_sel.mean(axis=0)
        log2fc = fc.apply(lambda x: math.log2(x))
        sort = log2fc.sort_values(ascending=False)
        df_marker[f'{i}'] = sort.iloc[:topk2].index
        df_logFC[f'{i}'] = log2fc
        top_k1_marker_all += sort.iloc[:topk1].index.tolist()

    df_metagene = pd.DataFrame(np.zeros([len(df_data), len(df_beta)]), columns=[f'{i}' for i in range(len(df_beta))])
    df_expr = df_data.iloc[:, 3:]
    df_expr = df_expr.apply(replace_top_k)
    df_expr = df_expr.apply(lambda col: col / col.max() if col.max() != 0 else col)
    for i in range(len(df_beta)):
        marker_i_list = df_marker.iloc[:, i].tolist()
        df_metagene[f'{i}'] = df_expr.loc[:, marker_i_list].mean(axis=1)
        df_metagene[f'{i}'] = df_metagene[f'{i}'] / df_metagene[f'{i}'].max()
        
    return df_logFC, top_k1_marker_all, df_metagene 


def metagene_plot(df_logFC, df_beta, top_k1_marker_all, sel_celltype, topk1=5):
    top_marker_all = []
    
    for celltype in sel_celltype:
        start_idx = celltype * topk1
        end_idx = (celltype + 1) * topk1

        celltype_data = top_k1_marker_all[start_idx:end_idx]
        top_marker_all.extend(celltype_data)
        
    df_top_FC = df_logFC.loc[top_marker_all]
    df_top_FC = df_top_FC.iloc[:, sel_celltype]
    df_top_beta = df_beta.T.loc[top_marker_all, sel_celltype]
    FC_arr = df_top_FC.T.values.flatten()
    beta_arr = df_top_beta.T.values.flatten()
    beta_arr /= beta_arr.max()
    x_label = []
    y_label = []
    for i in range(len(df_top_FC.columns.tolist())):
        for j in range(len(df_top_FC.index.tolist())):
            x_label.append(f'X{sel_celltype[i]}')
            y_label.append(df_top_FC.index.tolist()[j])

    fig, ax = plt.subplots(figsize=(4, 5))
    plt.subplots_adjust(left=0.25, right=0.9, top=0.98, bottom=0.15)
    im = ax.scatter(x_label, y_label, s=FC_arr * 30 * 2, c=beta_arr, marker='.', cmap='plasma')
    im.set_clim(0, 1)
    plt.xticks(fontsize=10, fontname='Arial')
    plt.yticks(fontsize=10, fontstyle='italic', fontname='Arial')
    cbar = plt.colorbar(im, ax=ax, cmap='plasma', location='top', pad=0.01)
    cbar.set_label('Deconvolved Expressions', labelpad=5, fontsize=10, fontfamily='Arial')
    cbar.ax.tick_params(labelsize=7)
    legend_sizes = [40, 80, 120]
    legend_labels = ['2', '4', '6']
    handles = [plt.scatter([], [], s=size, edgecolor='black', color='gray') for size in legend_sizes]
    legend = plt.legend(handles, legend_labels, title="Log$_{2}$fold_change", title_fontsize=10, frameon=False,
                        loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=3, prop={'family': 'Arial', 'size': 10})
    plt.show()