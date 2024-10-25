from matplotlib import pyplot as plt
import numpy as np
import math


def celltype_visualization_ST(pred_p, pos_all, col_num,plot=True):
    """
    pred_p: dataframe, spot * cell_type_num
    pos_all: dataframe, spot * 2
    """
    x_num = pos_all.iloc[:, 0].max() - pos_all.iloc[:, 0].min() + 1
    y_num = pos_all.iloc[:, 1].max() - pos_all.iloc[:, 1].min() + 1
    cell_type_num = pred_p.shape[1]
    cell_type_pred_map = np.zeros([x_num, y_num, cell_type_num])
    for i in range(len(pos_all)):
        row_i = int(pos_all.iloc[i, 0])
        col_i = int(pos_all.iloc[i, 1])
        cell_type_pred_map[row_i, col_i, :] = pred_p.iloc[i, :]

    row_num = math.ceil(cell_type_num / col_num)
    if row_num == 1:
        row_num = 2
    fig, ax = plt.subplots(row_num, col_num, figsize=(8, 8))

    for i in range(row_num):
        for j in range(col_num):
            ax[i, j].axis('off')
            ax[i, j].set_aspect('equal')

    for i in range(pred_p.shape[1]):
        row_i = i // col_num
        col_i = i - row_i * col_num
        im = ax[row_i, col_i].scatter(pos_all.iloc[:, 0], pos_all.iloc[:, 1], s=20, c=pred_p.iloc[:, i], marker='.', cmap='magma')
        plt.colorbar(im, ax=ax[row_i, col_i], cmap='magma')
        ax[row_i, col_i].set_title(pred_p.columns[i], fontdict={'fontsize': 20, 'fontname': 'Arial'})
        ax[row_i, col_i].set_aspect(np.sqrt(3))

    if plot:
        plt.show()


def celltype_visualization_visium(pred_p, pos_all, col_num):
    """
    pred_p: dataframe, spot * cell_type_num
    pos_all: dataframe, spot * 2
    """

    pos_all = pos_all.iloc[:, [1, 0]]
    pos_all.iloc[:, 1] = pos_all.iloc[:, 1].max() - pos_all.iloc[:, 1]

    cell_type_num = pred_p.shape[1]
    row_num = math.ceil(cell_type_num / col_num)
    if row_num == 1:
        row_num = 2
    fig, ax = plt.subplots(row_num, col_num, figsize=(30, 12))

    for i in range(row_num):
        for j in range(col_num):
            ax[i, j].axis('off')
            ax[i, j].set_aspect('equal')

    for i in range(pred_p.shape[1]):
        row_i = i // col_num
        col_i = i - row_i * col_num
        im = ax[row_i, col_i].scatter(pos_all.iloc[:, 0], pos_all.iloc[:, 1], s=20, c=pred_p.iloc[:, i], marker='.', cmap='magma')
        plt.colorbar(im, ax=ax[row_i, col_i], cmap='magma')
        ax[row_i, col_i].set_title(f'X{pred_p.columns[i]}', fontdict={'fontsize': 40, 'fontname': 'Arial'})
        ax[row_i, col_i].set_aspect(np.sqrt(3))
    plt.show()


def visualize_lineplot(cell_type_num, index, rare_cell_type_num):
    fig, ax1 = plt.subplots(figsize=(3, 2))
    ax1.plot(cell_type_num, index, color='#416FB9')
    ax1.set_xlabel('K', fontname='Arial', fontsize=9)
    ax1.set_ylabel('Pereplexity', color='#416FB9', fontsize=9)
    ax1.tick_params('y', colors='#416FB9')
    ax1.tick_params(axis='both', which='both', labelsize=6)

    ax2 = ax1.twinx()
    ax2.plot(cell_type_num, rare_cell_type_num, color='#948434')
    ax2.set_ylabel('Rare cell type number', color='#948434', fontsize=9)
    ax2.tick_params('y', colors='#948434')
    ax2.tick_params(axis='both', which='both', labelsize=6)

    plt.show()