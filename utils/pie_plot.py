import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
import torch.nn.functional as F
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm
import pandas as pd
import torch


def pie_plot(pred_p, pos_all, loc=0.7):
    df_pred=pred_p
    df_pos = pos_all.iloc[:, [1, 0]]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = [
    '#F9D3BE',
    '#ACB4D2',
    '#F4D27A',
    '#6DB0CC'
    ]
    labels = df_pred.columns.tolist()
    wedgeprops = {'linewidth': 0.05, 'edgecolor': 'black'}
    for (x, y), ratio in zip(df_pos.to_numpy(), df_pred.to_numpy()):
        ax.pie(ratio, center=(x, y), radius=0.4, colors=colors, wedgeprops=wedgeprops)

    ax.set_xlim(0, df_pos.iloc[:, 0].max())
    ax.set_ylim(0, df_pos.iloc[:, 1].max())

    patches = [mpatches.Patch(color=colors[i], label="{:s}".format(str(labels[i]))) for i in range(len(labels))]
    font_prop = fm.FontProperties(family='Arial', size=13)
    legend = plt.legend(handles=patches, ncol=6, fontsize=13, bbox_to_anchor=(loc, -0.1), prop=font_prop,
                        frameon=False, handlelength=1, columnspacing=0.9, labelspacing=0.05)
    for handle in legend.legendHandles:
        handle.set_edgecolor('gray')
        handle.set_linewidth(0.3)

    plt.show()