import scanpy as sc
import pandas as pd
import numpy as np
import anndata as ad
import matplotlib.pyplot as plt
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula
from rpy2.robjects.packages import importr
import math
mgcv = importr('mgcv')
stats = robjects.packages.importr('stats')


def bh_adjust(x, log=False):
    nai = np.where(~np.isnan(x))[0]
    ox = x.copy()
    x = x[nai]
    id = np.argsort(x)
    if log:
        q = x[id] + np.log(len(x) / np.arange(1, len(x) + 1))
    else:
        q = x[id] * len(x) / np.arange(1, len(x) + 1)
    a = np.minimum.accumulate(q[::-1])[::-1][np.argsort(id)]
    ox[nai] = a
    return ox


def find_overdispersed_genes(mat, gam_k=5, alpha=0.05, max_gene=1000):
    print('Gene filtering begin...')
    dfm = np.log(mat.mean(axis=0))
    dfv = np.log(mat.var(axis=0))
    df = pd.DataFrame({'m': dfm, 'v': dfv}, index=mat.columns)
    vi = np.isfinite(dfv)

    # Run R code in Python
    pandas2ri.activate()
    r_dataframe_vi = pandas2ri.py2rpy(df[vi])
    if len(vi) < gam_k * 1.5:
        gam_k = 1
    if gam_k < 2:
        m = stats.lm('v ~ m', data=r_dataframe_vi)
    else:
        fm = Formula('v ~ s(m, k = {})'.format(gam_k))
        m = mgcv.gam(fm, data=r_dataframe_vi)
    df['res'] = -np.inf
    df.loc[vi, 'res'] = stats.resid(m, type='response')

    df_res = pandas2ri.py2rpy(np.exp(df['res']))
    n_obs = mat.shape[0]

    lp = stats.pf(df_res, n_obs, n_obs, lower_tail=False, log_p=True)
    lpa = bh_adjust(lp, log=True)
    df['lp'] = lp
    df['lpa'] = lpa

    ods = df['lpa'] < np.log(alpha)
    print(f"{ods.sum()} overdispersed genes remaining.")

    if ods.sum() > max_gene:
        thres = sorted(df['lpa'].tolist())[max_gene]
        ods = df['lpa'] < thres
        print(f"select the top {ods.sum()}  most overdispersed genes by default.")
    df['ods'] = ods
    return df


def spatial_pp(df_expr, df_pos, barcodes, save_dir, min_counts=101, min_cells=None, max_gene_number=1000, filter_genes=True):
    """
    :param df_expr: (n_spots * n_genes), dataframe, with column names(gene names)
    :param df_pos: (n_spots * 2), dataframe
    :param barcodes: (n_spots,), numpy array
    :param save_dir: str
    """

    data_X = df_expr.to_numpy()
    data_obs = pd.DataFrame()
    data_obs['spot'] = barcodes
    data_obs['x'] = df_pos.iloc[:, 0]
    data_obs['y'] = df_pos.iloc[:, 1]
    data_var = pd.DataFrame(index=df_expr.columns)
    adata = ad.AnnData(data_X, obs=data_obs, var=data_var)

    print('Data preprocessing begin...')
    sc.pp.filter_cells(adata, min_counts=min_counts)
    print(f'Spots with no more than {min_counts - 1} gene counts are filtered out.')

    if filter_genes:
        sc.pp.filter_genes(adata, max_cells=len(adata.X) - 1)
        print('Genes present in all the spots are filtered out.')
        if min_cells == None:
            sc.pp.filter_genes(adata, min_cells=math.floor(0.05 * len(adata.X) + 1))
            print('Genes detected in fewer than 5% of spots are filtered out.')
        else:
            sc.pp.filter_genes(adata, min_cells=min_cells)
            print(f'Genes detected in fewer than {min_cells} spots are filtered out.')
        # find overdispersed genes
        mat = pd.DataFrame(adata.X, index=adata.obs['spot'], columns=adata.var.index)
        df = find_overdispersed_genes(mat, gam_k=5, alpha=0.05, max_gene=max_gene_number)
        data_index = np.nonzero((df['ods'] == 1).values)[0]
        data_select = adata.X[:, data_index]
        col_name = np.array(adata.var.index[df['ods']])
    else:
        data_select = adata.X
        col_name = adata.var.index.to_numpy()

    expr = pd.DataFrame(data_select, columns=col_name)
    id = pd.DataFrame(adata.obs['spot'].to_numpy(), columns=['spot_id'])
    df_pos_select = pd.DataFrame()
    df_pos_select['x'] = adata.obs['x']
    df_pos_select['y'] = adata.obs['y']
    df_pos_select.reset_index(drop=True, inplace=True)
    transcript_all = pd.concat([id, df_pos_select, expr], axis=1)

    # remove spots without gene expressions
    drop_index = []
    for i in range(len(transcript_all)):
        t = (transcript_all.iloc[i, 3:] > 0).sum()
        if t == 0:
            print([i, t])
            drop_index.append(i)
    transcript_all = transcript_all.drop(drop_index).reset_index(drop=True)
    transcript_all.to_csv(save_dir + '/spot_pos_and_expr.csv', index=False)

    print('\nData preprocessing end.')
    return transcript_all

