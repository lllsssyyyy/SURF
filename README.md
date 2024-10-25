# SURF
A self-supervised deep learning method for reference-free deconvolution

Data input
df_expr: (n_spots * n_genes), dataframe, with column names (gene names)
df_pos: (n_spots * 2), dataframe, with column names (‘x’, ‘y’)
barcodes: (n_spots,), numpy array

Data output
‘pred.csv’: The predicted cell types proportions in each spot.
‘beta.csv’: The deconvolved gene expressions of each cell type.
‘last.pkl’: The saved trained model.
