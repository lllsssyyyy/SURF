# SURF
A self-supervised deep learning method for reference-free deconvolution. The overall approach is detailed in the official paper out in xxx.

![Fig1](https://github.com/user-attachments/assets/cd371dab-fa9a-474d-9bfa-32b41adb8cbe)

# Data input  
df_expr: (n_spots * n_genes), dataframe, with column names (gene names)  
df_pos: (n_spots * 2), dataframe, with column names [‘x’, ‘y’]  
barcodes: (n_spots,), a numpy array  
  
# Data output     
‘pred.csv’: The predicted cell types proportions in each spot.  
‘beta.csv’: The deconvolved gene expressions of each cell type.  
‘last.pkl’: The saved trained model. 

# Installation


