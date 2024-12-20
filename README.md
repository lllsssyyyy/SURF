# SURF
A self-supervised deep learning method for reference-free deconvolution. The overall approach is detailed in the official paper out in xxx.

![Fig1](https://github.com/user-attachments/assets/cd371dab-fa9a-474d-9bfa-32b41adb8cbe)

# Data input  
df_expr: (n_spots * n_genes), dataframe, with column names (gene names)  
df_pos: (n_spots * 2), dataframe, with column names [‘x’, ‘y’]  
barcodes: (n_spots,), a numpy array  

# Installation
Before installation, make sure that R environment (https://cran.r-project.org/) and Pytorch (https://pytorch.org/) have been prepared.
We have tested the installation process under ubuntu 22.04, R 4.4.1, and torch2.0.0+cuda117.
1. Install R environment (https://cran.r-project.org/)
2. Create the virtual environment
```
conda create -n SURF python=3.9   
conda activate SURF   
```
3. Install Pytorch (https://pytorch.org/), please choose the suitable torch version according to your cuda version.
4. Install SURF
```
conda create -n SURF python=3.9
pip install spatialsurf
```


