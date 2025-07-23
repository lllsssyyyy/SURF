# SURF
A self-supervised deep learning method for reference-free deconvolution. The overall approach is detailed in the official paper out in xxx.

![Fig1](https://github.com/user-attachments/assets/45018ff0-2680-4ed5-9e09-3616b60f73cb)

# Data input  
**df_expr**: (dataframe), column names: gene names, shape: (n_spots, n_genes). The gene expression of ST data.  
**df_pos**: (dataframe), column names: ‘x’, ‘y’, shape: (n_spots, 2). The position data of ST data.  
**barcodes**: (list), len: n_spots. The barcodes of ST data.  

# Installation
We have tested the installation process under ubuntu 22.04, R 4.5.1, and torch 2.1.1 + cuda 12.1.
1. Install R environment (https://cran.r-project.org/).
   Please install R packages such as clusterProfiler, mgcv, Matrix, msigdbr, CellChat, presto, etc. to ensure all functions in the code will work.
3. Create the virtual environment
```
conda create -n SURF python=3.9   
conda activate SURF   
```
3. Install Pytorch (https://pytorch.org/), **please choose the suitable torch version according to your cuda version**.
```
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121 
```
**Note**: The installation command shown above is suitable for our cuda version and is provided as an example only. Please refer to the instructions at [https://pytorch.org/get-started/previous-versions/] to find the installation command appropriate for your cuda version.

4. Install SURF
```
pip install spatialsurf
```
# Tutorials
https://github.com/lllsssyyyy/SURF/tree/main/tutorials


