# SURF
A self-supervised deep learning method for reference-free deconvolution. The overall approach is detailed in the official paper out in xxx.

![Fig1](https://github.com/user-attachments/assets/ffda1ac8-4649-4da3-ac56-f58a1b5fe08e)

# Data input  
df_expr: (dataframe), column names: gene names, shape: (n_spots, n_genes). The gene expression of ST data.
df_pos: (dataframe), column names: ‘x’, ‘y’, shape: (n_spots, 2). The position data of ST data.
barcodes: (list), len: n_spots. The barcodes of ST data.

# Installation
Before installation, make sure that R environment (https://cran.r-project.org/) and Pytorch (https://pytorch.org/) have been prepared.
We have tested the installation process under ubuntu 22.04, R 4.4.1, and torch2.0.0+cuda117.
1. Install R environment (https://cran.r-project.org/)
2. Create the virtual environment
```
conda create -n SURF python=3.9   
conda activate SURF   
```
3. Install Pytorch ([https://pytorch.org/get-started/previous-versions/]), please choose the suitable torch version according to your cuda version.
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113   
```
Note: The installation command shown above is suitable for our cuda version and is provided as an example only. Please refer to the instructions at [https://pytorch.org/get-started/previous-versions/] to find the installation command appropriate for your cuda version.
4. Install SURF
```
pip install spatialsurf
```


