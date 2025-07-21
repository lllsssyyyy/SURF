import os
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()


def run_cellchat(count_matrix, pos_data, barcodes, meta, scale_data_dir, plot_dir, contact_range=100, spot_size=65, species='human'):
    robjects.globalenv['count_matrix'] = count_matrix
    robjects.globalenv['pos_data'] = pos_data
    robjects.globalenv['barcodes'] = barcodes
    robjects.globalenv['meta'] = meta
    robjects.globalenv['scale_data_dir'] = scale_data_dir
    robjects.globalenv['plot_dir'] = plot_dir
    robjects.globalenv['contact_range'] = contact_range
    robjects.globalenv['spot.size'] = spot_size
    robjects.globalenv['species'] = species
    
    os.makedirs(plot_dir, exist_ok=True)

    r_code = """
    library(CellChat)
    library(patchwork)
    library(Seurat)
    library(Matrix)
    options(stringsAsFactors = FALSE)
    
    print('Preprocess data...')
    scalefactors <- jsonlite::fromJSON(txt = file.path(scale_data_dir, 'scalefactors_json.json'))
    meta$samples <- factor(meta$samples)
    rownames(count_matrix) <- barcodes
    pos_data$barcode <- barcodes
    count_matrix <- t(count_matrix)
    rownames(pos_data) <- pos_data$barcode

    count_matrix <- count_matrix[, rownames(meta)]
    pos_data <- pos_data[rownames(meta), ]

    data.input <- normalizeData(count_matrix)
    spatial.locs <- pos_data[, c("x_img", "y_img")]
    colnames(spatial.locs) <- c("imagerow", "imagecol")

    conversion.factor = spot.size/scalefactors$spot_diameter_fullres
    spatial.factors = data.frame(ratio=conversion.factor, tol=spot.size/2)
    d.spatial <- computeCellDistance(coordinates=spatial.locs, ratio=spatial.factors$ratio, tol=spatial.factors$tol)
    
    if (species == "human") {
        CellChatDB <- CellChatDB.human
    } else if (species == "mouse") {
        CellChatDB <- CellChatDB.mouse
    } else {
        stop("Unsupported species. Please use 'human' or 'mouse'")
    }
   
    print('Create Cellchat data...')
    cellchat <- createCellChat(object=data.input, meta=meta, group.by="labels", datatype="spatial", coordinates=spatial.locs, spatial.factors=spatial.factors)
    cellchat@DB <- CellChatDB

    cellchat <- subsetData(cellchat)
    future::plan("multisession", workers=4)
    cellchat <- identifyOverExpressedGenes(cellchat)
    cellchat <- identifyOverExpressedInteractions(cellchat, variable.both=F)
    
    print('Compute communication probabilities...')
    cellchat <- computeCommunProb(cellchat, type="truncatedMean", trim=0.1, distance.use=TRUE, interaction.range=250, scale.distance=0.01, contact.dependent=TRUE, contact.range=contact_range)
    cellchat <- filterCommunication(cellchat, min.cells=20)
    df.net <- subsetCommunication(cellchat)
    cellchat <- computeCommunProbPathway(cellchat)
    cellchat <- aggregateNet(cellchat)

    groupSize <- as.numeric(table(cellchat@idents))
    
    print('Plot...')
    png(file.path(plot_dir, 'interaction_circle.jpg'), width=2000, height=2000, res=300)
    par(mfrow = c(1,1), xpd=TRUE)
    netVisual_circle(cellchat@net$weight,
        vertex.weight = rowSums(cellchat@net$weight),
        weight.scale = TRUE, label.edge = FALSE,
        title.name = 'Interaction weights/strength')
    dev.off()
    
   
    png(file.path(plot_dir, 'CLDN_interaction_circle.jpg'), width=2000, height=2000, res=300)
    pathways.show <- c("CLDN")
    par(mfrow=c(1,2), xpd = TRUE)
    netVisual_aggregate(cellchat, signaling = pathways.show, layout = "circle", sources.use = c(1, 2, 4), targets.use = c(5, 6))
    netVisual_aggregate(cellchat, signaling = pathways.show, layout = "circle", sources.use = c(5, 6), targets.use = c(1, 2, 4))
    dev.off()
    
    
    png(file.path(plot_dir, 'CypA_interaction_circle.jpg'), width=2000, height=2000, res=300)
    pathways.show <- c("CypA")
    par(mfrow=c(1,2), xpd = TRUE)
    netVisual_aggregate(cellchat, signaling = pathways.show, layout = "circle", sources.use = c(1, 2, 4), targets.use = c(5, 6))
    netVisual_aggregate(cellchat, signaling = pathways.show, layout = "circle", sources.use = c(5, 6), targets.use = c(1, 2, 4))
    dev.off()
    """
    robjects.r('options(device = "png")')
    robjects.r('options(width=100)')
    robjects.r(r_code)