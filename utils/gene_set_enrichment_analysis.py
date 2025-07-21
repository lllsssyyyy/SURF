import os 
import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()


def run_gsea_with_processed_data(df_sorted, species = 'Homo sapiens', category = 'H', subcategory = 'NULL'):
    robjects.globalenv['df_sorted'] = pandas2ri.py2rpy(df_sorted)
    if subcategory is None or subcategory == 'NULL':
        subcategory_code = "geneset <- msigdbr(species = species_name, category = category_name)"
    else:
        subcategory_code = f"geneset <- msigdbr(species = species_name, category = category_name, subcategory = '{subcategory}')"
    
    robjects.globalenv['species_name'] = species
    robjects.globalenv['category_name'] = category
    
    r_code = f"""
    library(clusterProfiler)
    library(msigdbr)
    
    gene_list <- df_sorted$logFC
    names(gene_list) <- df_sorted$SYMBOL
    
    {subcategory_code}
    TERM2GENE <- data.frame(term = geneset$gs_name, gene = geneset$gene_symbol)
    
    gsea_result <- GSEA(
      geneList = gene_list,
      TERM2GENE = TERM2GENE,
      pvalueCutoff = 0.05,
      pAdjustMethod = "BH",
      seed=0
    )
    
    results_df <- gsea_result@result
    """
    
    robjects.r(r_code)
    
    results_df = robjects.r['results_df']
    gsea_result = robjects.r['gsea_result']
    
    return pandas2ri.rpy2py(results_df), gsea_result


def create_gsea_plots_with_r(results_df, gsea_result, plot_dir):
    robjects.globalenv['results_df'] = pandas2ri.py2rpy(results_df)
    robjects.globalenv['gsea_result'] = gsea_result
    robjects.globalenv['plot_dir'] = plot_dir
    os.makedirs(plot_dir, exist_ok=True)
    
    robjects.r("""
    library(ggplot2)
    library(enrichplot)
    library(clusterProfiler)
    """)
    
    robjects.r("""
    p1 <- gseaplot2(gsea_result, 3, color="blue", title="", base_size=13, ES_geom="line")
    """)
    
    robjects.r("""
    p <- ggplot(results_df, aes(x = NES, y = Description)) +
      geom_point(aes(size = setSize, color = p.adjust)) +
      scale_size_continuous(range = c(4,8)) +
      scale_color_gradient(low = "blue", high = "white") +
      labs(title = "Hallmark",
            x = "Normalized Enrichment Score (NES)",
            y = "",
            color = expression(~italic(P)~".adjust"),
            size = "Count") +
      theme_bw() + 
      theme(panel.grid = element_blank(), 
            axis.text.x = element_text(family='Arial', size = 16, color = "black"), 
            axis.text.y = element_text(family='Arial', size = 14, color = "black"), 
            axis.title = element_text(family='Arial', size = 14, color = "black")) +
      theme(plot.title = element_text(family='Arial', vjust = 0.1, hjust = 0.5, size=18, face='bold'),
            legend.title = element_text(family='Arial', size=16), 
            legend.text = element_text(size = rel(0.95)),
            legend.key.size = unit(20, "pt"))
    
    """)
    
    robjects.r("""
    ggsave(plot = p1, filename = file.path(plot_dir, "gsea_enrichment_pathway_plot.jpg"), width = 5, height = 5, dpi = 500)
    ggsave(plot = p, filename = file.path(plot_dir, "gsea_scatter_plot.jpg"), width = 10, height = 5, dpi = 500)
    """)