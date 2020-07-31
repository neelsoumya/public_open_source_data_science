#################################################################
# Generic function for MA plot 
# GENERIC FUNCTION TO PLOT MA PLOT USING GGPLOT
# ACKNOWLEDGEMENTS: STEPHEN SANSOM
#
# Tested on cluster kGEn
# Nick folder
#    /gfs/work/nilott/proj020/analysis/differential_expression3
#
# analysis file nick_LPS_vs_LPSaIL10R_limma.csv generated using
# sqlite3 --header /gfs/work/nilott/proj020/analysis/differential_expression3/csvdb "select * from TreatLPS_TreatLPS_aIL10R_result order by logFC asc"
#################################################################

###################################
# Load libraries
###################################
stopifnot(
  require(DESeq2),
  require(reshape2),
  require(ggplot2),
  require(VennDiagram),
  require(ggrepel)
)


#######################################
# Load data
#######################################
# nick_LPS_vs_LPSaIL10R_limma_padjless005.csv
str_input_filename = "nick_LPS_vs_LPSaIL10R_limma.csv"
res_df = read.csv(str_input_filename, 
                  sep = '|', header = TRUE, stringsAsFactors=FALSE, na.strings="..") # ,strip.white = TRUE)


#######################################
# prettier MA plot using ggplot
#######################################

res_df$gene_name <- res_df$gene_symbol
res_df$baseMean  <- res_df$AveExpr
res_df$log2FoldChange <- res_df$logFC
res_df$padj <- res_df$adj_P_Val

data = res_df
data$color = "default"

# sort by FDR
data = data[order(data$padj),]

# view
head(data)

i_num_display_names = 20
str_plot_save_filename = "maplot_limma_pbmc_microarray_nick"

#############################################################################
# GENERIC FUNCTION TO PLOT MA PLOT USING GGPLOT
# ACKNOWLEDGEMENTS: STEPHEN SANSOM
# Expects a data object sorted on padj
#   with fields: baseMean, padj, log2FoldChange, gene_name
#
# other parameters:
# data - data frame with with fields: baseMean, padj, log2FoldChange, gene_name
#         
# top - top rows of data object whose gene names should be displayed on plot
# i_num_display_names - number of rows to display gene names from data (top)
# str_plot_save_filename - filename where plot should be saved
#############################################################################

fn_ma_ggplot_diff_abundant_genes <- function(data,
                                             top,
                                             i_num_display_names,
                                             str_plot_save_filename)
{
        
        ###################################
        # Load libraries
        ###################################
        stopifnot(require(reshape2),
          require(ggplot2),
          require(ggrepel)
        )
  
  
        data$color = "default"
  
        # sort by FDR
        data = data[order(data$padj),]
  
        top = head(data, n=i_num_display_names) # by -pvalue 
        data$color[abs(data$log2FoldChange)> 1 & data$padj<0.05] <- "> 2 fold, padj < 0.05"
        
        cols <- c("default"="darkgrey", "> 2 fold, padj < 0.05"="red")
        
        gp <- ggplot(data, aes(baseMean, log2FoldChange,color=color)) + geom_point(alpha=0.8)
        gp <- gp + scale_x_continuous(trans="log2")
        #gp <- gp + geom_text(data=top, aes(label=gene_name), vjust=1,hjust=-0.2, color="black")
        gp <- gp + geom_text_repel(data=top, aes(label=gene_name), vjust=1,hjust=-0.2, color="black") #(data=show, aes(label=gene_name))
        gp <- gp + scale_color_manual(values=cols)
        gp <- gp + ggtitle("Limma analysis of PBMCs in microarray (LPS vs. LPS + anti-IL10R) across all donors")
        gp <- gp + xlab("expression (baseMean)") + ylab("fold change (log2)")
        print(gp)
        
        str_maplot_filename = paste0(str_plot_save_filename, ".pdf")
        ggsave(filename = str_maplot_filename, gp, device = "pdf")
        
        
        ####################################
        # Number of significant genes
        ####################################
        
        i_num_sign_padj = sum(res_df$padj < 0.05, na.rm = TRUE)
        cat("Number of genes that are significant at padj alpha = 0.05: ", i_num_sign_padj, "\n")
        # sort by FDR
        #results_deseq <- results_deseq[order(results_deseq$padj),]
        
        
        # make a data frame from the results and show the top 10 up-regulated genes
        #res_df = data.frame(results_deseq)
        cat("All genes with adjusted p-value < 0.05 (sorted by p-value):", "\n")
        #print(head(res_df[res_df$log2FoldChange>log2(2),], n=i_num_sign_padj))
        print(head(res_df, n=i_num_sign_padj))

}

####################################################
# call function
####################################################

fn_ma_ggplot_diff_abundant_genes(data,
                                 top,
                                 i_num_display_names,
                                 str_plot_save_filename)
