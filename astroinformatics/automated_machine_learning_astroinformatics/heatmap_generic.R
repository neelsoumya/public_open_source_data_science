######################################################################
# Generic heatmap in R
#
# Usage: nohup R --no-save < heatmap_generic.R
# OR
# R # then
# source("heatmap_generic.R")
#
# Adapted from:
# https://bioconductor.org/packages/release/bioc/vignettes/ComplexHeatmap/inst/doc/s2.single_heatmap.html
#
# Installation:
# ## try http:// if https:// URLs are not supported
# source("https://bioconductor.org/biocLite.R")
# biocLite("ComplexHeatmap")
#
# install.packages("gplots")
#
######################################################################

library(ComplexHeatmap)
library(circlize)
library(gplots)
library(RColorBrewer)



set.seed(123)

# mat is generic matrix
mat = cbind(rbind(matrix(rnorm(16, -1), 4), matrix(rnorm(32, 1), 8)),
            rbind(matrix(rnorm(24, 1), 4), matrix(rnorm(48, -1), 8)))

# permute the rows and columns
mat = mat[sample(nrow(mat), nrow(mat)), sample(ncol(mat), ncol(mat))]

rownames(mat) = paste0("R", 1:12)
colnames(mat) = paste0("C", 1:10)

Heatmap(mat)






Heatmap(mat_mod2)
draw(Heatmap(mat_mod2, show_column_names = FALSE))

#rnames = mat_mod$gene_name

#install.packages("gplots")
library(gplots)
library(RColorBrewer)

#heatmap.2(mat_mod[,2:255])

matrix_mod2 = data.matrix(mat_mod2)
rownames(matrix_mod2) = mat_mod$gene_name




###############################
# Another heatmap
###############################

#######################
# plot heatmap
#######################

# 1. data must be log transformed before calling heatmap.2
# 2. must be in data.matrix format  before calling heatmap.2
# 3. must have rownames
# 4. mat_mod2 must have only data
# 5. mat_mod is full data frame with gene_name column
# 6. mat_mod and mat_mod2 must have a row for each gene, columns are patients


# no within sample normalization (since RPKM) but log tRANSFORMMED
data_matrix_mat_mod2 = data.matrix(mat_mod2)
log10_data_matrix_mat_mod2 = log10(data_matrix_mat_mod2 + 1)
rownames(log10_data_matrix_mat_mod2) = mat_mod$gene_name

m <- log10_data_matrix_mat_mod2

#Set up a color scale - could also use min/max instead of quantile.
low <- quantile(m,0.05)
high <- quantile(m,0.95)

low <- -2.5
high <- 2.5

nbreaks=200 # number of graduations
breaks=seq(low,high,(high-low)/nbreaks) #range (-2 -> 2) and breakpoint size = range_size / no. graduations
colors=colorRampPalette(c("yellow","black","red"))(nbreaks) # get the color pallete


# Plot a heatmap in which the genes and samples are ordered according
# to the computed clustering
# see ?heatmap.2
heatmap.2(m, 
          scale="row",
          trace="none", 
          col=colors, 
          breaks=breaks, 
          mar=c(10,5))



# explicitly perform colustering outside heatmap to get sampleids
dist_genes <- dist(m, method = "minkowski", p=1.44)
hclust_genes <- hclust(dist_genes, method = "average")
# get the cluster as a dendrogram
den_genes <- as.dendrogram(hclust_genes)

# cluster samples
dist_samples <- dist(t(m), method="minkowski", p=1.44)
hclust_samples <- hclust(dist_samples, method="average")
den_samples <- as.dendrogram(hclust_samples)

# call heatmap with these passed explicitly
heatmap.2(m,
          Rowv = den_genes,
          Colv = den_samples,
          scale = "row",
          trace = "none",
          col=colors,
          breaks = breaks,
          labRow = F,
          mar=c(10,5)
)


# which samples correspond to this cluster
# cut the tree
s_ids <- cutree(hclust_samples, k = 3)
which(s_ids == 2)





fn_generic_heatmap2 <- function(data_only_matrix, full_matrix_with_text, i_data_not_logged)
{
          ##########################################################################################
          # generic function to plot heatmap using heatmap.2
          #
          # arguments:
          # data_only_matrix: raw data NOT long transformed
          # full_matrix_with_text: full data frame (contains raw data NOT long transformed)
          #                           AND a column called gene_name
          # i_data_not_logged: if 1 then data in data_only_matrix and full_matrix_with_text not logged
          #                         the function will internally log10 transform this data
          #                    else no transformation will be done  
          #
          # Installation: 
          # install.packages("gplots")
          #
          # Usage:
          #   data_only_matrix = 2^mat_mod2   # THis is raw data NOT logged BUT ....
          #   full_matrix_with_text = mat_mod # CAUTION data in this is logged
          #   fn_generic_heatmap2(data_only_matrix, full_matrix_with_text, 1)
          ##########################################################################################
  
          require(gplots)
          require(RColorBrewer)
          
          
          
          #######################
          # plot heatmap
          #######################
          
          # 1. data must be log transformed before calling heatmap.2
          # 2. must be in data.matrix format  before calling heatmap.2
          # 3. must have rownames
          # 4. mat_mod2 must have only data
          # 5. mat_mod is full data frame with gene_name column
          # 6. mat_mod and mat_mod2 must have a row for each gene, columns are patients
          
          # no within sample normalization (since RPKM) but log tRANSFORMMED
          # TODO: use dplyr select to do this
          mat_mod = full_matrix_with_text  #df_intersect_signature_with_leuven_UCACTIVE_withpath
          # select all but last which has text
          mat_mod2 = data_only_matrix #df_intersect_signature_with_leuven_UCACTIVE_withpath[,1:ncol(df_intersect_signature_with_leuven_UCACTIVE_withpath)-1]
          
          data_matrix_mat_mod2 = data.matrix(mat_mod2)
          # CAUTION: data being logged here check here if data not already logged
          if (i_data_not_logged == 1)
          {
              log10_data_matrix_mat_mod2 = log10(data_matrix_mat_mod2 + 1)
              rownames(log10_data_matrix_mat_mod2) = mat_mod$gene_name
            
              m <- log10_data_matrix_mat_mod2
          }
          else {

              #log10_data_matrix_mat_mod2 = log10(data_matrix_mat_mod2 + 1)
              rownames(data_matrix_mat_mod2) = mat_mod$gene_name
            
              m <- data_matrix_mat_mod2
            
          }
          
          
          #Set up a color scale - could also use min/max instead of quantile.
          low <- quantile(m,0.05)
          high <- quantile(m,0.95)
          
          low <- -2.5
          high <- 2.5
          
          nbreaks=200 # number of graduations
          breaks=seq(low,high,(high-low)/nbreaks) #range (-2 -> 2) and breakpoint size = range_size / no. graduations
          colors=colorRampPalette(c("yellow","black","red"))(nbreaks) # get the color pallete
          
          
          # Plot a heatmap in which the genes and samples are ordered according
          # to the computed clustering
          # see ?heatmap.2
          heatmap.2(m, 
                    scale="row",
                    trace="none", 
                    col=colors, 
                    breaks=breaks, 
                    mar=c(10,5))
          
          
          
          ########################################################################
          # explicitly perform colustering outside heatmap to get sampleids
          #   clustering with Minkowski distance
          ########################################################################
          dist_genes <- dist(m, method = "minkowski", p=1.44)
          hclust_genes <- hclust(dist_genes, method = "average")
          # get the cluster as a dendrogram
          den_genes <- as.dendrogram(hclust_genes)
          
          # cluster samples
          dist_samples <- dist(t(m), method="minkowski", p=1.44)
          hclust_samples <- hclust(dist_samples, method="average")
          den_samples <- as.dendrogram(hclust_samples)
          
          # call heatmap with these passed explicitly
          heatmap.2(m,
                    Rowv = den_genes,
                    Colv = den_samples,
                    scale = "row",
                    trace = "none",
                    col=colors,
                    breaks = breaks,
                    labRow = F,
                    mar=c(10,5)
          )
          
          
          
          ##################################################
          # Heatmaps based on correlation
          ##################################################
          correlation_matrix <- cor(m, method = "pearson")
          
          heatmap.2(correlation_matrix,
                    trace = "none",
                    density.info = "none",
                    key.title = "Heatmap based on correlation",
                    key.xlab = "Pearson's \n correlation",
                    mar = c(10,10))
          
          
          
  
  
}
