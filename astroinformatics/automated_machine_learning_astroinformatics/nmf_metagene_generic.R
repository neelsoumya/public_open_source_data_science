####################################################################
# Non-negative matrix factorization of gene abundance data
#
# Installation:
# install.packages("NMF")
#
####################################################################

##################################
# Load library
##################################
library(NMF)

####################################################################
# generate a synthetic dataset with known classes
####################################################################
n <- 50; counts <- c(5, 5, 8);
V <- syntheticNMF(n, counts)
V

####################################################################
# perform a 3-rank NMF using the default algorithm
####################################################################
res <- nmf(V, 3)

basismap(res)
coefmap(res)

####################################################################
# reduced matrix
####################################################################
View(res@fit@H)

# see paper
# Metagenes and molecular pattern discovery using matrix factorization, PNAS, 2003


####################################################################
# Load data
####################################################################
setwd("~/periphery_project/machine_learning/COMBINED_PIPELINE_IBD osm_cohort_gse12251")

V_raw = read.csv('FINAL_intersected_genes_cohort_gse12251_osm_MOD.csv', sep = ',', 
                 header = TRUE, stringsAsFactors=FALSE,
                 na.strings="..") # ,strip.white = TRUE)

View(V_raw)

####################################################################
# get non-negative form of this no negative vqalues so 2^
# and transpose since rows are genes (see Fig. 1 PNAS paper)
####################################################################
V_data_matrix = 2^t( data.matrix(V_raw[,2:664]) )
View(V_data_matrix)

# perform a 2-rank NMF using the default algorithm
res_raw <- nmf(V_data_matrix, 2)

basismap(res_raw)
coefmap(res_raw)

####################################################################
# reduced matrix
####################################################################
View(res_raw@fit@H)

reduced_matrix = res_raw@fit@H

first_metagene = reduced_matrix[1,]
second_metagene = reduced_matrix[2,]

final_severity_metagene = colSums(reduced_matrix)
View(final_severity_metagene)

####################################################################
# save to file
####################################################################
write.csv(final_severity_metagene, file = "file_final_severity_metagene.csv",
          row.names = FALSE, quote=FALSE) # if no row names and no quote around characters

