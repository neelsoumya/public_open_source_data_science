############################################################
# Script to perform data munging on UCI dataset
############################################################


##############################
# Load library
##############################
library(sqldf)


##############################
# Load data
##############################
setwd("~/public_open_source_datascience/deep_learning_basic")

df_monocyte_genes_IGP <- read.csv('breast-cancer-wisconsin_MOD.data', 
                                  sep = ',', header = FALSE, 
                                  stringsAsFactors=FALSE, na.strings="..") # ,strip.white = TRUE)


df_monocyte_genes_IGP_curated = sqldf("select V2, V3, V4, V5, V6, V7, V8, V9, V10, V11
                                      from df_monocyte_genes_IGP")

##############################
# data munging
##############################
# replace 2 by 0 and 4 by 1
df_monocyte_genes_IGP_curated[ which(df_monocyte_genes_IGP_curated$V11 == 2), "V11"] = 0
df_monocyte_genes_IGP_curated[ which(df_monocyte_genes_IGP_curated$V11 == 4), "V11"] = 1

# replace ? by 0
df_monocyte_genes_IGP_curated[which(df_monocyte_genes_IGP_curated == '?', arr.ind = TRUE)] = 0

##############################
# write to disk
##############################
write.table(df_monocyte_genes_IGP_curated, file="breast-cancer-wisconsin_MOD_CURATED.data",
            row.names = FALSE, quote=FALSE, append = FALSE, sep = ",", col.names = FALSE)  #, col.names = NA)
