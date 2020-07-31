###############################################################################################
# Generic function for performing PCA on raw data
#
#   pca_generic.R
#
###############################################################################################


###############################################################################################
# Load library
###############################################################################################
library(ggplot2)
library(stats)


###############################################################################################
# PCA of raw data
###############################################################################################

###############################################################################################
# filtering of rawdata (if required)
###############################################################################################
summary(as.vector(rawdata))
dim(rawdata)

i_max_threshold_filter_row = 10
idx2 <- apply(rawdata, 1, max) > i_max_threshold_filter_row
rawdata <- rawdata[idx2, ]


###############################################################################################
# PCA analysis on raw data and transformations (if required: look at data first)
###############################################################################################
pca = prcomp(rawdata)
#pca = prcomp(log10(rawdata))
#pca = prcomp(log10(rawdata + 1))  
#pca = prcomp(log10(rawdata), scale = TRUE) 
#pca = prcomp(rawdata, scale. = TRUE)  


# get rotation
d <- as.data.frame(pca$rotation)

###############################################################################################
# PCA plot (not pretty)
###############################################################################################
plot(d$PC1,d$PC2)


###############################################################################################
# generate scree plot
###############################################################################################
head(pca$sdev)

(summary(pca))$importance["Proportion of Variance",]

plot( (summary(pca))$importance["Proportion of Variance",], 
      xlab="component",
      ylab="proportion of variance",
      main="scree plot")


###############################################################################################
# what are top loading genes
###############################################################################################
head(pca$x[order(pca$x[,1]),])
# order(pca$x[,1])
# idx = order(pca$x[,1]) # get indices of sorted array
# pca$x[idx,] # feed these indices into array to get sorted
# head(pca$x[idx,]) # head of that

#head(pca$x[order(pca$x[,1], decreasing = TRUE ),] , n = 20 )
select(x = hta20transcriptcluster.db, keys = c("TC04001102.hg.1"), columns = c("GENENAME"))
select(x = hta20transcriptcluster.db, keys = c("PSR10011618.hg.1"), columns = c("GENENAME"))


###############################################################################################
# Prettier plots using ggplot
###############################################################################################

# add a column for sample name OR stimulation condition etc (NOTE: d is a dataframe)
d$sample = rownames(d)

# add a column in the dataframe for stimulation condition
# join by column name with original metadata
d$stim = sqldf("select file_all_target_mappings_orig_LPS_LPSaIL10.STIMULATION as stim 
               from file_all_target_mappings_orig_LPS_LPSaIL10 
               inner join d 
               on file_all_target_mappings_orig_LPS_LPSaIL10.SAMPLE = d.sample")


# ggplot magic to plot with labels
gp <-  ggplot(d$stim, aes(d$PC1, d$PC2, color=stim)) + geom_point(size=5)
print(gp)

