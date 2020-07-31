# Fisher's meta-analysis method
# in clutsre open R
# BiocInstaller::biocLite("metap")
#install.packages("metap")
library(metap)

pvals <- c(0.001, 0.001, 0.999, 0.999)
sumlog(pvals)

# plot p-values
# The plot method for class ‘metap’ calls schweder on the valid p-values.
# Inspection of the distribution of p-values is highly recommended as 
# extreme values in opposite directions do not cancel out. See last example. This may not be what you want.

schweder(pvals)


data(teachexpect)
sumlog(teachexpect) # chisq = 69.473, df = 38, p = 0.0014, from Becker
#chisq =  69.47328  with df =  38  p =  0.001369431 
schweder(teachexpect)
