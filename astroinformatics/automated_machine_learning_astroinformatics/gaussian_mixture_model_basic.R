##################################################################################
# Example code for Gaussian mixtures modelling
#
# Usage:
#     R --no-save < gaussian_mixture_model_basic.R 
#
# Adapted from
#   https://cran.r-project.org/web/packages/sBIC/vignettes/GaussianMixtures.pdf
#
##################################################################################

#####################
# Load library
#####################
library(mclust)

#####################
# Load data
#####################
library(MASS)
data(galaxies)
X = galaxies / 1000

#####################
# Fit mixture model
#####################
fit = Mclust(X, G=4, model="V")
summary(fit)

#####################
# Plot distribution
#####################
plot(fit, what="density", main="", xlab="Velocity (Mm/s)")
rug(X)

plot(fit, what="classification", main="", xlab="Velocity (Mm/s)")
rug(X)


##############################
# Perform model selection
##############################
#     1. training and validation
#     2. iterate over upto n Gaussian (G = n)
#     3. pick best from validation set

# TODO:
# For example
fit = Mclust(X, G=10, model="V")
summary(fit)
