
##################################################################################################
# Name - rscript_sma_allometric_test.R
# Creation Date - 25th January 2016
# Author: Soumya Banerjee
# Website: https://sites.google.com/site/neelsoumya/
#
# Description: 
#     R script for performing SMA analysis of data
#
# Example usage -
#		rscript_sma_allometric_test.R
#
# License - BSD 
#
# Acknowledgements -
#       Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
#				and my wife Joyeeta Ghose.
#
##################################################################################################
  

# Load SMATR library for performing SMA (semi-major axis) regression
library(smatr) 

# Load data from csv files
# Load log transformed data (X)
# assumes X and Y variables are log transformed and stored in separate cs files
log10crime <- read.csv('log10crime.csv')
# Load log transformed data (Y)
log10comm_population <- read.csv('log10comm_population.csv')
# We want to check for relationship X ~ Y^beta where beta is a scaling exponent
# Log transformed logX ~ beta*logY + c

# assign to data frame
df = data.frame(log10crime,log10comm_population)

# check data
head(df)
nrow(df)

# Perform semi-major axis (SMA) regression
sma(formula='log10crime~log10comm', data=df)
sma(formula='log10crime~log10comm', data=df, method='SMA')
