#############################################################################
# Chi squared and Fisher's exact test example
#
# Usage:
#   R --no-save < chi_squared_and_fisher_exact_test.R
#
# Adapted from:
# https://stat.ethz.ch/R-manual/R-devel/library/stats/html/chisq.test.html
# https://www.r-bloggers.com/chi-squared-test/
# https://stat.ethz.ch/R-manual/R-devel/library/stats/html/fisher.test.html
#############################################################################


M <- as.table(rbind(c(762, 327, 468), c(484, 239, 477)))

dimnames(M) <- list(gender = c("F", "M"),
                    party = c("Democrat","Independent", "Republican"))

#View(M)

(Xsq <- chisq.test(M))  # Prints test summary
Xsq$observed   # observed counts (same as M)
Xsq$expected   # expected counts under the null
Xsq$residuals  # Pearson residuals
Xsq$stdres     # standardized residuals



# Fisher's exact test

fisher.test(M)
