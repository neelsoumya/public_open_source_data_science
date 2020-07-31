#################################################################
# meta analysis poc
#
# Usage: nohup R --no-save < meta_analysis_poc.R
# OR
# R # then
# source("meta_analysis_poc.R")
#
# Installation:
# install.packages("metafor")
#
# Adapted from
# https://cran.r-project.org/web/packages/metafor/index.html
#################################################################



############################################
# load library
############################################
library(metafor)


############################################
# load example data
############################################
data("dat.bcg", package="metafor")
print(dat.bcg, row.names=FALSE)

############################################
# calculate observed outcomes
############################################
dat <- escalc(measure = "RR", ai=tpos, bi=tneg, ci=cpos, di=cneg, data = dat.bcg, append = TRUE)

print(dat[,-c(4:7)], row.names=FALSE)


############################################
# fit random effects model
############################################
res <- rma(yi, vi, data = dat)
confint(res)
res


############################################
# forest plot
############################################
forest(res, slab = paste(dat$author, dat$year, sep = ","),
       xlim=c(-16,6), at = log(c(0.05, 0.25, 1, 4)),
       atransf = exp,
       ilab = cbind(dat$tpos, dat$tneg, dat$cpos, dat$cneg),
       ilab.xpos = c(-9.5, -8, -6, -4.5), cex = 0.75)
op <- par(cex = 0.75, font = 2)
text(c(-9.5, -8, -6, -4.5), 15, c("TB+", "TB-", "TB+", "TB-"))
text(c(-8.75, -5.25), 16, c("Vaccinated", "Control"))
text(-16, 15, "Author(s) and Year", pos = 4)
text(6, 15, "Relative Risk [95% CI]", pos = 2)
par(op)



############################################
# test residuals
############################################
rstudent(res)

inf <- influence(res)
inf
plot(inf, plotdfb=TRUE)


############################################
# cross validation (leave 1 one out)
############################################
res <- rma(yi, vi, data = dat)
leave1out(res, transf = exp, digits = 3)


############################################
# Q-Q plots
############################################
qqplot(res, main="Random effects model")


############################################
# regtest
############################################
regtest(res, model = "lm")


############################################
# funnel plot
############################################
rtf <- trimfill(res)
rtf
funnel(rtf)
