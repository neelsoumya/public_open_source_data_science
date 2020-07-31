############################################################################
# Supervised PCA example.
#
# Supervised principal components for regression and
# survival analysis for microarray data.
#
# Adapted from
#   http://statweb.stanford.edu/~tibs/superpc/tutorial.html
#
# Installation:
#   install.packages("superpc")
#
############################################################################


#######################################################
# Load library
#######################################################
library(superpc)
library(survival)

#######################################################
# Generate synthetic survival data.
#
# there are 1000 features and 60 samples
#  the outcome is highly correlated with the first principal component of the
#  first 80 features
#######################################################

x<-matrix(rnorm(1000*100),ncol=100)
v1<- svd(x[1:80,])$v[,1]

y<-2+5*v1+ .05*rnorm(100)

xtest<-x
ytest<-2+5*v1+ .05*rnorm(100)
censoring.status<- sample(c(rep(1,80),rep(0,20)))
censoring.status.test<- sample(c(rep(1,80),rep(0,20)))

featurenames <- paste("feature",as.character(1:1000),sep="")


###################################################################################################
# TODO: x is X in RISK cohort (microarray normalized and feature scaled log or log not?)
#       y is time to steroids, time to infliximab
###################################################################################################



##################################################################################
# create train and test data objects. censoring.status=1 means the event occurred;
#  censoring.status=0 means censored
##################################################################################
data<-list(x=x, y=y, censoring.status=censoring.status, featurenames=featurenames)
data.test<-list(x=xtest,y=ytest, 
                censoring.status=censoring.status.test, 
                featurenames=featurenames)

##################################################################################
# train the model. This step just computes the  scores for each feature
##################################################################################
train.obj<- superpc.train(data, type="survival")

# note for regression (non-survival) data, we leave the component "censoring.status"
# out of the data object, and call superpc.train with type="regression".
# otherwise the superpc commands are all the same


##################################################################################
# cross-validate the model
##################################################################################
cv.obj<-superpc.cv(train.obj, data)

##################################################################################
# plot the cross-validation curves. From this plot we see that the 1st 
# principal component is significant and the best threshold  is around 0.7
##################################################################################
superpc.plotcv(cv.obj)


##################################################################################
# here we have the luxury of  test data, so we can compute the  likelihood ratio statistic
# over the test data and plot them. We see that the threshold of 0.7
# works pretty well
##################################################################################
lrtest.obj<-superpc.lrtest.curv(train.obj, data,data.test)

superpc.plot.lrtest(lrtest.obj)


##################################################################################
# now we derive the predictor of survival for the test data, 
# and then then use it
# as the predictor in a Cox model . We see that the 1st supervised PC is
# highly significant; the next two are not
##################################################################################
fit.cts<- superpc.predict(train.obj, data, data.test, threshold=0.7, n.components=3, prediction.type="continuous")

superpc.fit.to.outcome(train.obj, data.test, fit.cts$v.pred)


##################################################################################
# sometimes a discrete (categorical) predictor is attractive.
# Here we form two groups by cutting the predictor at its median
# and then plot Kaplan-Meier curves for the two groups
##################################################################################
fit.groups<- superpc.predict(train.obj, data, data.test, threshold=0.7, n.components=1, prediction.type="discrete")

superpc.fit.to.outcome(train.obj, data.test, fit.groups$v.pred)

plot(survfit(Surv(data.test$y,data.test$censoring.status)~fit.groups$v.pred), col=2:3, xlab="time", ylab="Prob survival")


##################################################################################
# Finally, we look for a predictor of survival a small number of
# genes (rather than all 1000 genes). We do this by computing an importance
# score for each equal its correlation with the supervised PC predictor.
# Then we soft threshold the importance scores, and use the shrunken
# scores as gene weights to from a reduced predictor. Cross-validation
# gives us an estimate of the best amount to shrink and an idea of
# how well the shrunken predictor works.
##################################################################################
fit.red<- superpc.predict.red(train.obj, data, data.test, threshold=0.7)

fit.redcv<- superpc.predict.red.cv(fit.red, cv.obj,  data,  threshold=0.7)

superpc.plotred.lrtest(fit.redcv)

##################################################################################
# Finally we list the significant genes, in order of decreasing importance score
##################################################################################
superpc.listfeatures(data.test, train.obj, fit.red) #, 1, shrinkage=0.17)

##################################################################################
# A note on interpretation:
#  The signs of the scores (latent factors) v.pred returned by superpc.predict are chosen so that the regression of the outcome on each factor has a positive coefficient. This is just a convention to aid in interpretation.
# For survival data, this means
# Higher score => higher risk (worse survival)

# For regression data, this means
# Higher score => higher mean of the outcome

# How about the direction of effect for each individual feature (gene)? The function superpc.listfeatures reports an importance score equal to the correlation between each feature and the latent factor.

# Hence for survival data,

# Importance score positive means
# increase in value of feature=> higher risk (worse survival)

# For regression data,

# Importance score positive means
# increase in value of feature => higher mean of the outcome

# The reverse are true for Importance score negative. 
##################################################################################
