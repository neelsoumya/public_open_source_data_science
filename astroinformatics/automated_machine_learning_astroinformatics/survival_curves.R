################################################################################
# Fitting a Kaplan-Meier and a 
#   Cox proportional hazards model.
#
# survival curves and estimate parameters for a Cox proportional hazards model
#
# adapted from 
# https://stat.ethz.ch/R-manual/R-devel/library/survival/html/survfit.formula.html
################################################################################


########################################
# Load library
########################################
library(survival)

# aml has data like
# > aml
# time status             x
# 1     9      1    Maintained
# 2    13      1    Maintained
# 3    13      0    Maintained


########################################
#fit a Kaplan-Meier and plot it 
########################################
fit <- survfit(Surv(time, status) ~ x, data = aml) 
plot(fit, lty = 2:3) 
legend(100, .8, c("Maintained", "Nonmaintained"), lty = 2:3) 

################################################################################
#fit a Cox proportional hazards model and plot the  
#predicted survival for a 60 year old 
################################################################################
fit <- coxph(Surv(futime, fustat) ~ age, data = ovarian) 
plot(survfit(fit, newdata=data.frame(age=60)),
     xscale=365.25, xlab = "Years", ylab="Survival") 

