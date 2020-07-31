readme

Predict cancer survival using logistic regression (open source project)

Usage:  matlab < predict_mortality.m

1) haberman.data
	data from UCI machine learning repository
	https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival

2) haberman.names
	metadata from UCI machine learning repository
	https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival

3) predict_mortality.m
	Matlab code to load data, perform data munging, perform logistic regression
	and predict survival

4) lassoplot_1.eps and lassoplot_2.eps
	LASSO plots showing effect of varying lambda and values of lanbda from 
	cross-validation


5) generic_lasso_logistic.m
	Generic function to perform LASSO on logistic regression and predict


Interpretation
	Looking at lasso plots, second regressor (year of operation) is regularized to 0.
	Hence year of operation may not be a significant predictor.
	Only age and number of nodes found are good predictors of survival.

