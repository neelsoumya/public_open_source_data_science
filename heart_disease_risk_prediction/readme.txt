readme

Data from http://archive.ics.uci.edu/ml/datasets/Heart+Disease

This is a project that tries to predict heart disease risk from open source data
using random forests.

Usage:
matlab < predict_heart_disease_level.m

1) predict_heart_disease_level.m
	MATLAB script to load data, do data munging, perform classification 
	using random forests and then perform PCA analysis.

2) add_metadata.py
	Python script for data munging

3) generic_random_forests.m
	Generic function for training random forests

4) function_pca_plot.m
	Generic function for performing PCA analysis

5) processed.cleveland.data
	Processed dataset for heart disease from
	http://archive.ics.uci.edu/ml/datasets/Heart+Disease

6) randomforest_errorplot.pdf
	Plot of errors (out of bag errors) from training random forests

7) categoriesplot.pdf and pcaplot_2comp.pdf
	Plots from PCA analysis

8) screeplot.pdf
	Scree plot from PCA analysis