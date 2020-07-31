readme

Open source project to predict popularity of news articles from open data.

Data is from http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity

The project predicts news popularity using random forests and neural networks.

Usage:
matlab < predict_news_popularity.m


1) predict_news_popularity.m
	Function to load data and call generic random forests function
	Usage:
		matlab < predict_news_popularity.m

2) generic_random_forests.m
	Generic function for random forests


3) generic_neural_networks.m
	Generic function for neural networks.


4) OnlineNewsPopularity.csv
	News popularity data from http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity


5) OnlineNewsPopularity.names
	News popularity metadata from http://archive.ics.uci.edu/ml/datasets/Online+News+Popularity


6) randomforests_errorplot.pdf
	Out of bag classification error for random forests.

7) performance_neuralnetwork.pdf and prediction_neuralnetwork.pdf
	Plots of performance and prediction accuracy of neural networks.


Interpretation: In this analysis we observe that neural networks slightly outperform random forests
			in the task of predicting news article popularity.

