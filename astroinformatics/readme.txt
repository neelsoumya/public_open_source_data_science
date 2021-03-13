readme

Analysis of astronomy data using machine learning techniques and astroinformatics.

Usage: matlab < astroinformatics_kmeans_pca.m

1) quasar-candidates.csv
	Astroinformatics data from 
	
	https://people.astro.ruhr-uni-bochum.de/polsterer/ai2rub.html
	
	and 

	https://people.astro.ruhr-uni-bochum.de/polsterer/quasar-candidates.csv

2) astroinformatics_kmeans_pca.m
	MATLAB code to load data and perform k-means and PCA analysis

3) kmeans_generic.m
	Generic MATLAB function to perform kmeans

4) function_pca_plot.m
	Generic MATLAB function to perform PCA analysis

5) cluster_plot_10-Jul-2015.jpg
	Clustering using k-means

6) categoriesplot_10-Jul-2015.pdf, pcaplot_2comp_10-Jul-2015.pdf, screeplot_10-Jul-2015.pdf
	PCA plots

Analysis reveals two distinct clusters in the PCA plots and two distinct clusters in the
k-means plot.

7) autoencoder

python3 basic_autoencoder_keras_allnumeric.py 'quasar_candidates_no_id.csv' 'astronomy_data_' 0.4

