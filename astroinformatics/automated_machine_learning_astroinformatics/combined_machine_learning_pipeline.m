function combined_machine_learning_pipeline()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - combined_machine_learning_pipeline for astroinformatics
% Creation Date - 8th Aug 2019
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
% Description - Pipeline of machine learning algorithms 
%			to use on numeric data 
%
% Parameters - 
%	Input	
%
%	Output
%               BaggedEnsemble - ensemble of random forests
%               Plots of out of bag error
%			Example prediction	
%
% Example -
%		combined_machine_learning_pipeline()
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
% License - BSD
%
% Change History - 
%                   8th Aug 2015 - Creation by Soumya Banerjee
%                   22nd Aug 2017 - Modification by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load data
disp('Loading data ..')
fn_data_ptr = importdata('quasar_candidates_no_id.csv');
X = fn_data_ptr.data(:,1:end-1);
Y = fn_data_ptr.data(:,end);

%% Data munging
disp('Performing data munging ..')


%% Train and predict using random forests 
disp('Training random forests ..')
iNumTrees = 100

% TODO: vary number of trees and depth (number of lead nodes)
% TODO: use latest code from Matlab Central
BaggedEnsemble = generic_random_forests(X,Y,iNumTrees,'regression')

disp('Performing prediction ..')
predict(BaggedEnsemble,X(10,:))


%% PCA analysis
disp('Performing PCA ..')
names = fn_data_ptr.textdata(:,1);
categories = fn_data_ptr.textdata(1,2:end-1);
data_matrix = fn_data_ptr.data(:,1:end-1);
%function_pca_plot(data_matrix,categories,names)


%% Train and predict using neural networks
disp('Performing prediction using neural networks ..')
parallel = 0 % NOTE: parallel turned off
iNumHidden = 30
% TODO vary number of hidden layers

% call generic neural network function
generic_neural_network(X,Y,parallel,iNumHidden)


%% Train and predict using GLM with LASSO
% perform LASSO on GLM (in this case logistic regression)
str_distribution = 'binomial';
str_link_function = 'logit';
% [B preds] = generic_lasso_glm_predict(X,Y,str_distribution,str_link_function,10)


%% SVM

% TODO: multiclass SVM (see Matlab Central)
data_predictor = fn_data_ptr.data(:,1:end-1);
label          = fn_data_ptr.textdata(1,2:end-1);
one_label      = fn_data_ptr.textdata(1,2);

%kernel_function = 'polynomial' 
%x_svm = svm_classifier_matrix(data_predictor,label,one_label,kernel_function)

%kernel_function = 'quadratic' 
%x_svm = svm_classifier_matrix(data_predictor,label,one_label,kernel_function)

%kernel_function = 'rbf' 
%x_svm = svm_classifier_matrix(data_predictor,label,one_label,kernel_function)


%% t-SNE plots


%% k-means
%kmeans_generic(X1,X2,0)


%% DATA EXPLORATION AND INSIGHT GENERATION

% TODO: see 
%https://bitbucket.org/neelsoumya/hsph_work_allscripts
%https://bitbucket.org/neelsoumya/genomic_data_query_analyzer
%https://bitbucket.org/neelsoumya/data_munging_tools_public
%https://bitbucket.org/neelsoumya/ccfa_compound_prediction


%% save workspace variables
%save(sprintf('machinelearningpipeline_output_%s.mat',date))


%% Combine pdfs of all boxplots into one single pdf

%CAUTION - requires ghostscript and a UNIX/Mac OS X system
%This code courtesy of Decio Biavati at
%http://decio.blogspot.de/2009/01/join-merge-pdf-files-in-linux.html

unix('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=finished.pdf *.pdf')

