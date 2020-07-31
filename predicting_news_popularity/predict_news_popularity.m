function predict_news_popularity()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - predict_news_popularity
% Creation Date - 6th Aug 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
% Description - Function to predict news popularity data using 
% 			random forests
%
% Parameters - 
%	Input	
%
%	Output
%               BaggedEnsemble - ensemble of random forests
%               Plots of out of bag error
%		Example prediction	
%
% Example -
%		predict_news_popularity()
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
% License - BSD
%
% Change History - 
%                   6th Aug 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Loading data ..')
fn_data_ptr = importdata('OnlineNewsPopularity.csv');
X = fn_data_ptr.data(:,1:end-1);
Y = fn_data_ptr.data(:,end);

disp('Training random forests ..')
% log transform response
BaggedEnsemble = generic_random_forests(X,log10(Y),100,'regression')

disp('Performing prediction ..')
predict(BaggedEnsemble,X(10,:))

%disp('Performing PCA ..')
%names = fn_data_ptr.textdata(:,1);
%categories = fn_data_ptr.textdata(1,2:end-1);
%data_matrix = fn_data_ptr.data(:,1:end-1);
%function_pca_plot(data_matrix,categories,names)

disp('Performing prediction using neural networks ..')
parallel = 1
iNumHidden = 30

% call generic neural network function
% log transform response
generic_neural_network(X,log10(Y),parallel,iNumHidden)
