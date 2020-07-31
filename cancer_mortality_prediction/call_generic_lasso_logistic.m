function call_generic_lasso_logistic()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - call_generic_lasso_logistic
% Creation Date - 4th August 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to load data and call main 
%			LASSO GLM function.
%
% Example - 
%			call_generic_lasso_logistic()
%
% License - BSD
%
% Change History - 
%                   4th August 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load fisheriris
X = meas(51:end,:);
Y = strcmp('versicolor',species(51:end));
generic_lasso_logistic(X,Y,'binomial',10)
