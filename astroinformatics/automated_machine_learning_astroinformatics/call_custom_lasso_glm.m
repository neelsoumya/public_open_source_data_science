function call_custom_lasso_glm()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - call_custom_lasso_glm
% Creation Date - 17th June 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to load data and call main 
%			LASSO GLM function.
%
% Example - 
%			call_custom_lasso_glm()
%
% License - BSD
%
% Change History - 
%                   17th June 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load fisheriris
X = meas(51:end,:);
y = strcmp('versicolor',species(51:end));
custom_lasso_glm(X,y,'binomial',10)
