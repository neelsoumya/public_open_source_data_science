function call_generic_lasso_glm_predict()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - call_generic_lasso_glm_predict
% Creation Date - 8th August 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to load data and call main 
%			GLM LASSO function and do prediction.
%
% Example - 
%			call_generic_lasso_glm_predict()
%
% License - BSD
%
% Change History - 
%                   8th August 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load fisheriris
X = meas(51:end,:);
Y = strcmp('versicolor',species(51:end));

% perform LASSO on GLM (in this case logistic regression)
[B preds] = generic_lasso_glm_predict(X,Y,'binomial','logit',10)


