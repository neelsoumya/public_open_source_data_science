function [B preds] = generic_elasticnet_glm_predict(X,y,str_distribution,str_link_function,num_fold_cross_validation, alpha)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - generic_elasticnet_glm_predict
% Creation Date - 26th March 2018
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to perform LASSO on 
%			GLM and do prediction.
%
% Parameters - 
%               Input -
%                   X -  vector of values (response)
%                   y -  vector of values (data matrix)
%                   str_distribution  -  class of distribution
%			    str_link_function - link function
%			    num_fold_cross_validation - number of fold cross validation to use	
%               alpha - elastic net parameter
%
%               Output -
%                   B - matrix of estimated regularized coefficients
%			    preds - predictions using GLM	
%                   Plots of cross-validation
%
%
% Assumptions - 
%
% Comments - 
% str_distribution must be 'binomial' for logistic regression  
%
% Example -
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
% License - BSD
%
% Change History - 
%                   8th August 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%str_distribution must be 'binomial' for logistic regression

%% call GLM LASSO
disp('Performing elastic net on logistic regression ..')
[B FitInfo] = lassoglm(X,y,str_distribution,'CV',num_fold_cross_validation, 'Alpha', alpha, 'MaxIter',1e5)

%% plot cross validation results
lassoPlot(B,FitInfo,'PlotType','CV'); saveas(gcf,'lassoplot_1.eps', 'psc2')
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log'); saveas(gcf,'lassoplot_2.eps', 'psc2') 

%% perform prediction

% get coefficients for Lambda value with minimum
% deviance plus one standard deviation point
%indx = FitInfo.Index1SE;
indx = FitInfo.IndexMinDeviance;
B0 = B(:,indx);
disp('Number of non-zero predictors after Elastic Net')
nonzero_predictors = sum(B0 ~= 0)

%% what are these non-zero predictors?
disp('What are these non-zero predictors')
(B0 ~= 0)'


% get constant intercept term
cnst = FitInfo.Intercept(indx);
B1 = [cnst;B0];

% perform predictions using this model
disp('Performing prediction ..')
preds = glmval(B1,X,str_link_function);
%keyboard

% save workspace variables
save(sprintf('lasso_logistic_output_%s.mat',date))