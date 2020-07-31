function B = generic_lasso_logistic(X,y,str_distribution,num_fold_cross_validation)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - generic_lasso_logistic
% Creation Date - 4th August 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to perform LASSO on 
%			logistic regression and
%			do prediction.
%
% Parameters - 
%               Input -
%                   X -  vector of values (response)
%                   y -  vector of values (data matrix)
%                   str_distribution  -  class of distribution
%			    num_fold_cross_validation - number of fold cross validation to use	
%
%               Output -
%                   B - matrix of estimated regularized coefficients
%                   Plots of cross-validation
%
%
% Assumptions - 
%
% Comments - 
% str_distribution must be 'binomial' for logistic regression  
%
% Example -
% load fisheriris
% X = meas(51:end,:);
% y = strcmp('versicolor',species(51:end));
% custom_lasso_glm(X,y,'binomial',10)
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
% License - BSD
%
% Change History - 
%                   4th August 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%str_distribution must be 'binomial' for logistic regression

%% call GLM LASSO
disp('Performing LASSO on logistic regression ..')
[B FitInfo] = lassoglm(X,y,str_distribution,'CV',num_fold_cross_validation)

%% plot cross validation results
lassoPlot(B,FitInfo,'PlotType','CV'); saveas(gcf,'lassoplot_1.eps', 'psc2')
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log'); saveas(gcf,'lassoplot_2.eps', 'psc2') 

%% perform prediction

% get coefficients for Lambda value with minimum
% deviance plus one standard deviation point
%indx = FitInfo.Index1SE;
indx = FitInfo.IndexMinDeviance
B0 = B(:,indx);
nonzero_predictors = sum(B0 ~= 0)

% get constant intercept term
cnst = FitInfo.Intercept(indx);
B1 = [cnst;B0];

% perform predictions using this model
disp('Performing prediction ..')
preds = glmval(B1,X,'logit');
%keyboard

% save workspace variables
save(sprintf('lasso_logistic_output_%s.mat',date))