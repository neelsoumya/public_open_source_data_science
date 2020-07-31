function B = custom_lasso_glm(X,y,str_distribution,num_fold_cross_validation)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - custom_lasso_glm
% Creation Date - 17th June 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to perform LASSO on 
%			generalized linear model (GLM).
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
%                   17th June 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% call GLM LASSO
[B FitInfo] = lassoglm(X,y,str_distribution,'CV',num_fold_cross_validation)

% plot cross validation results
lassoPlot(B,FitInfo,'PlotType','CV'); saveas(gcf,'lassoplot_1.eps', 'psc2')
lassoPlot(B,FitInfo,'PlotType','Lambda','XScale','log'); saveas(gcf,'lassoplot_2.eps', 'psc2') 

