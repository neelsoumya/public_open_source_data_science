function B = lasso_generic(X,Y,iFold_cross_validation)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - lasso_generic
% Creation Date - 8th Jan 2014
% Author: Soumya Banerjee
% Website: https://sites.google.com/site/neelsoumya/
%
% Description: 
%   Function to generate test dataset and call generic function to
%   perform LASSO
%
% Input:  
%       X - matrix of predictors
%       Y - vector of responses
%       small_sigma_squared - standard deviation^2 (variance) for covariance matrix for Y
%       eta_sqaured - standard deviation^2 (variance) for covariance matrix for beta (regressors)
%
% Output: 
%       1) Matrix of inferred regressors (B)
%
% Assumptions -
%
% Example usage:
%		X = randn(100,5)
%		r = [0;2;0;-3;0] % only two nonzero coefficients
%		Y = X*r + randn(100,1)*.1 % small added noise
%		iFold_cross_validation = 10; % do 10 fold cross validation
%		B = lasso_generic(X,Y,iFold_cross_validation)
%
% License - BSD 
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
% Change History - 
%                   8th Jan 2014  - Creation by Soumya Banerjee
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%LASSO identifies and removes the redundant predictors.

%% Do k-fold cross validation where k = iFold_cross_validation
[B FitInfo] = lasso(X,Y,'CV',iFold_cross_validation)

%% Plot the cross-validated fits
lassoPlot(B, FitInfo, 'PlotType', 'CV')

%% Trace plot of coefficients fit by LASSO against L1 norm
lassoPlot(B)
