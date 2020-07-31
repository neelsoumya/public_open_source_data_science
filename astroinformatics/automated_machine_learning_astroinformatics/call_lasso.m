function call_lasso()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - call_lasso
% Creation Date - 8th Jan 2014
% Author: Soumya Banerjee
% Website: https://sites.google.com/site/neelsoumya/
%
% Description: 
%   Function to generate test dataset and call generic function to
%   perform LASSO
%
% Input:  
%   
% Output: 
%       1) Matrix of inferred regressors
%
% Assumptions -
%
% Example usage:
%   call_lasso
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


%% Construct a data set with redundant predictors, and identify those predictors using cross-validated lasso.
%Create a matrix X of 100 five-dimensional normal variables and a response vector Y from just two components of X, with small added noise.

X = randn(100,5)
r = [0;2;0;-3;0] % only two nonzero coefficients
Y = X*r + randn(100,1)*.1 % small added noise

%% Call LASSO generic function
iFold_cross_validation = 10; % do 10 fold cross validation

B = lasso_generic(X,Y,iFold_cross_validation)

