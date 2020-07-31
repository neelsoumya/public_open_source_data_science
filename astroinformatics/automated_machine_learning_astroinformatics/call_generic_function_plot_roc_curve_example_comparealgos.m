function call_generic_function_plot_roc_curve_example_comparealgos()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function to load example data and call generic MATLAB function to take
%   predictors and response
% 	and plot ROC curve and compute
%	AUC for 4 different classification
%	algorithms
%
% Usage:
% matlab < call_generic_function_plot_roc_curve_example_comparealgos.m
%
%
% Adapted from
% https://uk.mathworks.com/help/stats/perfcurve.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



load ionosphere

% X is a 351x34 real-valued matrix of predictors. Y is a character array of
% class labels: 'b' for bad radar returns and 'g' for good radar returns.
% Reformat the response to fit a logistic regression. Use the predictor variables 3 through 34.

resp = strcmp(Y,'b'); % resp = 1, if Y = 'b', or 0 if Y = 'g'
pred = X(:,3:34);
 
generic_function_plot_roc_curve_example_comparealgos(pred, resp)
