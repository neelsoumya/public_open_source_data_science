function call_generic_elasticnet_glm_predict()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - call_generic_elasticnet_glm_predict.m
% Creation Date - 6th April 2018
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to load data and call main 
%			GLM elastic net function and do prediction.
%
% Example - 
%			call_generic_elasticnet_glm_predict()
%
% License - BSD
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose, and my friend Irene Egli.
%
% Change History - 
%                   6th April 2018 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load fisheriris
X = meas(51:end,:);
Y = strcmp('versicolor',species(51:end));

% perform elastic net on GLM (in this case logistic regression)
[B preds] = generic_elasticnet_glm_predict(X,Y,'binomial','logit',10, 0.2)


