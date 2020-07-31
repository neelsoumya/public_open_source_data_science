function svm_call

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - svm_call
% Creation Date - 2nd Dec 2014
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to call SVM using example data (binary classifer)
%
%
% License - BSD
%
% Change History - 
%                   2nd Dec 2014   - Creation by Soumya Banerjee
%                   30th June 2017 - Modified by Soumya Banerjee generic
%                                       kernel
%                   3rd July 2017 -  Modified by Soumya Banerjee generic
%                                       one class label    
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


unix(‘wget https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data')

data_matrix = importdata('spambase.data');
icol = size(data_matrix,2)
data_predictor = data_matrix(:,1:icol-1); % predictors matrix
label = data_matrix(:,end); % last column is 0 or 1
one_label = data_matrix(1,end); % one label

kernel_function = 'polynomial' 
x_svm = svm_classifier_matrix(data_predictor,label,one_label,kernel_function)

kernel_function = 'quadratic' 
x_svm = svm_classifier_matrix(data_predictor,label,one_label,kernel_function)

kernel_function = 'rbf' 
x_svm = svm_classifier_matrix(data_predictor,label,one_label,kernel_function)






