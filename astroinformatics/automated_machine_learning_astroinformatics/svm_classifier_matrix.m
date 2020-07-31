function svmStruct = svm_classifier_matrix(data_predictors,label,one_label,kernel_function)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - svm_classifier
% Creation Date - 2nd Dec 2014
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to use an SVM to classify data.
%               Example code to classify data into two classes :
%                   Passerines (P) and non-Passerines (NP)
%
% Parameters - 
%               Input -
%                   meas_1 -  values for first class
%                   meas_2 -  values for second class
%                   label  -  labels for two classes
%                   one_label - label for just one class
%                   kernel_function - kernel function (rbf, linear)
%
%               Output -
%                   Plot of classification
%                   Structure containing trained SVM classifier
%
%
% Assumptions - 
%
% Comments -   
%
% Example - 
%
% meas_1 = [10          9           10.5        8.7         7.1         9           8.5         9.3         12           3          2.1       3.4        4.5        5          1.7      2.2        4           ]';
% meas_2 = [log10(0.03) log10(0.05) log10(0.09) log10(0.07) log10(0.15) log10(0.04) log10(0.07) log10(0.12) log10(0.06)  log10(3)   log10(2)  log10(2.2) log10(1.9) log10(1.9) log10(2) log10(2.9) log10(3.2)  ]';
% label = [ 'P '; 'P '; 'P '; 'P '; 'P '; 'P '; 'P '; 'P '; 'P ';    'NP'; 'NP'; 'NP'; 'NP'; 'NP'; 'NP'; 'NP'; 'NP';] 
% one_label = 'NP'
% kernel_function = 'rbf'
% svm_classifier(meas_1,meas_2,label,one_label,kernel_function)
%
%
% License - BSD
%
% Change History - 
%                   2nd Dec 2014 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic;


%% Create data, a two-column matrix
data   = data_predictors; %[meas_1, meas_2];

species = (label);

%% From the species vector, create a new column vector, groups, to classify data into two groups: Setosa and non-Setosa.
groups = ismember(species,one_label);


%% Randomly select training and test sets.
[train, test] = crossvalind('holdOut',groups);
cp = classperf(groups);

%% Use the svmtrain function to train an SVM classifier using a radial basis function and plot the grouped data.
%svmStruct = svmtrain(data(train,:),groups(train),'showplot',true);
svmStruct = svmtrain(data(train,:),groups(train),'showplot',true,'kernel_function',kernel_function);


%% Classify the test set using a support vector machine.
classes = svmclassify(svmStruct,data(test,:),'showplot',true);


%% Evaluate the performance of the classifier.
classperf(cp,classes,test);
cp.CorrectRate

toc;
