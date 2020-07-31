%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example MATLAB script to load data
% 	and plot ROC curve and compute
%	AUC for 4 different classification
%	algorithms:
%		GLM with logistic regression
%		SVM
%		Naive Bayes
%		Classification trees
%
% Usage:
% matlab < plot_roc_curve_example_comparealgos.m
%
% Adapted from
% https://uk.mathworks.com/help/stats/perfcurve.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load ionosphere

% X is a 351x34 real-valued matrix of predictors. Y is a character array of class labels: 'b' for bad radar returns and 'g' for good radar returns.
% Reformat the response to fit a logistic regression. Use the predictor variables 3 through 34.

% TODO: Input parameters are pred (predictor matrix) and resp (1 and 0)

resp = strcmp(Y,'b'); % resp = 1, if Y = 'b', or 0 if Y = 'g'
pred = X(:,3:34);

% Fit a logistic regression model to estimate the posterior probabilities for a radar return to be a bad one.

mdl = fitglm(pred,resp,'Distribution','binomial','Link','logit');
score_log = mdl.Fitted.Probability; % Probability estimates

% Compute the standard ROC curve using the probabilities for scores.

[Xlog,Ylog,Tlog,AUClog] = perfcurve(resp,score_log,'true');

% Train an SVM classifier on the same sample data. Standardize the data.

mdlSVM = fitcsvm(pred,resp,'Standardize',true);

% Compute the posterior probabilities (scores).

mdlSVM = fitPosterior(mdlSVM);
[~,score_svm] = resubPredict(mdlSVM);

% The second column of score_svm contains the posterior probabilities of bad radar returns.


% Compute the standard ROC curve using the scores from the SVM model.

[Xsvm,Ysvm,Tsvm,AUCsvm] = perfcurve(resp,score_svm(:,mdlSVM.ClassNames),'true');

% Fit a naive Bayes classifier on the same sample data.

mdlNB = fitcnb(pred,resp);

% Compute the posterior probabilities (scores).

[~,score_nb] = resubPredict(mdlNB);

% Compute the standard ROC curve using the scores from the naive Bayes classification.

[Xnb,Ynb,Tnb,AUCnb] = perfcurve(resp,score_nb(:,mdlNB.ClassNames),'true');

% Plot the ROC curves on the same graph.

figure
plot(Xlog,Ylog)
hold on
plot(Xsvm,Ysvm)
plot(Xnb,Ynb)
legend('Logistic Regression','Support Vector Machines','Naive Bayes','Location','Best')
xlabel('False positive rate'); ylabel('True positive rate');
title('ROC Curves for Logistic Regression, SVM, and Naive Bayes Classification')
hold off

% Although SVM produces better ROC values for higher thresholds, logistic regression is usually better at distinguishing the bad radar returns from the good ones. The % ROC curve for naive Bayes is generally lower than the other two ROC curves, which indicates worse in-sample performance than the other two classifier methods.

% Compare the area under the curve for all three classifiers.

AUClog
AUCsvm
AUCnb




% classification trees

load fisheriris

%The column vector, species, consists of iris flowers of three different species: setosa, versicolor, virginica. The double matrix meas consists of four types of % measurements on the flowers: sepal length, sepal width, petal length, and petal width. All measures are in centimeters.

% Train a classification tree using the sepal length and width as the predictor variables. It is a good practice to specify the class names.

Model = fitctree(meas(:,1:2),species, ...
    'ClassNames',{'setosa','versicolor','virginica'});

% Predict the class labels and scores for the species based on the tree Model.

[~,score] = resubPredict(Model);

%The scores are the posterior probabilities that an observation (a row in the data matrix) belongs to a class. The columns of score correspond to the classes specified % by 'ClassNames'. So, the first column corresponds to setosa, the second corresponds to versicolor, and the third column corresponds to virginica.

% Compute the ROC curve for the predictions that an observation belongs to versicolor, given the true class labels species. Also compute the optimal operating point and % y values for negative subclasses. Return the names of the negative classes.

% Because this is a multiclass problem, you cannot merely supply score(:,2) as input to perfcurve. Doing so would not give perfcurve enough information about the scores %for the two negative classes (setosa and virginica). This problem is unlike a binary classification problem, where knowing the scores of one class is enough to %determine the scores of the other class. Therefore, you must supply perfcurve with a function that factors in the scores of the two negative classes. One such function %is .

diffscore = score(:,2) - max(score(:,1),score(:,3));
[X,Y,T,~,OPTROCPT,suby,subnames] = perfcurve(species,diffscore,'versicolor');

%X, by default, is the false positive rate (fallout or 1-specificity) and Y, by default, is the true positive rate (recall or sensitivity). The positive class label is %versicolor. Because a negative class is not defined, perfcurve assumes that the observations that do not belong to the positive class are in one class. The function % %accepts it as the negative class.

OPTROCPT
suby
subnames

%OPTROCPT =
%
%    0.1000    0.8000



%subnames =
%
%  1x2 cell array
%
%    {'setosa'}    {'virginica'}

% Plot the ROC curve and the optimal operating point on the ROC curve.

figure
plot(X,Y)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off

% Find the threshold that corresponds to the optimal operating point.

T((X==OPTROCPT(1))&(Y==OPTROCPT(2)))


%Specify virginica as the negative class and compute and plot the ROC curve for versicolor.

%Again, you must supply perfcurve with a function that factors in the scores of the negative class. An example of a function to use is .

diffscore = score(:,2) - score(:,3);
[X,Y,~,~,OPTROCPT] = perfcurve(species,diffscore,'versicolor', ...
    'negClass','virginica');
OPTROCPT
figure, plot(X,Y)
hold on
plot(OPTROCPT(1),OPTROCPT(2),'ro')
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC Curve for Classification by Classification Trees')
hold off

