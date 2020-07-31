%% Multi-Layer Perceptron (MLP)

clear all; close all;
rng('default');

%% load data
classes = 0:9;
Ntrain = 60000;
[Xtrain, ytrain, Xtest, ytest] = mnistLoad(classes, Ntrain);

figure;
idx = randperm(Ntrain); idx = idx(1:64);
display_mnist(Xtrain(idx,:));

%% set model parameters
model.n_output = 10;
model.n_features = size(Xtrain,2);
model.n_hidden = 50;
model.l1 = 0;
model.l2 = 0.1;
model.epochs = 10;
model.eta = 0.001;
model.alpha = 0.001;
model.decrease_const = 0.00001;
model.minibatches = 50;

%% MLP
[model] = mlp_fit(Xtrain, ytrain, model);

y_train_pred = mlp_predict(Xtrain, model);
y_test_pred = mlp_predict(Xtest, model);

%% compute accuracy
acc = sum(ytrain' == y_train_pred) / size(Xtrain,1);
fprintf('Training accuracy: %.2f\n',(acc * 100));

acc = sum(ytest' == y_test_pred) / size(Xtest,1);
fprintf('Test accuracy: %.2f\n',(acc * 100));

%% generate plots
figure;
plot(model.cost); 
title('MLP Training Loss'); grid on;
ylabel('Loss'); xlabel('Epochs x Minibatch');

%% notes

