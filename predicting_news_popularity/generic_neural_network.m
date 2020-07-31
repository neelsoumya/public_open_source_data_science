

function generic_neural_network(X,T,parallel,iNumHidden)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - generic_neural_network
% Creation Date - 8th Aug 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to use neural networks for prediction.
%
% Parameters - 
%               Input -
%                   X -  vector of values (response)
%                   y -  vector of values (data matrix)
%                   parallel  -  1 (use multiple cores)
%						0 (do not use multiple cores)
%			    iNumHidden - number of hidden layers
%
%               Output -
%                   Plots of prediction and performance of neural networks
%
%
% Assumptions - 
%
% Comments -   
%
% Example -
% 	[X,Y] = house_dataset;
% 	parallel = 1
% 	iNumHidden = 20
% 	X = X';
% 	Y = Y';
% 	generic_neural_network(X,Y,parallel,iNumHidden)
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
% Has code from http://au.mathworks.com/help/nnet/examples/house-price-estimation.html
%
% License - BSD
%
% Change History - 
%                   8th Aug 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


tic;

if matlabpool('size') == 0 && parallel == 1
    matlabpool open
end

% take transpose of matrices
X = X';
T = T';

% fit model
net = fitnet(iNumHidden);
view(net)

% train network
disp('Training neural network ..')
[net,tr] = train(net,X,T,'useParallel','yes');
figID = figure;
plotperform(tr)
print(figID, '-dpdf', sprintf('performance_neuralnetwork_%s.pdf', date));


% test performance
testX = X(:,tr.testInd);
testT = T(:,tr.testInd);
testY = net(testX);
perf = mse(net,testT,testY)

% make predictions
disp('Predicting using neural network ..')
y = net(X);
figID = figure;
plotregression(T,y)
print(figID, '-dpdf', sprintf('prediction_neuralnetwork_%s.pdf', date));

% close matlabpool if open
if matlabpool('size') > 0 && parallel == 1
    matlabpool close
end

toc;