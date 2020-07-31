function call_generic_neural_network()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - call_generic_neural_network
% Creation Date - 8th Aug 2015
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to load example data
%			and call generic function for neural networks for 
%			prediction.
%
% Parameters - 
%               Input -
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
% 	call_generic_neural_network()
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
% Has code from http://au.mathworks.com/help/nnet/examples/house-price-estimation.html
% Also has code from
% https://uk.mathworks.com/matlabcentral/answers/235524-how-to-replace-matlapool-by-parpool-in-my-code
% by Matt J https://uk.mathworks.com/matlabcentral/profile/authors/1440443-matt-j
%
% License - BSD
%
% Change History - 
%                   8th Aug 2015 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% load example data
[X,Y] = house_dataset;

parallel = 1
iNumHidden = 20 %1 %5 10 20 50

% take transpose of matrices
X = X';
Y = Y';

% call generic neural network function
generic_neural_network(X,Y,parallel,iNumHidden)


