function call_kmeans()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - call_kmeans
% Creation Date - 10th Dec 2014
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to use cluster data using kmeans.
%               Takes two column vectors of data.
%
% Parameters -
%               Input -
%
%               Output -
%                   Plot of clustering
%
%
% Assumptions -
%
% Comments -
%
% Example -
%           call_kmeans
%
% License - BSD
%
% Change History -
%                   10th Dec 2014 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load data
load fisheriris
X1 = meas(:,3);
X2 = meas(:,4);

%% Call generic function
kmeans_generic(X1,X2,0)

