function kmeans_clustering()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - kmeans_clustering
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
%                   Data on campus crimes
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
%       kmeans_example3
%
%
% License - BSD
%
% Change History -
%                   10th Dec 2014 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load data
X = importdata('crime_univ_sept2014.txt')
%regress_custom(log10(fn_ptr(:,2)),(fn_ptr(:,1)),'','log10 enrollment','crime')

%% Initialize variables
iNumClusters = 4
iNumReplicates = 1000

%% Plot data
figID = figure;
plot(log10(X(:,2)),X(:,1),'k*','MarkerSize',5);
title 'Univerity sex crimes dataset';
xlabel 'Log_1_0 enrollment';
ylabel 'Crimes';
print(figID, '-djpeg', sprintf('cluster_ORIGINALDATA_%s.jpg', date));

%% Settings
opts = statset('Display','final');
[idx,C] = kmeans([ log10(X(:,2)) X(:,1) ],iNumClusters,'Distance','cityblock',...
    'Replicates',iNumReplicates,'Options',opts);


%%Plot the clusters and the cluster centroids.
figID = figure;
plot(log10(X(idx==1,2)), X(idx==1,1) ,'r.','MarkerSize',12)
hold on
plot(log10(X(idx==2,2)), X(idx==2,1) ,'b.','MarkerSize',12)
hold on
plot(log10(X(idx==3,2)), X(idx==3,1) ,'g.','MarkerSize',12)
hold on
plot(log10(X(idx==4,2)), X(idx==4,1) ,'k.','MarkerSize',12)
%hold on
%plot(log10(X(idx==5,2)), X(idx==5,1) ,'y.','MarkerSize',12)
%hold on
%plot(log10(X(idx==6,2)), X(idx==6,1) ,'c.','MarkerSize',12)
%hold on
%plot(log10(X(idx==7,2)), X(idx==7,1) ,'m.','MarkerSize',12)
title 'Univerity sex crimes dataset';
xlabel 'Log_1_0 enrollment';
ylabel 'Crimes';
hold off

%plot(C(:,1),C(:,2),'kx',...
%     'MarkerSize',15,'LineWidth',3)
%legend('Cluster 1','Cluster 2','Centroids',...
%       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

%% Save final plot
print(figID, '-dpdf', sprintf('cluster_plot_%s.pdf', date));

