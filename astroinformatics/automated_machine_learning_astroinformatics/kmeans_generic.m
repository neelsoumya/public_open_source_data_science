function kmeans_generic(X1,X2,iParallel)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - kmeans_generic
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
%                   X1 -  column vector of data (independent variable)
%                   X2 -  column vector of data (dependent  variable)
%                   iParallel - 1 if you want iterations to be on multiple
%                                   cores,
%                               0 if not using parallel processing
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
%           load fisheriris
%           X1 = meas(:,3);
%           X2 = meas(:,4);
%           kmeans_generic(X1,X2,1)
%
%
% License - BSD
%
% Change History -
%                   10th Dec 2014 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic; 

%% Plot data
figID = figure;
plot(X1,X2,'k*','MarkerSize',5);
title 'Original Data';
%xlabel 'Petal Lengths (cm)';
%ylabel 'Petal Widths (cm)';
print(figID, '-djpeg', sprintf('cluster_ORIGINALDATA_%s.jpg', date));

%% Settings
if iParallel == 1
    opts = statset('Display','final','UseParallel',1);
else
    opts = statset('Display','final');
end

iNumClusters = 2;
iNumReplicates = 10;
[idx,C] = kmeans([X1 X2],iNumClusters,'Distance','cityblock',...
    'Replicates',iNumReplicates,'Options',opts);


%%Plot the clusters and the cluster centroids.
figID = figure;
plot(X1(idx==1),X2(idx==1),'r.','MarkerSize',12)
hold on
plot(X1(idx==2),X2(idx==2),'b.','MarkerSize',12)
%hold on
% plot(X1(idx==3),X2(idx==3) ,'g.','MarkerSize',12)
%hold on
% plot(X1(idx==4),X2(idx==4) ,'k.','MarkerSize',12)


plot(C(:,1),C(:,2),'kx',...
     'MarkerSize',15,'LineWidth',3)

legend('Cluster 1','Cluster 2','Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off

%% Save final plot
print(figID, '-djpeg', sprintf('cluster_plot_%s.jpg', date));

toc;