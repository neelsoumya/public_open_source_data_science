function generic_hierarchical_clustering_matlab(data_matrix)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generic MATLAB function to perform
%   hierarchical clustering
%
% Arguments:
%           pred pedictors
%           resp logical True or False vector of class labels
%
% Usage:
%
% 
% generic_hierarchical_clustering_matlab(data_matrix)
%
% Adapted from
% https://uk.mathworks.com/help/stats/hierarchical-clustering.html
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% Distance Information The pdist function returns this distance 
%   information in a vector, Y, where each element contains the distance
%   between a pair of objects.

dist_clust = pdist(data_matrix);


% see distances
squareform(dist_clust)


%% Linkages

link_clust = linkage(dist_clust);


%% Dendrogram

figure;
dendrogram(link_clust)