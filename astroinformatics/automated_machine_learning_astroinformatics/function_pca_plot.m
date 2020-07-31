function function_pca_plot(data_matrix,categories,names)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - function_pca_plot
% Creation Date - 29th June 2015
% Author: Soumya Banerjee
% Website: https://sites.google.com/site/neelsoumya/
%
% Description: 
%   Generic function to perform PCA and plot results
%
% Input:  
%       data_matrix - matrix of data; rows are items and columns are dimensions
%       categories - string matrix containing name of indices
%       names - string matrix of names/items
%   
%
% Example usage:
% load cities
% data_matrix = ratings; 
% function_pca_plot(data_matrix,categories,names)
%
% License - BSD 
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Perform PCA
w = 1./var(data_matrix);
[wcoeff,score,latent,tsquared,explained] = pca(data_matrix,'VariableWeights',w);

% Transform coefficients so that they are orthonormal
coefforth = diag(sqrt(w))*wcoeff;


% Plots of first two principal components
figID = figure;
plot(score(:,1),score(:,2),'+')
xlabel('1st Principal Component')
ylabel('2nd Principal Component') 
print(figID, '-dpdf', sprintf('pcaplot_2comp_%s.pdf', date));


% Scree plot 
figID = figure;
pareto(explained)
xlabel('Principal Component')
ylabel('Variance Explained (%)')
print(figID, '-dpdf', sprintf('screeplot_%s.pdf', date));


% Find out most extreme points
[~, index] = sort(tsquared,'descend');
extreme = index(1:5)
disp('Most extreme points:')
names(extreme,:)


% Plot of two principal components and categories
figID = figure;
biplot(coefforth(:,1:2),'scores',score(:,1:2),'varlabels',categories);
print(figID, '-dpdf', sprintf('categoriesplot_%s.pdf', date));


%% Combine pdfs of all boxplots into one single pdf
%% CAUTION - requires ghostscript and a UNIX/Mac OS X system
% Commented out. Uncomment if you meet both of these requirements
% This code courtesy of Decio Biavati at
% http://decio.blogspot.de/2009/01/join-merge-pdf-files-in-linux.html
% unix('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=finished.pdf *.pdf')
