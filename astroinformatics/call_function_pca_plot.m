function call_function_pca_plot()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - call_function_pca_plot
% Creation Date - 29th June 2015
% Author: Soumya Banerjee
% Website: https://sites.google.com/site/neelsoumya/
%
% Description: 
%   Function that calls generic function to perform PCA
%
% Example usage -
%		call_function_pca_plot
%
% License - BSD 
%
% Acknowledgements -
%           Dedicated to my mother Kalyani Banerjee, my father Tarakeswar Banerjee
%				and my wife Joyeeta Ghose.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


load cities
data_matrix = ratings; 
function_pca_plot(data_matrix,categories,names)

