function wilcoxon_ranksum_boxplot(filename_healthy,filename_disease,yaxis_label)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name - wilcoxon_ranksum_boxplot
% Creation Date - 28th July 2014
% Author: Soumya Banerjee
% Website: https://sites.google.com/site/neelsoumya/
%
% Description: 
%  Perform a Wilcoxon rank-sum test or Mann-Whitney U-test (non-parametric test)
%  Takes measurements of genes/compounds/bugs/metabolites etc in healthy
%  (control) individuals and individuals with some disease. Then does a
%  Wilcoxon rank-sum test to find out how many of these genes/compounds/metabolites/bugs,
%  etc are differentially abundnant in healthy vs. disease in a
%  statistically significant manner.
%  Also generates box plots for each of the measured genes/compounds/bugs/metabolites
%  The data for healthy individuals and individuals with some disease must be supplied in
%  separate files. The rows will represent different measured genes/compounds/bugs/metabolites
%  and the columns will be different individuals. 
%
% Input:  
%       1) filename_healthy: Name of file containing matrix of measurements
%           of some quantity (genes/metabolites/bugs/compounds, etc) in healthy (control) individuals.
%           The columns are replicates or different individuals, that is (say) the same compound
%           measured in multiple healthy individuals.
%           The rows are different compounds.
%           There may be a header row that has some descriptive text (not
%           necessary).
%           It is necessary to have the first column of this file be a text
%           only column that has some description of the quantity being
%           measured. For example, if a compound is being measured in
%           multiple individuals, then the name of that compound should be
%           in the first column. If this is all too complicated, dont worry
%           (!), because two example files are attached.
%       2) filename_disease: Name of file containing matrix of measurements
%           of some quantity (genes/metabolites/bugs/compounds, etc) in individuals with some disease.
%           Same restrictions as above apply.
%       3) yaxis_label:   label for y-axis of boxplots (for example compound abundance) 
%   
% Output: 
%       1) A file containing results from a Wilcoxon rank-sum test comparing each row in the
%           file filename_healthy to each row in the file filename_disease.
%           This includes p-value, rank-sum score and a Benjamini-Hochberg corrected
%           FDR (False Discovery Rate) q-value   
%       2) The number of compounds/genes/metabolites etc (represented by the
%           rows) that are differentially abundant in healthy vs. disease
%           (statistically significant at p-value < 0.05 and also FDR q-value <
%           0.05)
%       3) A number of boxplots of each of the measured quantities (say compound abundance) in disease vs. healthy
%       4) A combined boxplot that combines all boxplots in 3) into one
%           (requires ghostscript to be installed and needs to be run on
%           UNIX/Mac OS X). Currently commented out.
%
% Assumotions -
%       1) Input file must be tab-delimited
%       2) The names of compounds/bugs/metabolites/genes etc must match (row-by-row) between the two
%           input files i.e. if say compound X is in row 3 of the healthy file
%           (e.g compounds_healthy.csv), then compound X must also be in row 3
%           of the disease file (e.g. compounds_disease.csv)
%
% Example usage:
%       wilcoxon_ranksum_boxplot('compounds_healthy.csv','compounds_disease.csv','Compound abundance')
%
% License - BSD 
%
% Acknowledgements -
%           Dedicated to my wife Joyeeta
%
% Change History - 
%                   28th July 2014  - Creation by Soumya Banerjee
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load the data
data_ptr_healthy = importdata(filename_healthy);
data_ptr_disease = importdata(filename_disease);

%% Initial checks
% number of rows in healthy file must be the same as the number of rows in
% disease file. This reflects the fact that both must have measured the
% same number of quantities (genes, compounds, bugs, metabolites, etc)
if size(data_ptr_healthy.data,1) ~= size(data_ptr_disease.data,1)
    disp('The number of rows in healthy file must be the same as the number of rows in')
    disp('disease file. This reflects the fact that both must have measured the same number of quantities (genes, compounds, bugs, metabolites, etc) in both healthy individuals and individuals with disease')
    return
end

iNumRows = size(data_ptr_healthy.data,1);

iTempCount = 1; % temporary counter variable to store index of array of wilcoxon stats
% Iterate through all rows
for iCount = 1:iNumRows
    %% Perform a Wilcoxon rank-sum test or Mann-Whitney U-test
    [temp_p_value temp_h_value temp_wilcox_stats] = ranksum( ...
                                                                data_ptr_healthy.data(iCount,:), ...
                                                                data_ptr_disease.data(iCount,:)  ...
                                                                );
                                                            
    array_wilcox_risk(iTempCount,:) = [temp_p_value temp_h_value temp_wilcox_stats.ranksum];
    iTempCount = iTempCount + 1;        
end


%% Perform a Benjamini-Hochberg correction of p-values to yield FDR (False Discovery Rate)
PValues = array_wilcox_risk(:,1);
FDR = mafdr(PValues,'BHFDR',true);

%% Generate a volcano plot (optional: commented out)
%mavolcanoplot(data_ptr_healthy.data,data_ptr_disease.data,PValues)

%% Calculate number of measurements (compounds, metabolites, bugs, etc) 
%% that are differentially abundant (statistically significant) in healthy vs. disease
temp_index_sign_FDR = find(FDR < 0.05);
disp('Number of measurements (compounds, metabolites, bugs, etc) that are differentially abundant (statistically significant) in healthy vs. disease at FDR < 0.05')
size(temp_index_sign_FDR,1)

temp_index_sign_pvalue = find(PValues < 0.05);
disp('Number of measurements (compounds, metabolites, bugs, etc) that are differentially abundant (statistically significant) in healthy vs. disease at p-value < 0.05')
size(temp_index_sign_pvalue,1)

%% Save list of p-values, ranksum score and FDR values to disk
dlmwrite('vital_stats.txt',[array_wilcox_risk(:,1) array_wilcox_risk(:,3) FDR],'delimiter','\t')
disp('Statistics saved in file vital_stats.txt')

%% Create box plot

% Iterate through all rows
for iCount = 1:iNumRows

    %% Combined data for healthy and disease
    data_box_plot = [ data_ptr_healthy.data(iCount,:)' ; data_ptr_disease.data(iCount,:)'];

    %% Replicates labels
    str_group = [repmat('Healthy',size(data_ptr_healthy.data(iCount,:),2),1); repmat('Disease',size(data_ptr_disease.data(iCount,:),2),1)];

    %% create figure for boxplot
    temp_figID = figure;
    boxplot(data_box_plot,str_group,'notch','on')
    title(data_ptr_healthy.textdata(iCount))
    ylabel(yaxis_label)

    %% save boxplot to disk
    print(temp_figID, '-dpdf', sprintf('boxplot_%d_%s.pdf', iCount, date));
end
disp('Box plots saved in file names boxplot_*.pdf')

%% Combine pdfs of all boxplots into one single pdf
%% CAUTION - requires ghostscript and a UNIX/Mac OS X system
% Commented out. Uncomment if you meet both of these requirements
% This code courtesy of Decio Biavati at
% http://decio.blogspot.de/2009/01/join-merge-pdf-files-in-linux.html

%unix('gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=finished.pdf *.pdf')