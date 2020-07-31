function query_file_v4b_compoundabundances_multiscatter()

%% takes a pcl file and does scatter (after filtering)
%% compound names dont need to match (this program does that)

tic;

%% importd data
file_ptr = importdata('quasar_candidates.csv')

%%
size(file_ptr.data)
size(file_ptr.textdata)

%%
%file_ptr.data.Sheet1(:,11)

%iNumIter = 476;

found_flag = 0;

%% can do this automatically later using textdata
% positional data for data columns (use only with data)
iTscore_cohort1 = 43;
iPvalue_cohort1 = 44;
iFdr_cohort1 = 45;

iTscore_cohort2 = 143;
iPvalue_cohort2 = 144;
iFdr_cohort2 = 145;

% here cohort1 is cohort2 sjhotgun
iExtent_cohort1_ibd     = [4:6];
iExtent_cohort1_healthy = [1:3];

iExtent_cohort2_ibd     = [11:13];
iExtent_cohort2_healthy = [7:10];

% positional data for text columns (use only with textdata)
iCompound_Name_cohort1  = 1;
iCompound_Name_cohort2 = 2;

iNum_metadata_rows = 1; % this is how many rows difference there will be bewteen textdata and data
iTHRESHOLD_FILTERING = 0.8; % threshold for filtering

%% find a list of compouds that are dff abundant in both cohort1 and cohort2
% at pvalue < 0.05
% so first find compounds in cohort1, same as befor ebu tnow add to dictionary
% find compounds that are diff expressed in cohort1 (pvalue < 0.05)
% put it into a hashtable
%disp('compounds diff abundant in cohort1 pvalue < 0.05')
% iTempCount = 1;
% for iCount = 1:size(file_ptr.textdata,1)
%     % iCount tracks number of textdata rows
%     % iCount minus iNum_metadata_rows tracks number of data rows
%     %iCount
%     %iFdr_cohort1
%     if iCount > iNum_metadata_rows 
%         if ~isnan(file_ptr.data(iCount - iNum_metadata_rows,iPvalue_cohort1)) && file_ptr.data(iCount - iNum_metadata_rows,iPvalue_cohort1) < 0.05
%             %file_ptr.textdata(iCount,iCompound_Name_cohort1)
%             key_cohort1(iTempCount)    = file_ptr.textdata(iCount,iCompound_Name_cohort1);
%             %values_cohort1(iTempCount) = file_ptr.data(iCount - iNum_metadata_rows,iPvalue_cohort1);
%             values_cohort1(iTempCount) = iCount - iNum_metadata_rows; % store the data "pointer" in values; you can use this to directly go into data matrix
%             iTempCount = iTempCount + 1;
%         end
%     end
%     
% end
% 
% % add to hash table
% if ~isempty(key_cohort1) && ~isempty(values_cohort1)
%     cohort1_Map = containers.Map(key_cohort1,values_cohort1)
% end
% 
% % size(values_cohort1)
% % values_cohort1(1)
% % return

% similarly for cohort2
%disp('compounds diff abundant in cohort2 pvalue < 0.05')
regress_array_disease = zeros(10,10);
regress_array_healthy = zeros(10,10);

iTempCount = 1;
for iCount = 1:size(file_ptr.textdata,1)
    % iCount tracks number of textdata rows
    % iCount minus iNum_metadata_rows tracks number of data rows
    %iCount
    %iFdr_cohort1
    if iCount > iNum_metadata_rows 
        
        % filetring step for cohort2 16S
        % go forward only if at least 80% of samples are noz-zero
        
        % CAUTION revverse logic
        if size(find(file_ptr.data(iCount - iNum_metadata_rows,iExtent_cohort1_ibd) > 0),2)/length(iExtent_cohort1_ibd) < iTHRESHOLD_FILTERING
            continue
        end
        
        temp_compound_name_cohort1 = file_ptr.textdata(iCount,iCompound_Name_cohort1);
        %temp_compound_name_cohort1{1}
        
        if ~isnan(temp_compound_name_cohort1{1}) %&& ~isempty(temp_compound_name_cohort1{1})
            
            
            % match compind names in cohort2 shotgun
            for iInnerCount = 1:size(file_ptr.textdata,1)
                 if iInnerCount > iNum_metadata_rows
                     
                     % CAUTION betweeen ICount anf iInnerCount
                    temp_compound_name_cohort2 = file_ptr.textdata(iInnerCount,iCompound_Name_cohort2);
                    %temp_compound_name_cohort2{1}
                     
                    if ~isnan(temp_compound_name_cohort2{1}) %%&& ~isempty(temp_compound_name_cohort2{1})
                        
                        
                        % if compiund names match
                        if strcmp(temp_compound_name_cohort1{1},temp_compound_name_cohort2{1})
                            %disp('in')
                            % compote average abundance for thjat compiund
                            % in cohort2 shotgun and in 16s
                            % CAUTION betweeen ICount anf iInnerCount
                            %iCount for cohort1 but iInnerCount for PRIMS
                            regress_array_disease(iTempCount,:) = [ mean(file_ptr.data(iCount - iNum_metadata_rows,iExtent_cohort1_ibd)) mean(file_ptr.data(iInnerCount - iNum_metadata_rows,iExtent_cohort2_ibd)) ];
                            regress_array_healthy(iTempCount,:) = [ mean(file_ptr.data(iCount - iNum_metadata_rows,iExtent_cohort1_healthy)) mean(file_ptr.data(iInnerCount - iNum_metadata_rows,iExtent_cohort2_healthy)) ];
                            iTempCount = iTempCount + 1;
                            break
                        end
                    end
                 end
            end
            
%         if ~isnan(file_ptr.data(iCount - iNum_metadata_rows,iPvalue_cohort2)) && file_ptr.data(iCount - iNum_metadata_rows,iPvalue_cohort2) < 0.05
%             %file_ptr.textdata(iCount,iCompound_Name_cohort2)
%             key_cohort2(iTempCount)    = file_ptr.textdata(iCount,iCompound_Name_cohort2);
%             %values_cohort2(iTempCount) = file_ptr.data(iCount - iNum_metadata_rows,iPvalue_cohort2);
%             values_cohort2(iTempCount) = iCount - iNum_metadata_rows; % store the data "pointer" in values; you can use this to directly go into data matrix
%             iTempCount = iTempCount + 1;
%         end
        end
    end
    
end



regress_custom(log10(regress_array_disease(:,1)),log10(regress_array_disease(:,2)),'DISEASE','log10 cohort1 shotgun compound rel. abundance','log10 cohort2 shotgun compound rel. abundance')
regress_custom(log10(regress_array_healthy(:,1)),log10(regress_array_healthy(:,2)),'HEALTHY','log10 cohort1 shotgun compound rel. abundance','log10 cohort2 shotgun compound rel. abundance')

end


function [plot_y plot_x] = regress_custom(input_x,input_y,desc,x_label,y_label)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Author: Soumya Banerjee
% Website: www.cs.unm.edu/~soumya
%
% Acknowledgements: Dr. Melanie Moses
%
% Description: Takes two column vectors, a description, x label and y label and 
% plots the data and outputs all the statistics (r2, OLS slope, RMA slope, 95% CI
% intervals)
% Same syntax as plot(x,y) : x-axis data 1st followed by y-axis data
%
% input_x: column vector x
% input_y: column vector y
% desc:    description of plot (goes in the title of the plot)
% x_label: label for x-axis of plot
% y_label: label for y-axis of plot
%
% Example usage:
% regress_custom(log10(data(:,1)),log10(data(:,2)),'Scaling of p (model fit
%   to data)','Log 10 of host mass (Kg)','Pathogen replication rate log10(p) (/day)')
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[a_rowcount x] = size(input_y);
[b_rowcount x] = size(input_x);


counter = 0;
for iRow = 1 : a_rowcount
    if input_y(iRow, 1) == 0   
    else
        counter = counter + 1;
        temp = input_y(iRow, 1);
        plot_y(counter, 1) = temp;
        temp = input_x(iRow, 1);
        plot_x(counter, 1) = temp;
    end
end

display('counter:')

msapop_ones = [ones(size(plot_x)), plot_x];
[b1 bint r rint stats] = regress(plot_y,msapop_ones);
display('RMA Slope')
rmaslope = b1(2,1) / stats(1,1) ^ 0.5
display('bint')
bint
display('95% CI on intercept')
bint(1,:)
display('95% CI on slope')
bint(2,:)
display('r2')
stats(1,1)
display('OLS Slope')
b1(2,1)
olsslope = b1(2,1);
display('p value')
stats(1,3)
display('b1')
b1

figure
plot(plot_x,plot_y,'ro','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',10)
xlabel(x_label,'fontsize',23);
ylabel(y_label,'fontsize',23);
title(strcat(strcat(strcat(strcat(desc,strcat(strcat(strcat(strcat('r^2 value = ',num2str(stats(1,1))), strcat('OLSslope = ',num2str(olsslope))), 'RMAslope = '),num2str(rmaslope)))),'p value = '),num2str(stats(1,3))),'fontsize',18);
hold on
% axis([0 95 -1.4 0.2])
% axis([1 11 1 17])
plot(plot_x,olsslope .* plot_x + b1(1,1),'-b');
hold on
% plot([-2 -1 0 1 2 3],[5 5 5 5 5 5],'-r')
% plot(plot_x,rmaslope * plot_x + 1,'-g');
% legend('data','OLS','RMA','Location','NorthEast')

hold off

display('b1')
b1
end