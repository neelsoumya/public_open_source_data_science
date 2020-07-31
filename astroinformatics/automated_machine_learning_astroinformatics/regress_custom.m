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
% License:  BSD
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
display('r2')
stats(1,1)
display('OLS Slope')
b1(2,1)
olsslope = b1(2,1);
display('p value')
stats(1,3)

figure
plot(plot_x,plot_y,'bo');
xlabel(x_label,'fontsize',18);
ylabel(y_label,'fontsize',18);
title(strcat(strcat(strcat(strcat(desc,strcat(strcat(strcat(strcat('r^2 value = ',num2str(stats(1,1))), strcat('OLSslope = ',num2str(olsslope))), 'RMAslope = '),num2str(rmaslope)))),'p value = '),num2str(stats(1,3))),'fontsize',18);
hold on
% axis([-2 3 1 12])
% axis([1 11 1 17])
plot(plot_x,olsslope .* plot_x + b1(1,1),'-r');
hold on
% plot([-2 -1 0 1 2 3],[5 5 5 5 5 5],'-r')
% plot(plot_x,rmaslope * plot_x + 1,'-g');
% legend('data','OLS','RMA','Location','NorthEast')

hold off

display('b1')
b1
end

