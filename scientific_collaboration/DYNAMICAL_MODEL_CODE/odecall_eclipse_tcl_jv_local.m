function virus_vector = odecall_eclipse_tcl_jv_local(target,infected1,...
                            infected2,virus,beta,p,delta,gamma,k,...
                            time_phase,time_vector,filename,plotfig,...
                            figorssr,mac,interp,...
                            fileptr_target ...
                            )
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to call ODE solver and plot results and calculate SSR
%
% Name - odecall_eclipse_tcl_jv_local
% Creation Date - 10th Mar 2014
% Author - Soumya Banerjee
% Website - www.cs.unm.edu/~soumya
%
%
% Description - function to call ODE solver with eclipse phase, immune
%                   limited model, and plot results and 
%                   calculate SSR
%
% Parameters - 
%               target - initial target cell density (/mL)
%               infected1 - initial latently infected cell density (/mL)
%               infected2 - initial productively infected cell density
%               (/mL)
%               virus - initial virus density (/mL) in log 10
%               beta - infectivity log10
%               p - replication rate (PFU/day) in log 10
%               delta - infected cell death rate )/day)
%               gamma - immune system clearance rate (/day)
%               k - eclipse phase (rate of transition from I1 to I2) (/day)
%               rho - efficacy of antibody neutralization (/PRNT50 /day) 
%               eta - rate of antibody production (/day)
%               t_i - time at which adaptive antibody kicks in (days)
%               time_phase - duration of simulation (days)
%               time_vector - vector of measured times (days)
%               filename - name of data file
%               plotfig - 1 if data and model simulation needs to be
%                          plotted,
%                         0 if no plot needed  
%               figorssr - 1 if there is a need to access the data file
%                           (needed when need to calculate SSR or
%                            need to plot the data in the figure),
%                          0 if access to data file is not needed 
%               mac - 1 if this is my mac, 
%                     0 if my linux desktop
%               interp - 1 if interpolation of simulation needed,
%                        0 if not needed
%
%
% Example usage - virus_vector = odecall_eclipse_tcl_jv_local(2.3e5,0,...
%                   0,4.50638,0.00219,log10(5.13),log10(2.36),44.43,4,...
%                   10,[2 4 6 8 10],'knockout_RNA_div_500.tex',1,1,...
%                   1,1)
%
%
% Assumptions - 1) all parameters passed in numerical values i.e. not 
%                  logged except virus, beta, p and delta 
%                       (which are logged to base 10)
%               2) Phase 1 model with correction term with one infected
%                   cell compartment
%               3) File has two columns - first column has time in days
%                   and second column has virus load in PFU/mL in log10
%                   (see attached file)    
%               4) Model parameters passed locally in intial conditons
%                   to ode solver  
%               5) Immune system limited model with eclipse phse
%
% Comments -    1) Make this function inline to speed up computation 
%               2) Consider tinkering with ode options (using odeset)
%                   to speed up for your specific case
%
% License - BSD 
%
% Change History - 
%                   10th Mar 2014 - Creation by Soumya Banerjee
%                   3rd Aug  2014 - Modified by Soumya Banerjee
%                                       pretty printing of ODE parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 
options = odeset('RelTol', .001, ...
                 'NonNegative', [1 2 3 4]);
[t1 z1] = ode15s(@odecore_jv_tcl_eclipse_inline,[0, time_phase], ...
                                            [target ...
                                            infected1 ... 
                                            infected2 ...
                                            double(10^virus) ...
                                            double(10^beta) ...
                                            double(10^p) ...
                                            gamma ...
                                            double(10^delta) ...
                                            k ],...
                                            options);
 
% V is the fourth row of the output array
% CAUTION - change this if not true, e.g. in an ODE with one infected cell
% compartment

%T  = log10(z1(:,1)');
%I1 = log10(z1(:,2)');
 T  = (z1(:,1)');
 I1 = (z1(:,2)');
 V  = log10(z1(:,4)');
 
% If figure being plotted or ssr required, then import data
if figorssr == 1 || plotfig == 1
    % If run on my mac then data file initial pathname is different
    if mac == 1
        %fileptr = importdata(strcat('',filename), '\t');
    else
        % if run on my linux desktop data file initial pathname is
        % different
        %fileptr = importdata(strcat('~/matlab/linux_stage98/',filename), '\t');
    end
end
 
% Plot the results
if plotfig == 1
    
%     subplot (2,2,1); plot(t,log10(zT))
%     xlabel('Time (in days)');  ylabel('Count in log base 10'); title('Target cells progression');
% 
%     subplot (2,2,2); plot(t,zI)
%     xlabel('Time (in days)');  ylabel('Count in log base 10'); title('Infected cells progression');
% 
%     subplot (2,2,3); 
%     figure
    
    figID = figure
    plot(t1,T,'-r','linewidth',2)
    xlabel('Time ','fontsize',15); ylabel('Domestic/self connections (x)','fontsize',15);
    hold on
    
%     plot(t1,T,'-k','linewidth',2)
%     xlabel('Days post infection','fontsize',15); ylabel('Target cells concentration(log_1_0 (cells/mL))','fontsize',15);
    
    %plot(t1,log10(T + I1),'-k','linewidth',2)
    %xlabel('Days post infection','fontsize',15); ylabel('Target cells concentration (log_1_0 cells/mL)','fontsize',15);


    hold on
    % subplot (2,2,3); 
    %plot(fileptr.data(:,1),fileptr.data(:,2),'ro','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',8)
    %hold on
    %plot(fileptr_target.data(:,1),fileptr_target.data(:,2),'ro','MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',8)
    
    %legend('viremia','cell viability','Location','NorthEast')

    print(figID, '-dpdf', sprintf('delta_hist_%s.pdf', date));

    %% plot x vs. y/(x+y) or T vs. I1/(T + I1)
    figID_2 = figure
    plot(I1./(T + I1),(T), '-r','linewidth',2)
    xlabel('Percentage of foreign connections: y/(x+y)');  ylabel('Domestic/self connections (x)'); title('Evolution');

    print(figID_2, '-dpdf', sprintf('plot_evolution_%s.pdf', date));
end
 
% If interpolation needed, then do
if interp == 1
    virus_vector = interp1(t1, V, time_vector); 
else
    virus_vector = [];
end

% display parameters
disp('ODE Parameters (V0, beta, p, delta)')
[virus beta p delta] 
disp('ODE Parameters (T0, gamma, k)')
[target gamma k]

 
function dydt = odecore_jv_tcl_eclipse_inline(t,y)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function to solve ODEs
%
% Name - odecore_jv_tcl_eclipse_inline
% Creation Date - 27th Aug 2011
% Author - Soumya Banerjee
% Website - www.cs.unm.edu/~soumya
%
% Acknowledgements - Drew Levin
%
% Description - core function to solve ODEs, to be called from ode45, etc
%                   with eclipse phase for JV paper
%
% Assumptions - 1) All parameters passed in numerical values i.e. not 
%                  logged
%               2) Immune limited model with correction term
%               3) Eclipse phase included
%               4) Inline version
%               5) Intermediate variables removed
%
% Comments - Make this function inline to speed up computation as this
%               is called repeatedly by ode45 or ode15s
%          - Input argument not used and replaced by ~ 
%
% License - BSD
%
% Change History - 
%                   27th Aug 2011 - Creation by Soumya Banerjee
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    dydt = y;

    % Initial conditions
    T  = y(1); % Target cells
    I1 = y(2); % Infected cells not producing virus (latently infected)
    I2 = y(3); % Infected cells producing viru (productively infected)
    V  = y(4); % Virus

    % Values of model parameters passed in y
    beta   = y(5);
    p      = y(6);
    gamma  = y(7);
    delta  = y(8);
    k      = y(9); % eclipse phase (rate of transition from I1 to I2) 
    
    % System of differential equations
	dydt(1) = beta*y(1) + delta*y(1)*y(2);
	dydt(2) = 0; % beta*T*V - k*I1;
    dydt(3) = 0; % k*I1 - delta*I2;
    dydt(4) = 0; % p*I2 - gamma*V - beta*T*V;
	
    % These are the model parameters. Since they do not change with time,
    % their rate of change is 0
	dydt(5) = 0;
	dydt(6) = 0;
	dydt(7) = 0;
	dydt(8) = 0;
    dydt(9) = 0;
    
    