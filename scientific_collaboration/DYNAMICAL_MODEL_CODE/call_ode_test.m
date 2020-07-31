function call_ode_test()


% Equations are Berkeley MAdonna
%d/dt(x) = alpha*x + beta*x*y
%d/dt(y) = 0
%Init x = 1e3
%Init y = 1000
%alpha = 1e-2
%beta = 1e-5
%delta = 1e-4
%gamma = 1e-2

% alpha is beta here
% beta is delta here 


target = 1000; %1e6
V0 = 1
beta = (1e-2); %1e-5
p = 10
delta = (1e-5); %2
gamma = 10
k = 4
endtime = 100
time_vector = [1:100]
filename = ''
fileptr_target = ''

infected1 = 1000

sim_virus_vector = odecall_eclipse_tcl_jv_local(target,infected1,...
        0,log10(V0),log10(beta),log10(p),log10(delta),gamma,k,...
        endtime,...
        time_vector,filename,1,1,1,1,...
        fileptr_target ...
        );