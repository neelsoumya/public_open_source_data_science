function predict_mortality()

% load data
disp('Loading data ..')
fn_data_ptr = importdata('haberman.data');
X = fn_data_ptr(:,1:3);
Y = fn_data_ptr(:,4);

% 0 represents survival and 1 means mortality
disp('Perform data munging ..')
Y(find(Y == 1)) = 0;
Y(find(Y == 2)) = 1;

% LASSO on GLM (logistic regression)
disp('Perform LASSO on GLM (logistic regression) ..')
generic_lasso_logistic(X,Y,'binomial',10)
