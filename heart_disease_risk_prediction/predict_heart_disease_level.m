
function predict_heart_disease_level()

%% Function to train random forests and predict 
%% risk of heart disease

tic;

%% load data
disp('Loading data ...')
fn_data = importdata('processed.cleveland.data')
X = fn_data(:,1:13)
Y = fn_data(:,14)

%% train random forest
disp('Training random forest ...')
BaggedEnsemble = generic_random_forests(X(1:80,:),Y(1:80,:),5000,'classification')

%% predict heart disease risk level using random forests
 predict(BaggedEnsemble,fn_data(85,1:13))



%% perform PCA analysis

% add dummy names and add categories (python script)
disp('Perform data munging ...')
unix('python add_metadata.py')

% perform PCA analysis
fn_sec_data = importdata('numbered_cleveland_data.txt')
names  = int2str(fn_sec_data.data(1:end-1,1));
data_matrix  = fn_sec_data.data(1:end-1,2:end);

% and the categories are ...
cell_array_categories{1} = 'age'
cell_array_categories{2} = 'sex'
cell_array_categories{3} = 'cp'
cell_array_categories{4} = 'trestbps'
cell_array_categories{5} = '(chol)'
cell_array_categories{6} = '(fbs)'
cell_array_categories{7} = '(restecg)'
cell_array_categories{8} = '(thalach)'
cell_array_categories{9} = '(exang)'
cell_array_categories{10} = '(oldpeak)'
cell_array_categories{11} = '(slope)'
cell_array_categories{12} = '(ca)'
cell_array_categories{13} = '(thal)'
cell_array_categories{14} = '(num)'

disp('Performing PCA analysis ...')
function_pca_plot(data_matrix,cell_array_categories,names)

toc;
