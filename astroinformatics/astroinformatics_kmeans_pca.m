
function astroinformatics_kmeans_pca()

%% load astronomy data
fn_data = importdata('quasar-candidates.csv')
X1 = fn_data.data(:,10);
X2 = fn_data.data(:,11);

%% Call generic kmeans function
kmeans_generic(X1,X2,0)

%% PCA analysis
data_matrix = fn_data.data(1:10000,2:end); 
size(data_matrix)                  

categories  = fn_data.colheaders(2:end)
names = num2str(fn_data.data(1:10000,1));
size(names)
size(categories)

%% call generic PCA function
function_pca_plot(data_matrix,categories,names)
