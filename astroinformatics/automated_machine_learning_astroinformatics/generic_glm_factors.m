function glm = generic_glm_factors(data_matrix, formula, str_distribution)

%
%    load hospital
%    data_matrix = hospital
%    formula = 'Smoker ~ Age*Weight*Sex - Age:Weight:Sex';
%    str_distribution = 'binomial';
%    generic_glm_factors(data_matrix, formula, str_distribution)
%

    
glm = fitglm(data_matrix, formula, 'distr', str_distribution)

