load hospital
formula = 'Smoker ~ Age*Weight*Sex - Age:Weight:Sex';
glm = fitglm(hospital,formula,'distr','binomial')

