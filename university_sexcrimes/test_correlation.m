function test_correlation()

fn_ptr = importdata('crime_univ_sept2014.txt')

fn_ptr

regress_custom(log10(fn_ptr(:,2)),(fn_ptr(:,1)),'','log_1_0 enrollment','crime')

