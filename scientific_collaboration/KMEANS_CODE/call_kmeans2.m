function call_kmeans2()

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Name - call_kmeans
% Creation Date - 10th Dec 2014
% Author - Soumya Banerjee
% Website - https://sites.google.com/site/neelsoumya/
%
%
% Description - Function to use cluster data using kmeans.
%               Takes two column vectors of data.
%
% Parameters -
%               Input -
%
%               Output -
%                   Plot of clustering
%
%
% Assumptions -
%
% Comments -
%
% Example -
%           call_kmeans
%
% License - BSD
%
% Change History -
%                   10th Dec 2014 - Creation by Soumya Banerjee
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Load data

meas_1 = [17
42
45
21
17
40
20
14
29
27
24
27
16
53
19
31
43
31
24
14
36
22
31
10
35
29
41
35
32
26
18
27
33
28
27
69
44
47
28
38
31
60
42
34
47
31
27
64
47
40
80
91
12
90
82
43
50
54
24
46
47
97
46
27
32
72
32
89
51
43
60
70
96
10
90
59
92
50
49
71
67
99
55
34
84
85
74
83
74
91
27
70
36
97
65
89
97
83
100
100
96
94
47
94
96
24
100
100
95
99
93
100
100];


meas_2 = [117
113
54
91
48
45
78
44
82
67
70
34
41
27
50
79
29
64
41
32
35
49
60
18
58
62
29
32
20
37
41
15
50
49
52
8
39
27
26
32
28
20
24
43
24
40
32
17
25
30
33
25
2
25
17
17
15
18
27
17
22
16
14
10
15
20
22
17
27
4
13
4
13
3
8
6
3
15
11
11
25
12
12
21
12
3
11
4
3
1
3
12
9
7
6
6
9
12
6
3
11
7
5
4
7
2
8
4
2
4
4
7
3];



%% Call generic kmeans function
kmeans_generic2(meas_1,meas_2,0)

