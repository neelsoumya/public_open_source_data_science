Open source tools for clinical informatics

Most of these are based on a platform called TranSMART

1) Useful Links
A demo is available here:

http://etriks1.uni.lu/transmart/datasetExplorer

2) Curated datasets are available here:

https://wiki.transmartfoundation.org/display/transmartwiki/Curated+Data+Repository

IBD (Janssen In iximab cohort) data on tranSMART available here:

http://library.transmartfoundation.org/datasets/EtriksGSE16879/

3) More information on tranSMART: 

http://transmartfoundation.org/scientists-clinicians/

4) Publications on tranSMART: 

tranSMART: An Open Source Knowledge Management and High Content Data Analytics Platform 

https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4333702/

6) Online training on tranSMART: 
http://transmartfoundation.org/training_recordings/
http://transmartfoundation.org/the-i2b2-transmart-foundation-2018-training-program/ 


7) Different data loading tools:
https://wiki.transmartfoundation.org/display/transmartwiki/Data+loading+tools

8) Data upload tricks

http://transmartfoundation.org/training_recordings/

https://drive.google.com/ le/d/0B8lizkKDeaKhbDEzcWxBdk5KYTQ/view?usp=sharing

	a) Save as <StudyID>_Gene_Expression_Data_R.txt Tab delimited le

	Substitute L or Z for R when R is used for normalized data L is used for log2 transformed normalized data
	Z is used for z-score normalized data
	Find out more about HDD normalization and tM loading in
	https://wiki.transmartfoundation.org/display/transmartwiki/HDD+Data+Curation+and+Normalization 
	by ‘tMF Standards Working Group’

	Take home message: It is highly recommended to load HDD data as log2 transformed. Data specific and
	experiment specific approaches should be used to deal with "0" and negative values.

	b) Downloading platform files 
	Download from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi
	using as query say GPL17586

	and then download full table

	c) Best practices for creating tree structures

	CDISC tree standard SDTM
	
	https://www.cdisc.org/standards/foundational/sdtm

	d) Best practice for QCing data before ETL upload

	elastic search kibana for QC of data prior to ETL

	e) Multi-cohort comparison using glowing bear 

	https://github.com/thehyve/glowing-bear


9) Resources
	http://athena.ohdsi.org/

	https://icd.who.int/browse10/2016/en

	https://termbrowser.nhs.uk/
