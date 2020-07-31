#############################################################
# Analysis for Accidents data
# Visualization and data exploration 
#############################################################

###############################################################
# Load required libraries
###############################################################

library("sqldf")
library(ggplot2)
library(timeSeries)

###############################################################
# Parameters of run
###############################################################
data_path_importsfile = ""


###############################################################
# Load data
###############################################################


df_accident <- read.csv(paste(data_path_importsfile,"ACCIDENT.csv",sep=""), 
                                  colClasses = "character")
head(df_accident)
names(df_accident)

# rename column names since not SQL compatible
names(df_accident)[5] <- "Accident_Type_Desc"
names(df_accident)[7] <- "Day_Week_Description"
names(df_accident)[16] <- "Light_Condition_Desc"


# Recast datetime columns as R Date object (makes SQL queries easier)
#df_accident$ACCIDENTDATE <- (as.Date(df_accident$ACCIDENTDATE, "%dd/%mm/%yyyy"))


########################################
# Data Exploration and visualization
########################################

sqldf("select distinct(ACCIDENT_TYPE) from df_accident ")
sqldf("select distinct(Accident_Type_Desc) from df_accident ")
sqldf("select distinct(Light_Condition_Desc) from df_accident ")
sqldf("select distinct(Day_Week_Description) from df_accident ")

temp_df = sqldf("select * from df_accident 
        order by ACCIDENTDATE
      ")

df_atmospheric_cond <- read.csv(paste(data_path_importsfile,"ATMOSPHERIC_COND.csv",sep=""), 
                        colClasses = "character")
head(df_atmospheric_cond)
names(df_atmospheric_cond)


###########################################################
# Combine two datasets
###########################################################

df_combined_accident_atmospheric = sqldf("select * 
                                         from df_accident
                                          inner join df_atmospheric_cond
                                          on df_accident.ACCIDENT_NO = df_atmospheric_cond.ACCIDENT_NO
                                         ")

names(df_combined_accident_atmospheric)
names(df_combined_accident_atmospheric)[29] <- "ACCIDENT_NO_2"
names(df_combined_accident_atmospheric)[32] <- "Atmosph_Cond_Desc"

#sqldf("select distinct(DAY_OF_WEEK) from df_combined_accident_atmospheric")
# NOTE: 0 and 1 both refer to Sunday
# t <- sqldf("select * from df_combined_accident_atmospheric where DAY_OF_WEEK = '0' ")
# t1 <- sqldf("select * from df_combined_accident_atmospheric where DAY_OF_WEEK = '1' ")
# t3 <- sqldf("select * from df_combined_accident_atmospheric where ATMOSPH_COND = '8' ")

df_combined_accident_atmospheric_sorted = sqldf(" select count(*) as count, ACCIDENTDATE, DAY_OF_WEEK
                                                  from df_combined_accident_atmospheric    
                                                  group by ACCIDENTDATE   
                                                  order by ACCIDENTDATE  
                                                ")


output_df = df_combined_accident_atmospheric_sorted
#output_df$intended_export_date = (as.Date(df$intended_export_date, "%d/%m/%y"))

output_df_ts = ts(output_df$count, start = c(2006,1,13), frequency = 365.25)
#pdf(file = "exports_dailycustoms_SEA_contain.pdf")
plot.ts(output_df_ts, main = "All Accidents over time", 
        xlab="Time", ylab="Accidents", 
        col = "blue")
#dev.off()


###########################################################
# Data Exploration for accident data
###########################################################

hist(output_df$count, main="Histogram for accidents", 
     xlab="Accidents daily", 
     border="black", 
     col="blue", breaks = 100)

hist(as.numeric(df_combined_accident_atmospheric$DAY_OF_WEEK),
     main="Histogram for day of week when accident occurred", 
     xlab="Day of week (1 - Sunday, 7 - Saturday)", 
     border="black", 
     col="blue", breaks = 100)

hist(as.numeric(df_combined_accident_atmospheric$ATMOSPH_COND),
     main="Histogram for atmospheric condition", 
     xlab="Atmospheric condition code (1 - Clear, 7 - Strong winds)", 
     border="black", 
     col="blue", breaks = 100)


###########################################################
# Build Random forest model (python)
###########################################################

# Save as csv with target
df_combined_accident_atmospheric_sorted_csv = sqldf(" select count(*) as count, DAY_OF_WEEK, ATMOSPH_COND
                                                  from df_combined_accident_atmospheric    
                                                group by ACCIDENTDATE   
                                                order by ACCIDENTDATE  
                                                ")

output_df_csv = df_combined_accident_atmospheric_sorted_csv

write.csv(output_df_csv, file = "aggregated_timeseries.csv", row.names = FALSE)

# call python script for random forests
# system("python predict_survival_randomforest.py")


############################################################
# Testing code
############################################################

# training_last_year = 2014
# training_last_quarter = 4
# 
# # Other parameters
# training_start_year = 2006
# training_start_quarter = 3
# 
# train_start = c(training_start_year,training_start_quarter) 
# train_end   = c(training_last_year,training_last_quarter) 
# cat(training_start_year,"\n")
# cat(training_start_quarter,"\n")
# #save()
# window(output_df_ts, start=train_start, end=train_end)

# Combine all PDFs (requires TeX or ghostscript to be installed on a UNIX or OSX machine)
#system("gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=finished.pdf *.pdf")



