library(shiny)
library("sqldf")
library(ggplot2)
library(timeSeries)
library(forecast)
library("vars")
library('MARSS')
library('TTR')



# Preload data and perform computations
cat("Preloading data. Please wait ...\n")
source('data_analyzer_R.R')
cat("Preloading functions. Please wait ...\n")
source("functions_lib.R")



# Define server logic required to draw a histogram
shinyServer(function(input, output) {
  
  # Expression that generates a histogram. The expression is
  # wrapped in a call to renderPlot to indicate that:
  #
  #  1) It is "reactive" and therefore should re-execute automatically
  #     when inputs change
  #  2) Its output type is a plot
  
    
  output$distPlot <- renderPlot({
    
    # get parameter from the front-end sliders
    numpoints_forecast = input$horizon
    training_last_year = input$slider_train_lastyear
    training_last_quarter = input$slider_train_lastquarter
    
    # Other parameters
    training_start_year = 2006
    training_start_quarter = 3
    
    # Load/create training data based on new training year end date
    train_start = c(training_start_year,training_start_quarter) 
    train_end   = c(training_last_year,training_last_quarter) 
    #cat(training_start_year,"\n")
    #cat(training_start_quarter,"\n")
    #save()
    target_ts_train = window(output_df_ts, start=train_start, end=train_end)
    
    if (input$select_algorithm == 1) {
      # fit best ARIMA model
       best_arima_model <- fit_best_ARIMA(target_ts_train, FALSE)
       
       #numpoints_forecast = 365
       best_arima_forecast <- forecast.Arima(best_arima_model, h = numpoints_forecast)
       plot.forecast(best_arima_forecast, xlab="Time", ylab="Accidents daily", 
                     pch=18, col="blue",main = "Best fit ARIMA model")
       
#       # Show metrics
#       cat("Performance on test set (best ARIMA)\n")
#       calculate_model_accuracy(best_arima_forecast,target_ts_test)
      
    }
    

    if (input$button_data_explore == 2)
    {
      # if you want to do data exploration
      
      if (input$select_data == 2)
      {
          # if day of week  
          hist(as.numeric(df_combined_accident_atmospheric$DAY_OF_WEEK),
               main="Histogram for day of week when accident occurred", 
               xlab="Day of week (0,1 - Sunday, 7 - Saturday)", 
               border="black", 
               col="blue", breaks = 100)
      }
      else if (input$select_data == 3)
      {
          # else if atmospheric condition
          hist(as.numeric(df_combined_accident_atmospheric$ATMOSPH_COND),
               main="Histogram for atmospheric condition", 
               xlab="Atmospheric condition code (1 - Clear, 7 - Strong winds)", 
               border="black", 
               col="blue", breaks = 100)
        
      }
      else if (input$select_data == 1)
      {
          # number of accidents daily
          hist(output_df$count, main="Histogram for daily accidents", 
              xlab="Accidents daily", 
              border="black", 
              col="blue", breaks = 100)
      }
      
      
    }
    
  })
})