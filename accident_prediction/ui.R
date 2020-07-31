library(shiny)

# Define UI for application that draws a histogram
shinyUI(fluidPage(
  
  # Application title
  titlePanel("Road Accident Forecasting and Data Exploration Tool"),
  
  fluidRow(
    
    column(3,
           radioButtons("button_data_explore", label = h3("Perform data exploration"),
                        choices = list("No" = 1, "Yes" = 2 #,"Choice 3" = 3
                                       ),selected = 1)),

    column(3,
           selectInput("select_data", label = h3("Data Columns"), 
                       choices = list("Number of accidents daily" = 1,
                                      "Day of week" = 2 , "Atmospheric Condition" = 3 #, "VAR" = 5
                       ), selected = 1)),
    

    column(3,
           selectInput("select_algorithm", label = h3("Algorithm"), 
                       choices = list("ARIMA" = 1 #, "Exponential smoothing" = 2 #, "VAR" = 5
                                      ), selected = 1)),
    
    column(3, 
           sliderInput("slider_train_lastyear", label = h3("Training end year"),
                       min = 2006, max = 2016, value = 2014),
           #sliderInput("slider_train_lastquarter", "",
          #             min = 1, max = 4, value = c(1, 4))
           sliderInput("slider_train_lastquarter", label = h3("Training end month"),
                       min = 1, max = 12, value = 2)
    ) 
  ),
  
  
  # Sidebar with a slider input for the number of bins
  sidebarLayout(
    sidebarPanel(
      sliderInput("horizon",
                  "Forecast horizon (days):",
                  min = 1,
                  max = 1000,
                  value = 365, round = TRUE)
    ),
    
    
    
    # Show a plot of the generated distribution
    mainPanel(
      plotOutput("distPlot")
    )
  )
))