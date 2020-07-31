#######################################################
# Required functions 
#######################################################

# function to difference time series data (by taking diff) 
# in order to get stationary data
diff_ts <- function(data_handle,i_diff)
{
  # i_diff: number of times differencing is to be done
  differenced_timeseries <- diff(data_handle, differences = i_diff)
}
#------------------------------------------------------------------------------------


fit_forecast_HoltWinters <- function(b_constant_trend, b_seasonality, timeseries_data_handle, param_beta, param_gamma, numpoints_forecast)
{
  # function to fit using Holt-Winters exponential smoothing model and forecast
  # assumes this is the correct model to use
  # requires forecast package
  #
  # Input parameters
  # b_constant_trend: if TRUE then constant trend, and FALSE if increasing/decreasing trend   # Y.T.: what do you mean by "constant trend" and "increasing/decreasing trend"? 
  # b_seasonality: if TRUE then seasonality trend, and FALSE if no seasonality
  # timeseries_data_handle: time series object
  # param_beta: TRUE/FALSE
  # param_gamma: TRUE/FALSE
  # numpoints_forecast: number of points to be forecast for in time series object
  #
  # Returns: List of objects (access using $)
  # fit_model: (Holt-Winters fit model or error flag 1)
  # forecast_ts: forecast timeseries
  
  require(forecast)
  
  # error flag to be returned
  b_error = 0
  
  if (b_constant_trend == TRUE) {
    if (b_seasonality == FALSE) {
      holtwinters_fit_model <- HoltWinters(timeseries_data_handle, beta = param_beta, gamma = param_gamma)
      # beta = param_beta, gamma = param_gamma should all be FALSE
    }
    else {
      cat("Error: cannot have constant trend and also seasonality")   # Y.T.: why?
      b_error = 1
      return (b_error)
    }
  }
  else {
    if (b_seasonality == FALSE) {
      # increasing/decreasing trend but no seasonality
      holtwinters_fit_model <- HoltWinters(timeseries_data_handle, gamma = FALSE)
    }
    else {
      # increasing/decreasing trend AND seasonality
      holtwinters_fit_model <- HoltWinters(timeseries_data_handle)
    }
  }
  
  cat("SSE of model fit: ",holtwinters_fit_model$SSE,"\n")
  
  plot(holtwinters_fit_model)
  
  forecast_timeseries <- forecast.HoltWinters(holtwinters_fit_model, h = numpoints_forecast)
  
  plot.forecast(forecast_timeseries, xlab="Years", ylab="Port throughput (AUD, imports)", pch=18, col="blue")
  
  # pack all objects to be returned in a list
  ret_list <- list("fit_model"=holtwinters_fit_model, "forecast_ts"=forecast_timeseries)
  
  #return(holtwinters_fit_model)
  return(ret_list)
  
}
#------------------------------------------------------------------------------------

errors_forecast_HoltWinters <- function(holtwinters_forecast, max_lag=20)
{
  # Function to test if errors are normally distributed, 
  # non-significant autocorrelations, plot auto-correlogram
  # plot histogram of errors
  # requires plotForecastErrors function
  #
  # Input:
  # holtwinters_forecast: Holt Winters forecast object
  # max_lag: Maximum lag
  # Returns:
  # boxtest_stats (object returned by Box.test)
  
  acf(holtwinters_forecast$residuals, lag.max = max_lag)
  
  boxtest_stats <- Box.test(holtwinters_forecast$residuals, lag=max_lag, type="Ljung-Box")
  
  cat("Box-test stats: p-value",boxtest_stats$p.value, "statistic", boxtest_stats$statistic, "\n")
  
  # plot in-sample forecast errors over time
  plot.ts(holtwinters_forecast$residuals)
  
  # plot histogram of errors (are they normally distributed?)
  plotForecastErrors(holtwinters_forecast$residuals)
  
  return(boxtest_stats)
}
#------------------------------------------------------------------------------------


fit_forecast_ARIMA <- function(timeseries_data_handle, p, d, q, numpoints_forecast)
{
  # Function to fit ARIMA(p, d, q) model
  # requires librar(forecast)
  #
  # Input parameters:
  # timeseries_data_handle: time series object (non-stationary
  # p: AR order,
  # d: the degree of differencing
  # q: MA order
  # numpoints_forecast: number of points for which forecasting is to be done
  #
  # Returns:
  # arima_forecast: ARIMA forecast model
  
  require(forecast)
  
  fit_arima_model <- arima(timeseries_data_handle, order=c(p, d, q))
  
  arima_forecast <- forecast.Arima(fit_arima_model, h = numpoints_forecast)
  
  plot.forecast(arima_forecast, xlab="Years", ylab="Port throughput (AUD, imports)", pch=18, col="blue")
  
  # pick the best model based on AIC
  auto.arima(timeseries_data_handle,ic="bic")
  
  return(arima_forecast)
}
#------------------------------------------------------------------------------------

fit_best_ARIMA <- function(timeseries_data_handle,flag_exhaustive_search)
{
  # Function to fit BEST ARIMA(p, d, q) model
  # requires librar(forecast)
  #
  # Input parameters:
  # timeseries_data_handle: time series object (non-stationary)
  # flag_exhaustive_search: TRUE: do exhaustive search for best ARIMA model, FALSE: use auto.arima()
  #
  # Returns:
  # best_arima_forecast: BEST ARIMA forecast model (ARIMA object)
  
  # pick the best model based on AIC
  
  best_arima_forecast = NULL
  least_aic = 100000000 # set to a very large value
  
  if (flag_exhaustive_search == FALSE){
    best_arima_forecast <- auto.arima(timeseries_data_handle,ic="bic")    
  } 
  else {
    # perform exhaustive search
    
    #p = 1
    #d = 1
    #q = 1
    #P = 1
    #D = 1
    #Q = 1
    
    list_p = c(0,1,2,3,4,5)
    list_d = c(0,1,2,3,4,5)
    list_q = c(0,1,2,3,4,5)
    list_P = c(0,1,2,3,4)
    list_D = c(0,1,2,3,4)
    list_Q = c(0,1,2,3,4)
    
    for (p in list_p) {
      for (q in list_q){
        for (d in list_d) {
          for (P in list_P) {
            for (D in list_D) {
              for (Q in list_Q) {
                
                #
                fit_arima_model <- try( arima( timeseries_data_handle, order=c(p, d, q),
                                               seasonal = list(order = c(P, D, Q)) ) ,
                                        silent = TRUE )
                #cat(fit_arima_model$aic)
                #cat(grep(class(fit_arima_model),"try-error"))
                #x <- class(fit_arima_model)
                #cat(x,"\n")
                #cat(class(fit_arima_model),"\n")
                #if (grepl("try-error",x)){
                #cat((any(grepl("error", class(fit_arima_model)))))
                if (any(grepl("error", (fit_arima_model)))){
                  #if (x == "arima"){
                  if (fit_arima_model$aic < least_aic){
                    #cat(fit_arima_model$aic,"\n")
                    #cat("FOUND\n")
                    best_arima_forecast <- fit_arima_model
                  }
                  
                }
                #else
                #{
                #  cat("arima gave an error (skipping) \n")
                #}
                
                #
                
                
              }
            }
          }
          
        }
        
      }
    }
    
    
    #can also try fit_arima_model$loglik
    #arima_forecast <- forecast.Arima(fit_arima_model, h = numpoints_forecast)
    
  }
  
  
  return(best_arima_forecast)
}
#------------------------------------------------------------------------------------


calculate_model_accuracy <- function(forecast_object,test_data_timeseries)
{
  # Calculate model accuracy given generic forecast object and test data time series object
  
  diff_timeseries = forecast_object$mean - test_data_timeseries
  sse  = sum(diff_timeseries^2)
  cat('SSE',sse,'\n')
  mse  = sum(diff_timeseries^2)/length(diff_timeseries)
  cat('MSE',mse,'\n')
  rmse = sqrt(sum(diff_timeseries^2)/length(diff_timeseries))
  cat('RMSE',rmse,'\n')
  
  # TODO calculate AIC, AICc also from test data ?
  
  ret_list <- list("sse"=sse, "mse"=mse, "rmse"=rmse)
}
#------------------------------------------------------------------------------------


calculate_model_accuracy_VAR <- function(forecast_object, test_data_timeseries, 
                                         num_diff_inv, base_exponentiation, all_combined_data_log,
                                         target_field_name)
{
  # Calculate VAR model accuracy given generic forecast object and test data time series object
  
  diff_timeseries = (base_exponentiation^diffinv(forecast_object$mean$target_ts, 
                                                 differences = num_diff_inv, 
                                                 xi = all_combined_data_log[1,target_field_name]))
                    - test_data_timeseries
  sse  = sum(diff_timeseries^2)
  cat('SSE',sse,'\n')
  mse  = sum(diff_timeseries^2)/length(diff_timeseries)
  cat('MSE',mse,'\n')
  rmse = sqrt(sum(diff_timeseries^2)/length(diff_timeseries))
  cat('RMSE',rmse,'\n')
  
  # TODO calculate AIC, AICc also from test data ?
  
  ret_list <- list("sse"=sse, "mse"=mse, "rmse"=rmse)
}
#------------------------------------------------------------------------------------


find_best_VAR_model_bic <- function(all_combined_data_log_diff, max_lag_VAR)
{
  # Find and return VAR model with best BIC score
  require("vars")
  best_var_model <- VARselect(all_combined_data_log_diff, lag.max = max_lag_VAR, type = "const")
  cat("Best VAR model\n",best_var_model$criteria)
  cat("Swartchz criterion SC (same as BIC):",best_var_model$criteria[3,],"\n")
  x <- best_var_model$criteria[3,]
  x <- setdiff(x, c(Inf, -Inf))  # Remove Inf
  temp_min_index = which(x == min(x, na.rm = TRUE)) # Find min BIC value. Note: can also use order()
  x[temp_min_index]
  # temp_min_index
  best_lag_VAR = temp_min_index # The index/lag at which BIC is minimum
  
  return(best_lag_VAR)
}
#------------------------------------------------------------------------------------


find_best_VAR_model_residuals <- function(stdized_all_combined_data, max_lag_VAR_residuals)
{
  # Find and return VAR model with best BIC score
  require("vars")
  
  array_lag_var = seq(1,max_lag_VAR_residuals)
  for (temp_var in array_lag_var) {
    temp_var_model <- try(
      VAR(stdized_all_combined_data, p = temp_var, type = "const" ),
      silent = TRUE
    )
    try(
      summary(temp_var_model),
      silent = TRUE
    )
    
    # look at errors
    cat("For lag = ", temp_var, "\n")
    try(
      serial.test(temp_var_model, lags.pt = temp_var, type = "PT.asymptotic"),
      silent = TRUE
    )
    
  }
  
  # this should return something
  # return()
  
  
}

