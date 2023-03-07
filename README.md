# FlightDelay

## Overview
A 2015 dataset of flight delays across USA is detailed in csv files listing the time of departure, time of arrival, airports at which a flight departed and arrived, airlines, and more. The month of January was extracted with a dataset size of approximately half a million flights. The dataset used can be found here: https://www.kaggle.com/datasets/usdot/flight-delays

## Details of the Project
* The location of each airport is shown to give a visualization.
* A visualization of the average delay of arrival and departure per airport is shown with:
  * Green Data: No Delay
  * Yellow Data: 15 MIN Delay Or Less
  * Orange Data: 30 MIN Delay Or Less
  * Red Data: 30+ MIN Delay
* A visualization of the average departure delay per airline is shown to help capture airlines that struggle with punctuality.
* An additional plot shows the total number of cancelled flights per airline.
* Predictive models (linear regression and ridge) are used for model analysis with R squared values of 0.99.
