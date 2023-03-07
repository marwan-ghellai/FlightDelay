import pandas as pd
import numpy as np
import scipy
import scipy.stats as st
import matplotlib.pyplot as plt
# allows for plots to always be displayed in Jupyter notebook
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



# Upload airlines.csv, airpots.csv, flights.csv
from google.colab import files
csvFiles = files.upload()

# Read file
airlines_data = pd.read_csv('airlines.csv')
airports_data = pd.read_csv('airports.csv')
flights_data = pd.read_csv('flights.csv')

# The column names as indices.
# The two brackets are required because you are passing a list of columns
loc_data = airports_data[['LATITUDE', 'LONGITUDE']]

fig, ax = plt.subplots()
# s=0.1 specifies the size
ax.scatter(loc_data.LONGITUDE, loc_data.LATITUDE, s=0.1)
ax.set_title('USA Airport Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude');
fig.set_dpi(200)

# [['ARRIVAL_DELAY','DEPARTURE_DELAY']] shows dataframe of delays
df = flights_data[['DESTINATION_AIRPORT','ARRIVAL_DELAY','DEPARTURE_DELAY']]
# adding .DESTINATION_AIRPORT at the end creates the rows to be CALLABLE
airports = df.DESTINATION_AIRPORT

# The following calculates the mean of all columns 
# and groups all similar names i.e. airports under the same group 
delay_mean = df.groupby(airports).mean()

# .sort_values will sort alphabetically the column specified in a csv file
key_array = flights_data.sort_values(['DESTINATION_AIRPORT'])
# adding .IATA_CODE at the end creates the rows to be CALLABLE
key_array = key_array.DESTINATION_AIRPORT

# .values allows for the mean values of arrival delay to be callable
mean_arr_array = delay_mean[['ARRIVAL_DELAY']].values[:,0]
# .around will round up values throughout the entire array
mean_arr_array = np.around(mean_arr_array, decimals=2)

# .values allows for the mean values of departure delay to be callable
mean_dep_array = delay_mean[['DEPARTURE_DELAY']].values[:,0]
# .around will round up values throughout the entire array
mean_dep_array = np.around(mean_dep_array, decimals=2)

# .sort_values will sort alphabetically the column specified in a csv file
key_array = flights_data.sort_values(['DESTINATION_AIRPORT'])
# adding .IATA_CODE at the end creates the rows to be CALLABLE
key_array = key_array.DESTINATION_AIRPORT
# drops any duplicates
key_array = key_array.drop_duplicates(keep = 'last')

# air_flightsdata and air_airportsdata creates a list of the airports from the csv files
air_flightsdata = flights_data['DESTINATION_AIRPORT'].sort_values()
air_airportsdata = airports_data['IATA_CODE']
# coordinate data of airports
longitude_coor = airports_data['LONGITUDE']
latitude_coor = airports_data['LATITUDE']

# removal of duplications
air_flightsdata = air_flightsdata.drop_duplicates()

# the variables are created into lists
# a for loop goes through the data in the airport csv file and checks if
# it is amongst the flights csv file. Any airport that does not match has the index stored
# .count(example) checks if the example is found in the list interested in
removeindex = []
air_airportsdata = list(air_airportsdata)
air_flightsdata = list(air_flightsdata)
longitude_coor = list(longitude_coor)
latitude_coor = list(latitude_coor)
for i in range(len(air_airportsdata)):
  if air_flightsdata.count(air_airportsdata[i]) < 1:
    removeindex.append(i)

# the loop uses the stored index from the previous code block
# removing airports that are not shared between the two csv sheets
for i in range(len(removeindex)):
  k = removeindex[i] - i
  air_airportsdata.remove(air_airportsdata[k])
  longitude_coor.remove(longitude_coor[k])
  latitude_coor.remove(latitude_coor[k])

# dictionary created for airport stats
airport_stats = {}
i = 0
for key in key_array:
  airport_stats[key] = {'Mean Arrival Delay': mean_arr_array[i], 'Mean Departure Delay': mean_dep_array[i],
                        'LONGITUDE': longitude_coor[i], 'LATITUDE': latitude_coor[i]}
  i += 1

# Arrival Delay Figure
fig, ax = plt.subplots()
# s=0.2 specifies the size
for key in key_array:
  if airport_stats[key]['Mean Arrival Delay'] <= 0:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=0.1, color = 'green')
  elif airport_stats[key]['Mean Arrival Delay'] <= 15:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=1, color = 'yellow')
  elif airport_stats[key]['Mean Arrival Delay'] <= 30:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=2, color = 'orange')
  else:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=5, color = 'red')
ax.set_title('Delay of Arrivals at USA Airport Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude');
fig.set_dpi(200)

# Departure Delay Figure
fig, ax = plt.subplots()
# s=0.2 specifies the size
for key in key_array:
  if airport_stats[key]['Mean Departure Delay'] <= 0:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=0.1, color = 'green')
  elif airport_stats[key]['Mean Departure Delay'] <= 15:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=1, color = 'yellow')
  elif airport_stats[key]['Mean Departure Delay'] <= 30:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=2, color = 'orange')
  else:
    ax.scatter(airport_stats[key]['LONGITUDE'], airport_stats[key]['LATITUDE'], s=5, color = 'red')
ax.set_title('Delay of Departures at USA Airport Locations')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude');
fig.set_dpi(200)

# [['AIRLINE']] shows dataframe of airlines
df = flights_data[['AIRLINE', 'DEPARTURE_DELAY']]

# adding .AIRLINE at the end creates the rows to be CALLABLE
airlines = df.AIRLINE

# The following calculates the mean of all columns 
# and groups all similar names i.e. airlines under the same group 
dep_delay_mean = df.groupby(airlines).mean()

# .sort_values will sort alphabetically the column specified in a csv file
key_array = airlines_data.sort_values(['IATA_CODE'])
# adding .IATA_CODE at the end creates the rows to be CALLABLE
key_array = key_array.IATA_CODE

# .values allows for the mean values of departure delay to be callable
mean_val_array = dep_delay_mean[['DEPARTURE_DELAY']].values[:,0]
# .around will round up values throughout the entire array
mean_val_array = np.around(mean_val_array, decimals=2)

# The LHS creates a new column, restoring it with the cancelled flights from flights.csv
df['CANCELLED'] = flights_data['CANCELLED']
# remove departure delay
df = df.drop(['DEPARTURE_DELAY'], axis = 1)

# The following sums the total for all columns 
# and groups all similar names i.e. airlines under the same group 
canceled_flight_sum = df.groupby(airlines).sum()

# .values allows for the sum values of cancelled flights to be callable
sum_val_array = canceled_flight_sum [['CANCELLED']].values[:,0]
# .around will round up values throughout the entire array
sum_val_array = np.around(sum_val_array, decimals=2)

# Loop through each key and adds to an empty dictionary
# adding the appropriate stats for each key
airline_stats = {}
i = 0
for key in key_array:
  airline_stats[key] = {'Mean Departure Delay': mean_val_array[i], 
            'Cancelled Flights': sum_val_array[i]}
  i += 1

# Turn dictionary keys to a list 
# Following can also be done list(key_arrays)
airlines_list = list(airline_stats.keys())

fig1 = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(airlines_list, sum_val_array, color ='maroon')
plt.xlabel("Airline")
plt.ylabel("No. of Cancelled Flights")
plt.title("Cancelled Flights")
plt.show()

print()

fig2 = plt.figure(figsize = (10, 5))
# creating the bar plot
plt.bar(airlines_list, mean_val_array, color ='maroon')
plt.xlabel("Airline")
plt.ylabel("Avg. Delay (mins.)")
plt.title("Average Delay")
plt.show()



# dftrain drops cols listed, keeping the cols we want to work with
dftrain = flights_data.drop(['MONTH', 'DAY','CANCELLATION_REASON', 'AIR_SYSTEM_DELAY',
                        'SECURITY_DELAY', 'AIRLINE_DELAY',
                        'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'], axis = 1)
dftrain = dftrain.dropna()
# pred_input is all columns except the column being predicted
# pred is the column that will be predicted
pred_input = dftrain.drop('ARRIVAL_DELAY', axis = 1)
ydata = dftrain['ARRIVAL_DELAY']

# Train data
x_train, x_test, y_train, y_test = train_test_split(pred_input, ydata, test_size = 0.30)

# .info() helps us see what is not an int or float which will then be removed
# this is necessary for standard scaling of data
dftrain.info()

# Airline, origin airport, and destination airport are dropped as they are not int or float
x_train = x_train.drop(['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT'], axis = 1)
x_test = x_test.drop(['AIRLINE','ORIGIN_AIRPORT','DESTINATION_AIRPORT'], axis = 1)

# train and test data is standardized. standard scaling is explained well here:
# https://www.digitalocean.com/community/tutorials/standardscaler-function-in-python

sc = StandardScaler()
X_train_sc = sc.fit_transform(x_train)
X_test_sc = sc.transform(x_test)

# models used for training
LinR = LinearRegression()
Rid = Ridge()

# Output of model prediction vs truth as well as MSE, MAE, Root MSE, and R2
for model, name in zip([LinR, Rid],  ['Linear Regression','Ridge']):
    model1 = model.fit(X_train_sc,y_train)
    Y_predict=model1.predict(X_test_sc)
    print(name)
    print()
    plt.scatter(y_test, Y_predict)
    plt.title("Model Analysis")
    plt.xlabel("Truth")
    plt.ylabel("Prediction")
    plt.show()
    print('Mean Absolute Error:', mean_absolute_error(y_test, Y_predict))  
    print('Mean Squared Error:', mean_squared_error(y_test, Y_predict))  
    print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_test, Y_predict)))
    print('R2 : ',r2_score(y_test, Y_predict))
    print()
