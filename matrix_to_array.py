# Below code is for converting 2-d munpy array of
#  a pandas dataframe of power data
# given in the format of date (365 days) vs (96 timestamps)
#  so it is a 2-dimensional data
# But as we want data for every 15 min we will convert it into 1-d array
# by using .flatten()

import numpy as np
import pandas as pd

# df = pd.read_csv("one_month_power.csv").to_numpy().flatten()
df = pd.read_csv("2021_weather_lastmonth.csv").to_numpy().flatten()
# .to_numpy().flatten()
# print(df)
df1=pd.DataFrame(df)
# print(df.info)

# df1.to_csv("flattened_power_4.csv")
# df1.to_csv("flattened_weather_1.csv")
# ----------------------------------------------------------
# Below code is for reading values for the weather data
# observed that many values were mssing and many very not 
# present at 15 min interval so resampled the data at 15 min interval
# and filled missing values using KNNImputer

import pandas as pd
from sklearn.impute import KNNImputer

# Load the CSV file into a DataFrame
df = pd.read_csv('convertcsv (7).csv')

# Convert the 'Timestamp' column to a datetime object and set it as the index
df.loc[:,'date'] = pd.to_datetime(df.date.astype(str)+' '+df.time.astype(str))
# df['time'] = pd.to_datetime(df['time'])
df.set_index('date', inplace=True)

# Resample the DataFrame to create time steps for every 15 minutes
df = df.resample('15T').mean()

# Use KNN imputer to fill the missing values in the DataFrame
imputer = KNNImputer(n_neighbors=5)
df_filled = imputer.fit_transform(df)
df_filled = pd.DataFrame(df_filled, index=df.index, columns=df.columns)

# Save the filled DataFrame to a new CSV file
# df_filled.to_csv('data.csv')

# print(new)
# new.to_csv("weather_data_missed.csv")

# ----------------------------------------------------------
# array=['00:00:00',
# '00:15:00',
# '00:30:00',
# '00:45:00',
# '01:00:00',
# '01:15:00',
# '01:30:00',
# '01:45:00',
# '02:00:00',
# '02:15:00',
# '02:30:00',
# '02:45:00',
# '03:00:00',
# '03:15:00',
# '03:30:00',
# '03:45:00',
# '04:00:00',
# '04:15:00',
# '04:30:00',
# '04:45:00',
# '05:00:00',
# '05:15:00',
# '05:30:00',
# '05:45:00',
# '06:00:00',
# '06:15:00',
# '06:30:00',
# '06:45:00',
# '07:00:00',
# '07:15:00',
# '07:30:00',
# '07:45:00',
# '08:00:00',
# '08:15:00',
# '08:30:00',
# '08:45:00',
# '09:00:00',
# '09:15:00',
# '09:30:00',
# '09:45:00',
# '10:00:00',
# '10:15:00',
# '10:30:00',
# '10:45:00',
# '11:00:00',
# '11:15:00',
# '11:30:00',
# '11:45:00',
# '12:00:00',
# '12:15:00',
# '12:30:00',
# '12:45:00',
# '13:00:00',
# '13:15:00',
# '13:30:00',
# '13:45:00',
# '14:00:00',
# '14:15:00',
# '14:30:00',
# '14:45:00',
# '15:00:00',
# '15:15:00',
# '15:30:00',
# '15:45:00',
# '16:00:00',
# '16:15:00',
# '16:30:00',
# '16:45:00',
# '17:00:00',
# '17:15:00',
# '17:30:00',
# '17:45:00',
# '18:00:00',
# '18:15:00',
# '18:30:00',
# '18:45:00',
# '19:00:00',
# '19:15:00',
# '19:30:00',
# '19:45:00',
# '20:00:00',
# '20:15:00',
# '20:30:00',
# '20:45:00',
# '21:00:00',
# '21:15:00',
# '21:30:00',
# '21:45:00',
# '22:00:00',
# '22:15:00',
# '22:30:00',
# '22:45:00',
# '23:00:00',
# '23:15:00',
# '23:30:00',
# '23:45:00'
# ]
# test_list1 = array
# test_list2 = array
# # print(len(array))
# # using naive method to concat
# y=(test_list1)*31
# # +test_list1

# # for i in range(2) :
# #     print(i)
# #     test_list1.append(test_list1)
 
# # Printing concatenated list
# # print ("Concatenated list using naive method : "
#                             #   + str(y))

# # print(len(y))
# df1=pd.DataFrame(np.array(y))
# df1.to_csv("flattened_power_6.csv")


# ----------------------------------------------------------
# Below code is for generating dates of an year
# where each date needs to be repeated 96 times and
# all such 365*96 values need to be output to a csv file
import datetime

start = datetime.date(2021,12,1)

# initializing K
k = 31

res = []

for day in range(k):
	date = (start + datetime.timedelta(days = day)).isoformat()
	res.append(date)


df1=pd.DataFrame(np.array(res).repeat(96))
# df1.to_csv("flattened_power_5.csv")

# print(df1)
# # printing result
# print("Next K dates list: " + str(res))
# -----------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Load the data from CSV file
data = pd.read_csv('flattened_power_5.csv', parse_dates=[['date', 'time']])
data = data.set_index('date_time')

# Split the data into train and test sets
train_data = data.loc['2021-12-01 00:00:00':'2022-11-30 23:45:00']
test_data = data.loc['2022-12-01 00:00:00':'2022-12-31 23:45:00']

# Normalize the data
mean = train_data.mean()
std = train_data.std()
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

# Define the function to create the input and output sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length])
        y.append(data.iloc[i+seq_length])
    return np.array(X), np.array(y)

# Define the sequence length
seq_length = 4*24*7 # 4 weeks

# Create the input and output sequences for training and testing
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(y_train.shape[1]))
model.compile(loss='mse', optimizer='adam')

# Train the LSTM model
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.1, verbose=1)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Denormalize the data
y_pred = (y_pred * std) + mean
y_test = (y_test * std) + mean

# Define the function to calculate the error
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Calculate the error for every 15 minutes and multiples of 15 minutes
step = 15
error = []
for i in range(0, len(y_test), step):
    error.append(rmse(y_pred[i:i+step], y_test[i:i+step]))

# Plot the error vs time step predicted
plt.plot(np.arange(0, len(y_test), step), error)
plt.xlabel('Time step predicted (in 15 minute intervals)')
plt.ylabel('RMSE')
plt.show()