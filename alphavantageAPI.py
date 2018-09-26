from pandas_datareader import data
import matplotlib.pyplot as plt 
import pandas as pd 
import datetime as dt 
import urllib.request, json 
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

data_source = "alphavantage"
api_key = "TCJIE94ECXFL3UMT"
ticker = "GOOG"
url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
file_to_save = "stock_market_data_%s.csv"%ticker
print ("Testing: ", file_to_save)
if not os.path.exists(file_to_save):
    with urllib.request.urlopen(url_string) as url:
        data = json.loads(url.read().decode())
        # extract stock market data
        data = data['Time Series (Daily)']
        df = pd.DataFrame(columns=['Date','Low','High','Close','Open'])
        for k,v in data.items():
            date = dt.datetime.strptime(k, '%Y-%m-%d')
            data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                        float(v['4. close']),float(v['1. open'])]
            df.loc[-1,:] = data_row
            df.index = df.index + 1
    print('Data saved to : %s'%file_to_save)        
    df.to_csv(file_to_save)

# If the data is already there, just load it from the CSV
else:
    print('File already exists. Loading data from CSV')
    df = pd.read_csv(file_to_save)
df = df.sort_values("Date")
# print (df.head(5))

# plt.figure(figsize = (18,9))
# plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
# plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
# plt.xlabel('Date',fontsize=18)
# plt.ylabel('Mid Price',fontsize=18)
# plt.show()

# First calculate the mid prices from the highest and lowest
high_prices = df.loc[:,'High'].as_matrix()
low_prices = df.loc[:,'Low'].as_matrix()
mid_prices = (high_prices+low_prices)/2.0

# Determine train data length to be 70% of the mid_prices array
train_data_len = int (len(mid_prices)*0.7)
train_data = mid_prices[:train_data_len]
test_data = mid_prices[train_data_len:]
# print ("Train Data: ", train_data)
# print ("Test Data: ", test_data)
# print ("Length: ", len(train_data))
# print ("Length: ", len(test_data))

# Scale the data to be between 0 and 1
# When scaling remember! You normalize both test and train data with respect to training data
# Because you are not supposed to have access to test data
# scaler = MinMaxScaler()
# train_data = train_data.reshape(-1,1)
# test_data = test_data.reshape(-1,1)
# # Train the Scaler with training data and smooth data
# smoothing_window_size = 2500
# for di in range(0,10000,smoothing_window_size):
#     scaler.fit(train_data[di:di+smoothing_window_size,:])
#     train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

# # You normalize the last bit of remaining data
# scaler.fit(train_data[di+smoothing_window_size:,:])
# train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

# # Reshape both train and test data
# train_data = train_data.reshape(-1)

# # Normalize test data
# test_data = scaler.transform(test_data).reshape(-1)
