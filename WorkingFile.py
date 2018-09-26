"""
Author: Long Nguyen
Date: 09/14/18
Purpose: Adding technical indicators to historical data of currency pair.

"""

"""
First Step: Read data from the .csv files into the program. 
After successfully read data from .csv files - add moving average indicators, RSI
Split data into training set and validation set
Since the purpose of this ML is to predict future price, closing price will be shifting forward by 1 or n depends on need

"""

import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
from  source import technical_indicators as ta 
import pandas as pd 

data = pd.read_csv("EURUSD240.csv", na_values = ['no info','.'])

print ("Shape: ", data.shape)
data = ta.moving_average(data, 5)
data = ta.moving_average(data, 15)
data = ta.moving_average(data, 30)
data = ta.relative_strength_index (data, 14)

train_size = int (data.shape[0] * 0.7)
validation_size = data.shape[0]- train_size

print ("Train Size: ", train_size)
print ("Validation Size: ", validation_size)
print ("New Shape: ", data.shape)
print (data.head(3))
print ("New data with closing price shifting forward")
data["Close"][:-1] = data["Close"][1:]
# data["Close"][-1] = np.NaN                 # set the last closing price to be NaN
print (data.head(3))
print (data.tail(3))


