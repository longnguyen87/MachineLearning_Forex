"""
Author: Long Nguyen
Purpose: This module will help with getting financial data from alphavantageAPI. 
It will download data to the current folder
"""

# from pandas_datareader import data
import matplotlib.pyplot as plt 
import pandas as pd 
import datetime as dt 
import urllib.request, json, os
import numpy as np 


API_KEY = "TCJIE94ECXFL3UMT" # set API key use to request data
CURRENCY_FUNC = ["CURRENCY_EXCHANGE_RATE", "FX_INTRADAY", "FX_DAILY", "FX_WEEKLY", "FX_MONTHLY"]
# Define a get currency function
def getCurrency(func_name, from_symbol, to_symbol, outputsize= "full", datatype = "json"):
    """This function is currently working with FX_DAILY request function only
        Other features will be developed later.
    """
    if not (func_name in CURRENCY_FUNC):
        print ("Please enter function value as provided in this list: ", CURRENCY_FUNC)
        return
    url_string = "https://www.alphavantage.co/query?function=%s&from_symbol=%s&to_symbol=%s&outputsize=%s&dataype=%s&apikey=%s"%(func_name, from_symbol, to_symbol, outputsize, datatype, API_KEY)
    file_name = "FX_%s-%s" %(from_symbol, to_symbol)
    if not os.path.exists (file_name):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            if func_name ==  "FX_DAILY":
                data = data["Time Series FX (Daily)"]
                df = pd.DataFrame(columns = ['Date','Low','High','Close','Open'])
                for k, v in data.items():
                    date = dt.datetime.strptime(k, '%Y-%m-%d')
                    data_row = [date.date(),float(v['3. low']),float(v['2. high']),
                                float(v['4. close']),float(v['1. open'])]
                    df.loc[-1,:] = data_row
                    df.index = df.index + 1
        df.to_csv(file_name)
        print ("Complete download and process data")
    else:
        print ("File already exists. Loading data from CSV file")
        df = pd.read_csv(file_name)
    return df
# End getCurrency function

def plotData (data):
    # priceMin = int(min(data["Close"])*10000)
    # priceMax = int(max(data["Close"])*10000)
    # print ("PriceMin: ", priceMax)
    plt.figure(figsize=(20,10))
    plt.plot(range(data.shape[0]), (data["Close"])) # plot price using close values
    plt.xticks (range(0, data.shape[0], 100), data["Date"].loc[::100], rotation = 45)
    # plt.yticks (data["Close"])

    plt.xlabel ("Date", fontsize = 18)
    plt.ylabel ("Price", fontsize = 18)
    plt.show()





