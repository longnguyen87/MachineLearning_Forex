import requests
import pandas as pd
from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime as datetime

# Get OHLC data from kraken api [time,open,high,low,close,vwap,volume,count]
ticker='XXBTZEUR'   
period='5'      
starting='1505677500'

parameters={"pair":ticker,"interval":period,"since":starting}
response=requests.get("https://api.kraken.com/0/public/OHLC", params=parameters)
krakohlc=response.json()['result'][ticker]

ohlc=[]

for i in range(len(krakohlc)):
    ohlcdata=krakohlc[i][0:5]
    ohlc.append(ohlcdata)       #Make data array (time,O,H,L,C)

labels = ['Date', 'Open', 'High', 'Low', 'Close']
ohlc_df=pd.DataFrame.from_records(ohlc, columns=labels)
# cast data to float
ohlc_df = ohlc_df.astype(float)
# convert timestamp column to matplotlib date numbers
f = lambda x: mdates.date2num(datetime.datetime.fromtimestamp(x))
ohlc_df[labels[0]] = ohlc_df[labels[0]].apply(f)

#Making plot area
fig = plt.figure()
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=6, colspan=1)

#Making candlestick plot
width = 1./(24*60)*5  # make candles 5 minutes wide
candlestick_ohlc(ax1,ohlc_df.values,width=width, colorup='g', colordown='k',alpha=0.75)
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
ax1.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,12]))
fig.autofmt_xdate()
ax1.grid(True)

plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()