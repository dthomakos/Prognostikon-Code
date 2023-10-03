#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/10/03/a-complex-neighbor-out-of-sample/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import pandas_datareader as pdr
import math
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.neighbors import NearestNeighbors

# A set of functions to obtain the iterated logarithm
def _log(x, base=math.e):
    return int(np.log(x) / np.log(base))

def recursiveLogStar(n, b=math.e):
    if n > 1.0:
        return 1.0 + recursiveLogStar(_log(n, b), b);
    else:
        return 0

def logstar(x):
    n = len(x)
    z = np.zeros([n, 1])
    for i in range(n):
        z[i] = recursiveLogStar(x[i]) # + np.log(2.865064) the extra constant not needed
    return z

# A function to apply complexity-based weighted for computing a weighted mean
def cwm(x, h_ahead, weight_type=0):
    if weight_type == 0:
        z = x.mean()
    elif weight_type == 1:
        w = 2.0 ** (-np.arange(1, h_ahead+1, 1))
        z = (w*x).sum()/w.sum()
    elif weight_type == 2:
        w = (2.0**(-logstar(np.arange(1, h_ahead+1, 1)))).flatten()
        z = (w*x).sum()/w.sum()
    return z

# Download some data - for the post I used:
# the US CPI-based inflation CPIAUCSL
# the global price of wheat PWHEAMTUSDM
# the global price of Brent oil POILBREUSDM
#
fred_tickers = ['POILBREUSDM', 'PWHEAMTUSDM', 'CPIAUCSL']

# Select if you want to forecast the growth rate or the level of the series but be careful:
# strictly speaking, the nearest neighbors require a stationary series to work well...
do_growth = True
# set periods for growth rate if true
set_periods = 12
# Set number of steps-ahead for your forecast
h_ahead = 24
# Set the type of weighting
set_weighting = 2

# and do a quick loop over the tickers to produce results
for ticker in fred_tickers:
    data = pdr.fred.FredReader(ticker, start='1990-01-01').read()
    if do_growth:
        y = data.pct_change(periods=set_periods).dropna()*100
    else:
        y = data

    # Get the nearest neighbors of the whole series
    knn = NearestNeighbors(n_neighbors=h_ahead+1)
    z = knn.fit(y)
    distance_mat, neighbours_mat = knn.kneighbors(y)
    # Extract the nearest neighbors of the last observation
    last = neighbours_mat[-1, 1:]

    # Create a new dates index
    tf = pd.date_range(y.index[-1]+relativedelta(months=1), periods=h_ahead, freq=y.index.inferred_freq)
    # and a dataframe to hold the nearest neighbors
    zf = pd.DataFrame(data=None, index=tf, columns=np.arange(1, h_ahead+1, 1))
    # Fill-in via a quick loop
    for i in np.arange(1, h_ahead+1, 1):
        seq = np.arange((last[i-1]+1),(last[i-1]+(h_ahead+1)))
        if any(seq >= y.shape[0]):
            set_len = np.where(seq < y.shape[0])[0]
            zf.iloc[set_len, i-1] = y.iloc[(last[i-1]+1):(last[i-1]+(h_ahead+1))].to_numpy()
        else:
            zf[i] = y.iloc[(last[i-1]+1):(last[i-1]+(h_ahead+1))].to_numpy()

    # And apply complexity-based weighting across the nearest neighbors
    yf = zf.apply(cwm, args=(h_ahead, set_weighting,), axis=1)
    yf = pd.DataFrame(data=yf, index=yf.index, columns=y.columns)

    # Print the forecast
    print(yf)

    # Merge the historical observations with the forecast, note the trick
    yt = pd.concat([y, yf], axis=1)
    yt.columns = np.hstack([y.columns.tolist(), y.columns+' forecast'])
    # Fill in the last historical value in the forecast
    yt.iloc[y.shape[0]-1,1] = y.iloc[-1]

    # Plot the last few values, including the forecast
    plot_last = 60
    if do_growth:
        set_title = 'Actual values and future forecast for '+str(set_periods)+'-period growth rate'
        set_ylabel = 'percent'
    else:
        set_title = 'Actual values and future forecast'
    #
    yt.iloc[-plot_last:].plot(grid='both', title=set_title, xlabel='Date', ylabel=set_ylabel)
    plt.show()