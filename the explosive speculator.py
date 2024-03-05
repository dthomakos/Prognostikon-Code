#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-explosive-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Select ticker to analyze
ticker = 'BTC-USD'

# Set starting and ending dates and rebalancing frequency
start_date = '2022-01-01'
end_date = '2024-02-29'
freq = '1d'
# Set some default values for the initial window
if freq == '1d':
    ini_wind = 21
elif freq == '1wk':
    ini_wind = 26
elif freq == '1mo':
    ini_wind = 24
# Set the max number of cross-correlations to consider
roll_max = 10

# Import the data
data = yf.download(ticker, start=start_date, end=end_date, interval=freq)['Adj Close'].dropna()
# Compute the returns and absolute returns
r = data.pct_change().dropna()
v = r.abs()

# Get the sample size and initialize storage
nobs = r.shape[0]
store = pd.DataFrame(data=None, index=r.index, columns=[ticker, 'ES-max', 'ES-min', 'ES-max+1', 'ES-min+1'])
store_abs = pd.DataFrame(data=None, index=r.index, columns=[ticker, 'ES-max', 'ES-min', 'ES-max+1', 'ES-min+1'])

# The evaluation loop
for i in np.arange(0, nobs-ini_wind, 1):
    # Crop to get the training data
    ri = r.iloc[i:(i+ini_wind)]
    vi = v.iloc[i:(i+ini_wind)]
    # Set the rolling windows for the cross-correlation computation
    roll = np.arange(1, roll_max+1, 1)
    # and storage
    store_corr = pd.Series(data=None, index=roll, name='Corr', dtype='float64')

    # Next compute all the cross-correlations
    for j in roll:
        cr = pd.concat([vi.shift(periods=j), ri], axis=1).dropna().corr().iloc[1,0]
        store_corr.loc[j] = cr

    # Compute the minimum and maximum, also in absolute values, of the cross-correlations
    actual = r.iloc[ini_wind+i]
    max_corr = store_corr.argmax()+1
    min_corr = store_corr.argmin()+1
    max_abs_corr = store_corr.abs().argmax()+1
    min_abs_corr = store_corr.abs().argmin()+1

    # Compute the signals, based on the change in the lagged volatility, and trade them
    signal_max = np.sign(vi.diff().shift(periods=max_corr).iloc[-1])
    signal_min = np.sign(vi.diff().shift(periods=min_corr).iloc[-1])
    signal_max1 = np.sign(vi.diff().shift(periods=max_corr+1).iloc[-1])
    signal_min1 = np.sign(vi.diff().shift(periods=min_corr+1).iloc[-1])
    store.iloc[ini_wind+i] = np.hstack([actual, actual*signal_max, actual*signal_min, actual*signal_max1, actual*signal_min1])
    # repeat for the absolute cross-correlations
    signal_abs_max = np.sign(vi.diff().shift(periods=max_abs_corr).iloc[-1])
    signal_abs_min = np.sign(vi.diff().shift(periods=min_abs_corr).iloc[-1])
    signal_max1 = np.sign(vi.diff().shift(periods=max_abs_corr+1).iloc[-1])
    signal_min1 = np.sign(vi.diff().shift(periods=min_abs_corr+1).iloc[-1])
    store_abs.iloc[ini_wind+i] = np.hstack([actual, actual*signal_max, actual*signal_min, actual*signal_max1, actual*signal_min1])

# Done, drop the NAs, compute the total return
store = store.dropna()
store_abs = store_abs.dropna()
cr = ((store+1).cumprod()-1)*100
cr_abs = ((store_abs+1).cumprod()-1)*100

# Plot and print
# cr[[ticker, 'ES-max+1']].plot(title='The explosive speculator strategy for '+ticker+', daily rebalancing', xlabel='Date', ylabel='return in percent', grid='both')
# plt.show()
#
print(cr.iloc[-1])
print(cr_abs.iloc[-1])
