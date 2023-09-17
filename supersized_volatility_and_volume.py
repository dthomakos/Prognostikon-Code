#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/09/17/supersized-volatility-and-volume-as-signal-enhancers/
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
import yfinance as yf
import math

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

# A function to get the complexity-weighted sample mean, with complexity measured by the sample size
def complexity_weighted_mean(x, nroll, weight_type=0):
    z = pd.DataFrame(data=None, index=x.index, columns=nroll)
    if weight_type == 0:
        scale = len(nroll)
        for i in range(len(nroll)):
            z[nroll[i]] = (x.rolling(window=nroll[i]).mean())
    elif weight_type == 1:
        scale = (2.0**(-nroll)).sum()
        for i in range(len(nroll)):
            z[nroll[i]] = (2.0**(-nroll[i]))*(x.rolling(window=nroll[i]).mean())
    elif weight_type == 2:
        scale = (2.0**(-logstar(nroll))).sum()
        for i in range(len(nroll)):
            z[nroll[i]] = (2.0**(-recursiveLogStar(nroll[i])))*(x.rolling(window=nroll[i]).mean())
    return z.sum(axis=1)/scale

# Select asset to work with and parametrization
ticker = 'TNA'
set_R1 = 2
set_RM = 12
# Select averaging type, 0 for equal, 1 for complexity, 2 for prior
set_avg_type = 1
# Select type of variable boosting, 'volatility', 'volume' or 'both'
set_booster = 'both'
# Set frequency of data, '1d', '1wk', '1mo'
set_freq = '1wk'
# Set starting date
set_start = '2021-01-01'

# Download the data, note the weekly, '1wk' frequency below
data = yf.download(ticker, period='max', interval=set_freq)[['High', 'Low', 'Adj Close', 'Volume']].dropna()
data = data.loc[set_start:]

# Get the variables, pure returns first
r = data['Adj Close'].pct_change().dropna()

# Then the booster variables
if set_booster == 'volatility':
    boost = (data['High']/data['Low'].shift(periods=1)-1)
elif set_booster == 'volume':
    boost = data['Volume']/data['Volume'].shift(periods=1)
elif set_booster == 'both':
    h = (data['High']/data['Low'].shift(periods=1)-1)
    v = data['Volume']/data['Volume'].shift(periods=1)
    boost = h*v
# OK, now get the booster-augmented returns
y = r*boost

# Get the rolling windows
nroll = np.arange(set_R1, set_RM+1, 1)
roll = nroll[0]

# Compute the rolling means and apply rolling window averaging, with and without the booster variable
#
# First, without the boosting
mu0 = complexity_weighted_mean(r, nroll, set_avg_type)
# and then with the boosting
mu1 = complexity_weighted_mean(y, nroll, set_avg_type)

# Trade next, easy!
str0 = r*(mu0.shift(periods=1).apply(np.sign))
str1 = r*(mu1.shift(periods=1).apply(np.sign))

# Collect the results, remove impact of maximum rolling window
rr = pd.concat([r, str0, str1], axis=1).iloc[nroll[-1]:]
rr.columns = [ticker, 'spec.complexity', 'supersized']

# Do the plot
tr = ((rr+1).cumprod()-1)*100
tr.plot(title='Total trading returns of the speculative complexity strategy supersized for '+ticker, color=['red', 'black', 'blue', 'green'], ylabel='return in percent')
plt.grid(visible=True, which='both')
plt.show()

# and print the total trading return, along with the window end-points
print("End-points of windows are: ", nroll[0], nroll[-1])
print(tr.iloc[-1])

# tr[[ticker, 'supersized']].plot(title='Total trading returns of the speculative complexity strategy in '+ticker, color=['red', 'blue'], ylabel='return in percent')
# plt.grid(visible=True, which='both')
# plt.show()