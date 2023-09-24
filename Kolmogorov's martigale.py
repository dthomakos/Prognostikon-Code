#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/09/24/kolmogorovs-martigale/
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
from scipy import stats

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

# Define the gain of Kolmogorov's martigale within the complexity-based weighting, in two ways
def complexity_weighted_kg(x, nroll, weight_type=0):
    z = pd.DataFrame(data=None, index=x.index, columns=nroll)
    if weight_type == 0:
        scale = len(nroll)
        for i in range(len(nroll)):
            if nroll[i] > 0:
                sumx = x.rolling(window=nroll[i]).sum()
            else:
                sumx = x.expanding().sum()
            z[nroll[i]] = 2*x*(sumx.shift(periods=1))
    elif weight_type == 1:
        scale = (2.0**(-nroll)).sum()
        for i in range(len(nroll)):
            if nroll[i] > 0:
                sumx = x.rolling(window=nroll[i]).sum()
            else:
                sumx = x.expanding().sum()
            z[nroll[i]] = (2.0**(-nroll[i]))*(2*x*(sumx.shift(periods=1)))
    elif weight_type == 2:
        scale = (2.0**(-logstar(nroll))).sum()
        for i in range(len(nroll)):
            if nroll[i] > 0:
                sumx = x.rolling(window=nroll[i]).sum()
            else:
                sumx = x.expanding().sum()
            z[nroll[i]] = (2.0**(-recursiveLogStar(nroll[i])))*(2*x*(sumx.shift(periods=1)))
    return z.sum(axis=1)/scale

# Note that for the function below I have not written the expanding part!!!
def complexity_weighted_ekg(x, nroll, weight_type=0, L=3):
    z = pd.DataFrame(data=None, index=x.index, columns=nroll)
    seqL = np.arange(1, L+1, 1)
    if weight_type == 0:
        scale = len(nroll)
        for i in range(len(nroll)):
            zzz = pd.DataFrame(data=None, index=x.index, columns=seqL)
            for j in seqL:
                zzz[j] = x*(x.shift(periods=j))
            z[nroll[i]] = 2*(zzz.sum(axis=1).rolling(window=nroll[i]).mean())
    elif weight_type == 1:
        scale = (2.0**(-nroll)).sum()
        for i in range(len(nroll)):
            zzz = pd.DataFrame(data=None, index=x.index, columns=seqL)
            for j in seqL:
                zzz[j] = x*(x.shift(periods=j))
            z[nroll[i]] = (2.0**(-nroll[i]))*2*(zzz.sum(axis=1).rolling(window=nroll[i]).mean())
    elif weight_type == 2:
        scale = (2.0**(-logstar(nroll))).sum()
        for i in range(len(nroll)):
            zzz = pd.DataFrame(data=None, index=x.index, columns=seqL)
            for j in seqL:
                zzz[j] = x*(x.shift(periods=j))
            z[nroll[i]] = (2.0**(-recursiveLogStar(nroll[i])))*2*(zzz.sum(axis=1).rolling(window=nroll[i]).mean())
    return z.sum(axis=1)/scale


# Select whether to add the recursive computation in averaging
add_recursive = False

# Use signs for the computations?
use_signs = True

# Use expectation and cross-moments?
use_expectation = True

# If you use expectation select number of lags
set_L = 3

# Select evaluation start date: 2020-01-01 for monthly rebalancing and 2022-01-01 for weekly rebalancing
set_start_eval = '2020-01-01'

# Download some data
ticker = 'TNA'
freq = '1mo'
R1 = 2
RM = 8

data = yf.download(ticker, period='max', interval='1mo')['Adj Close'].dropna()
# Get percent returns, select period of estimation/evaluation
r = data.pct_change().dropna().loc['2018-01-01':] # uncomment to start from another date
y = r
if use_signs:
    y = np.sign(r)

# Compute the rolling mean and apply rolling window averaging
nroll = np.arange(R1, RM+1, 1)
if add_recursive:
    nroll = np.hstack([nroll, -1])
    end_roll = nroll[-2]
else:
    end_roll = nroll[-1]

# Compute all three weighted means
if use_expectation:
    mu0 = complexity_weighted_ekg(y, nroll, 0, set_L)
    mu1 = complexity_weighted_ekg(y, nroll, 1, set_L)
    mu2 = complexity_weighted_ekg(y, nroll, 2, set_L)
else:
    mu0 = complexity_weighted_kg(y, nroll, 0)
    mu1 = complexity_weighted_kg(y, nroll, 1)
    mu2 = complexity_weighted_kg(y, nroll, 2)

# Trade next, easy!
str0 = r*(mu0.shift(periods=1).apply(np.sign))
str1 = r*(mu1.shift(periods=1).apply(np.sign))
str2 = r*(mu2.shift(periods=1).apply(np.sign))

# Collect the results, remove impact of maximum rolling window
rr = pd.concat([r, str0, str1, str2], axis=1).iloc[end_roll:]
rr.columns = [ticker, 'equal weighting', 'complexity weighting', 'prior weighting']

# Evaluate from 2020 (monthly) or 2022 (weekly) for comparability
rr = rr.loc[set_start_eval:]

# Do the plot
tr = ((rr+1).cumprod()-1)*100
tr.plot(title='Total trading returns of Kolmogorov martingale strategy in '+ticker, color=['red', 'black', 'blue', 'green'], ylabel='return in percent')
plt.grid(visible=True, which='both')
plt.show()

# and print the total trading return, along with the window end-points
print("End-points of windows are: ", nroll[0], nroll[-1])
print(tr.iloc[-1])

tr[[ticker, 'prior weighting']].plot(title='Total trading returns of Kolmogorov martingale strategy in '+ticker, color=['red', 'black', 'blue', 'green'], ylabel='return in percent')
plt.grid(visible=True, which='both')
plt.show()
