#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/10/07/get-rich-quick-or-the-μ-strategy/
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


# Get some data, for the post I used monthly rebalancing for SPY, DBA, TNA and EEM
tickers = ['SPY', 'DBA', 'TNA', 'EEM']

# Forecasts are based on signs
use_signs = True

# Set the rolling windows for the speculative complexity averaging
set_R1 = 2
set_RM = 4
nroll = np.arange(set_R1, set_RM+1, 1)
# also set the weighting type
set_weight_type = 2

# Then, a loop over the tickers
for ticker in tickers:
    data = yf.download(ticker, period='max', interval='1mo')['Adj Close'].dropna()
    r = data.pct_change().dropna().loc['2021-01-01':]
    y = r
    if use_signs:
        y = np.sign(r)

    # Compute the complexity mean for the forecast
    frc = complexity_weighted_mean(y, nroll, set_weight_type).apply(np.sign)

    # and then trade according to the rules of the post
    standard = r*frc.shift(periods=1)
    predict_long = (frc.shift(periods=2) == r.shift(periods=1).apply(np.sign))
    predict_long_short = 2*predict_long - 1
    mu_long = r*predict_long
    mu_long_short = r*predict_long_short

    # Collect the results, remove impact of maximum rolling window
    rr = pd.concat([r, standard, mu_long, mu_long_short], axis=1).iloc[nroll[-1]:]
    rr.columns = [ticker, 'complexity', 'μ-long', 'μ-long/short']

    # Do the plot
    tr = ((rr+1).cumprod()-1)*100
    tr.plot(title='Total trading returns of the μ-strategy in '+ticker, color=['red', 'black', 'blue', 'green'], ylabel='return in percent')
    plt.grid(visible=True, which='both')
    plt.show()

    # and print the total trading return, along with the window end-points
    print("End-points of windows are: ", nroll[0], nroll[-1])
    print(tr.iloc[-1])

    tr[[ticker, 'μ-long/short']].plot(title='Total trading returns of the μ-strategy in '+ticker, color=['red', 'green'], ylabel='return in percent')
    plt.grid(visible=True, which='both')
    plt.show()
