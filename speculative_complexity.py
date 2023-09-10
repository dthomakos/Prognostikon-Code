#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/09/10/speculative-complexity/
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

# Select whether the end-points for the rolling window are fixed or random
do_random = False

# Download some data - for the post used 'FXE' and 'DBA'
tickers = ['FXE', 'DBA', 'DBC', 'DBB', 'USCI']
for ticker in tickers:
    data = yf.download(ticker, period='max', interval='1mo')['Adj Close'].dropna()
    # Get percent returns, select period of estimation/evaluation
    y = data.pct_change().dropna()#.loc['2018-01-01':] # uncomment to start from another date

    # Compute the rolling mean and apply rolling window averaging
    #
    # First select the number of rolling windows, fixed or random!
    if do_random:
        # You can of course experiment with the values below
        R1 = np.random.randint(low=2, high=5, size=1)
        RM = np.random.randint(low=6, high=24, size=1)
    else:
        # As you can experiment below as well!
        R1 = 4
        RM = 18
    nroll = np.arange(R1, RM+1, 1)
    roll = nroll[0]
    # Compute all three weighted means
    mu0 = complexity_weighted_mean(y, nroll, 0)
    mu1 = complexity_weighted_mean(y, nroll, 1)
    mu2 = complexity_weighted_mean(y, nroll, 2)

    # Trade next, easy!
    str0 = y*(mu0.shift(periods=1).apply(np.sign))
    str1 = y*(mu1.shift(periods=1).apply(np.sign))
    str2 = y*(mu2.shift(periods=1).apply(np.sign))

    # Collect the results, remove impact of maximum rolling window
    rr = pd.concat([y, str0, str1, str2], axis=1).iloc[nroll[-1]:]
    rr.columns = [ticker, 'equal weighting', 'complexity weighting', 'prior weighting']

    # Evaluate from 2012 for comparability
    rr = rr.loc['2012-01-01':]

    # Do the plot
    tr = ((rr+1).cumprod()-1)*100
    tr.plot(title='Total trading returns of the speculative complexity strategy in '+ticker, color=['red', 'black', 'blue', 'green'], ylabel='return in percent')
    plt.grid(visible=True, which='both')
    plt.show()

    # and print the total trading return, along with the window end-points
    print("End-points of windows are: ", nroll[0], nroll[-1])
    print(tr.iloc[-1])

    # tr[[ticker, 'prior weighting']].plot(title='Total trading returns of the speculative complexity strategy in '+ticker, color=['red', 'blue'], ylabel='return in percent')
    # plt.grid(visible=True, which='both')
    # plt.show()