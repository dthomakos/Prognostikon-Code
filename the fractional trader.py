#
# Python code adding for the post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-fractional-trader-supercharged/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

## Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.fft import fft, fftfreq
from scipy.optimize import minimize

## Define the Whittle log-likelihood for the estimation of the fractional order
def loglf_Whittle(d, x):
    n = len(x)
    m = n//2
    y = fft(x)
    f = fftfreq(n)[:m]
    P = (2.0/n) * np.abs(y[:m])
    gd = 4.0*(np.sin(np.pi*f) ** 2)
    sd = np.mean((gd ** d)*P)
    loglf = m*np.log(sd) + d*np.sum(np.log(gd[1:])) + m
    return loglf

## Estimate the fractional order
def fractional_order(x):
    cr = (x+1).cumprod().to_numpy()
    out = minimize(loglf_Whittle, 0.5, method='SLSQP', bounds=((0, 1),), args=(cr, ))
    d = out.x[0]
    return d

## Get some data
ticker = 'TNA'
start_from = '2000-01-01'
freq = '1mo'
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
rets = data.pct_change().dropna()
rets = rets.loc[start_from:'2024-12-02']

## Set the parameters
set_delay = 2
set_min_roll = 8
set_max_roll = 26
roll_range = np.arange(set_min_roll, set_max_roll+1, 1)
kappa_range = [0.5, 0.6, 0.7]
total_cases = len(kappa_range)*len(roll_range)

# Superchange? Works for TNA, DBC, OIH!
supercharge = True

## Initialize storage
store_er = pd.DataFrame(data=None, index=range(total_cases), columns=['Excess Return', 'd-order'])
counter = 0

## Now a double loop for evaluation
for set_roll in roll_range:
    for set_kappa in kappa_range:

        ## Get the signals
        dhat = rets.rolling(window=set_roll).apply(fractional_order)
        signals = (dhat < set_kappa).astype(float) - (dhat >= set_kappa).astype(float)
        if supercharge:
            threshold = dhat.expanding().quantile(0.95)
            signals = (dhat < threshold).astype(float) - (dhat >= threshold).astype(float)
        strategy = rets*signals.shift(periods=set_delay)

        ## Merge, process and store
        both = pd.concat([rets, strategy], axis=1).iloc[set_roll:]
        dhat = dhat.iloc[set_roll:]
        #
        tr = ((both + 1).cumprod()-1)*100
        tr.columns = [ticker, 'Fractional trader']
        all = pd.concat([tr, dhat], axis=1)
        all.columns = [ticker, 'Strategy', 'd-order']
        #
        # all.plot(secondary_y='d-order')
        # plt.show()
        er = tr.iloc[-1,1] - tr.iloc[-1,0]
        store_er.iloc[counter] = np.hstack([er, dhat.mean()])
        counter = counter + 1

## Print the summary
print(store_er.astype(float).describe())
print((store_er['Excess Return'] > 0).mean())