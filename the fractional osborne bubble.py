#
# Python code adding material for the results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/gbm-and-the-osborne-bubble/
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

## Define the trading function
def fractional_trader(x, kappa=0.5):
    cr = (x+1).cumprod().to_numpy()
    out = minimize(loglf_Whittle, 0.5, method='SLSQP', bounds=((0, 1),), args=(cr, ))
    d = out.x[0]
    #
    if (d > kappa):
        return -1.0
    else:
        return +1.0

## Get some data
ticker = 'OIH'
start_from = '2000-01-01'
freq = '1mo'
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
rets = data.pct_change().dropna()
rets = rets.loc[start_from:'2024-12-02']

## Set the parameters and initialize storage
kappa = [0.3] # [0.5, 0.6, 0.7, 0.8]

## Do a loop over the values of kappa
for set_kappa in kappa:
    min_tau = 4
    max_tau = 36
    tau = np.arange(min_tau, max_tau+1, 4)
    store = pd.DataFrame(data=None, index=rets.index, columns=tau)
    signals = pd.DataFrame(data=None, index=rets.index, columns=tau)
    avg_signal = pd.DataFrame(data=None, index=rets.index, columns=['Avg'])
    avg = 0

    # Get the signals and the average signal
    for i in tau:
        si = rets.rolling(window=i).apply(fractional_trader, args=(set_kappa,))
        signals[i] = si
        store[i] = si.shift(periods=1)*rets
        avg = avg + si
        avg_signal = avg

    # Compute the average signal correctly, get the benchmark
    avg_rets = (avg_signal/(max_tau-min_tau+1)).shift(periods=1)*rets
    both = pd.concat([rets, avg_rets], axis=1)
    both.columns = [ticker, 'Avg']

    # Compute total returns
    ir = ((store + 1).cumprod()-1)*100
    tr = ((both + 1).cumprod()-1)*100

    # Compute descriptives on excess returns
    er = ir.iloc[-1] - tr.iloc[-1,0]
    ds = er.describe()
    pr_er = (ir.iloc[-1] > tr.iloc[-1,0]).mean()*100

    # Print results
    print('kappa = ', set_kappa)
    print('Range of tau = ', min_tau, max_tau)
    print(er)
    print(round(ds,2))
    print(round(pr_er, 2))

# A plot for the post in LinkedIn
etr = pd.concat([ir[24], tr.iloc[:,0]], axis=1).dropna()
etr.plot(title='The Fractional Trader for '+ticker+' and for threshold d=0.3 \n for monthly returns and a 24 month rolling window', xlabel='Date', ylabel='Total return in %', grid='both')
plt.show()

