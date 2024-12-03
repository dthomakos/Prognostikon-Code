#
# Python code replicating results on this post:
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

## Define the trading function
def osborne(x, kappa=-1):
    nu = len(x)
    s2 = x.var()
    br = (x+1).prod()-1
    er = br - 0.5*s2*nu
    if kappa == -2:
        threshold = np.sqrt(s2)
    elif kappa == -1:
        threshold = s2
    elif kappa >= 0:
        threshold = kappa
    #
    if er > threshold:
        return -1.0
    else:
        return +1.0

## Get some data
ticker = 'GLD'
start_from = '2022-01-01'
freq = '1d'
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
rets = data.pct_change().dropna()
rets = rets.loc[start_from:]

## Set the parameters and initialize storage
kappa = [-2, -1, 0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3]

## Do a loop over the values of kappa
for set_kappa in kappa:
    min_tau = 3
    max_tau = 26
    tau = np.arange(min_tau, max_tau+1, 1)
    store = pd.DataFrame(data=None, index=rets.index, columns=tau)
    signals = pd.DataFrame(data=None, index=rets.index, columns=tau)
    avg_signal = pd.DataFrame(data=None, index=rets.index, columns=['Avg'])
    avg = 0

    # Get the signals and the average signal
    for i in tau:
        si = rets.rolling(window=i).apply(osborne, args=(set_kappa,))
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

