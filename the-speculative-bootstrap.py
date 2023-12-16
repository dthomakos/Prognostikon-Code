#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/12/16/the-speculative-bootstrap/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Define the trading function, a simple mean defines the forecast
def trade_the_mean(y, rwind):
    frc = y.rolling(window=rwind).mean().apply(np.sign)
    ret = (frc.shift(periods=1))*y
    ben = y.loc[ret.index]
    xrt = ret
    return xrt

# Define the boostrapped trades
def boot_the_mean(y, rwind, B=1000, n=30):
    nobs = y.shape[0]
    store = pd.DataFrame(data=None, index=range(B), columns=['Bootstraps'])
    xrt0 = trade_the_mean(y, rwind).dropna()
    tst0 = (xrt0+1).prod()
    for i in range(B):
        t0 = np.random.randint(low=0, high=nobs-1, size=1)[0]
        t1 = t0 + n
        if t1 > nobs-1:
            t1 = nobs
        yi = y.iloc[t0:t1]
        xrti = trade_the_mean(yi, rwind)
        tsti = (xrti+1).prod()
        store.iloc[i] = tsti
    return xrt0, tst0, store

# Define the distribution of bootstrapped trades over rolling windows
def boot_the_window(y, nrwind, boot=1000, size=30):
    rseq = np.arange(nrwind[0], nrwind[-1]+1, 1)
    tstar = pd.DataFrame(data=None, index=rseq, columns=['tstar'])
    for i in rseq:
        u = boot_the_mean(y, i, B=boot, n=size)
        tstar.loc[i] = np.mean(u[2] < (u[1] ** (1/size)), axis=0)[0]
    return tstar

# Get some data, daily from 2023-01-01
ticker = 'UNG'
data = yf.download(ticker, period='max', interval='1d')['Adj Close'].dropna().loc['2023-01-01':]
rets = data.pct_change().dropna()
nobs = rets.shape[0]

# Define the training and evaluation split dates
xdates = ['2023-03-31', '2023-04-30', '2023-05-31', '2023-06-30', '2023-07-31', '2023-08-31', '2023-09-30', '2023-10-30']
ydates = ['2023-04-01', '2023-05-01', '2023-06-01', '2023-07-01', '2023-08-01', '2023-09-01', '2023-10-01', '2023-11-01']

# A simple outer loop over these days
for d in range(len(xdates)):
    # Get the right dates
    xd = xdates[d]
    yd = ydates[d]
    print('----------------------------------------')
    print('Now doing asset', ticker)
    print('Evaluation starts', yd)
    print('----------------------------------------')
    # Split into training and testing sample
    x = rets.loc[:xd]
    y = rets.loc[yd:]

    # Define sequence of rolling windows
    set_nroll = [2, 14]

    # Define sequence of bootstraps
    set_boot = [25, 50, 150]

    # Define sequence of trading days to boostrap
    set_n_size = [7, 10, 14]

    # A simple loop evaluates the results
    for b in set_boot:
        for n in set_n_size:
            train_rwind = boot_the_window(x, set_nroll, boot=b, size=n)
            rstar = train_rwind.index[np.where(train_rwind.max() == train_rwind)[0]][0]
            sb = trade_the_mean(y, rstar)
            all = (pd.concat([y, sb], axis=1).dropna()+1).cumprod()
            all.columns = [ticker, 'Speculative Bootstrap']
            print('Now doing combination of b =',b,' and n =',n)
            print(round(all.iloc[-1, 1], 5), round(all.iloc[-1, 0], 5))
