#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2024/02/12/the-random-standardized-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import acf

# Define the standardizing and acf-minimizing function
def minimize_rho(y, B=100, upper_bound=0.1, rho=1, set_seed=-1):
    nobs = y.shape[0]
    store_acf = pd.DataFrame(data=None, index=range(B), columns=[y.name, 'Z', 'mZ'])
    if set_seed > 0:
        np.random.seed = set_seed
    for i in range(B):
        e = pd.Series(np.random.uniform(size=nobs, low=0, high=upper_bound), index=y.index, name=y.name)
        z = y.div(e, axis='index')
        mz = z.mean()
        a1 = acf(y)[1:(rho+1)].sum()
        a2 = acf(z)[1:(rho+1)].sum()
        store_acf.iloc[i] = np.hstack([a1, a2, mz])
    idxmin = store_acf['Z'].abs().to_numpy().argmin()
    bm = (upper_bound/2.0)
    z_min = store_acf['mZ'].iloc[idxmin]
    return store_acf, bm*z_min

# Get some data
tickers = ['BTC-USD', 'SPY', 'QQQ', 'IWF', 'DBA', 'DBC', 'OIH', 'GLD', 'EEM', 'TLT', 'TNA']

# loop over the tickers and values of the upper bound and rolling window
for ticker in tickers:
    data = yf.download(ticker, start='2022-01-01', end='2024-01-31', interval='1wk')['Adj Close'].dropna()
    y = data.pct_change().dropna()
    nobs = y.shape[0]
    y.name = ticker
    print('Now doing ticker=',ticker)

    # Set the upper bound here
    bounds = np.array([0.1, 0.25, 0.5, 0.75, 1, 2])
    # and the rolling window here
    rolls = np.array([4, 5, 6, 8, 12, 14])
    # Choose a seed for the random number generator
    seed = 123
    # and the number of replications
    repls = 200

    # Loop over the bounds and rolling windows
    for bound in bounds:
        for roll in rolls:
            #
            store_frc = pd.DataFrame(data=None, index=y.index, columns=['Bench', 'Idea'])
            store_err = pd.DataFrame(data=None, index=y.index, columns=['Bench', 'Idea'])
            store_ret = pd.DataFrame(data=None, index=y.index, columns=['B&H', 'Bench', 'Idea'])
            # Do the computations and store
            for i in np.arange(0, nobs-roll-1, 1):
                yi = y.iloc[i:(i+roll)]
                ya = y.iloc[i+roll]
                f0 = yi.mean()
                store, f1 = minimize_rho(yi, B=repls, upper_bound=bound, set_seed=seed)
                ff = np.hstack([f0, f1])
                store_frc.iloc[i+roll] = ff
                store_err.iloc[i+roll] = np.sign(ya) - np.sign(ff)
                store_ret.iloc[i+roll] = np.hstack([ya, ya*np.sign(ff)])
            # Crop...
            store_frc = store_frc.dropna()
            store_err = store_err.dropna()
            store_ret = store_ret.dropna()

            # Compute the total return and print only if better than the benchmark
            tot_ret = (store_ret+1).prod()-1
            if (tot_ret['Idea'] > tot_ret['B&H']):
                print('Bound=',bound)
                print('Roll=',roll)
                print((store_ret+1).prod()-1)