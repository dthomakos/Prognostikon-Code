#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-speculative-sectors/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos

# Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
import yfinance as yf

# Linear model, least squares or robust, from statsmodels with sequential eliminination based on p-values
def sequential_elimination_lm(set_Y, set_X, set_alpha, robust=False):
    if robust:
        out = sm.RLM(endog=set_Y, exog=set_X).fit()
    else:
        out = sm.OLS(endog=set_Y, exog=set_X, hasconst=True).fit()
    pv_old = out.pvalues
    ip_old = pv_old[pv_old <= set_alpha[0]].index
    if len(ip_old) == 0:
        return out

    # and with a simple loop remove the rest in the proper way with diminishing p-values
    for aa in np.arange(1, len(set_alpha)):
        xa = set_X[ip_old]
        ya = set_Y
        if robust:
            out = sm.RLM(endog=set_Y, exog=xa).fit()
        else:
            out = sm.OLS(endog=ya, exog=xa, hasconst=True).fit()
        pv_new = out.pvalues
        ip_new = pv_new[pv_new <= set_alpha[aa]].index
        if len(ip_new) > 0:
            pv_old = pv_new
            ip_old = ip_new

    # and this is the final model
    xa = set_X[ip_old]
    ya = set_Y
    out = sm.OLS(endog=ya, exog=xa, hasconst=True).fit()

    # Done!
    return out

# A simple regression predictor based on the above function and the data structure of the post
def srp(data, alpha, lag, robust):
    y = data.iloc[:, 0]
    x = sm.add_constant(data.iloc[:, 1:])
    model = sequential_elimination_lm(y.iloc[lag:], x.shift(periods=lag).iloc[lag:], alpha, robust)
    beta = model.params
    xfor = x[model.model.exog_names].iloc[-1]
    fcst = np.sign((beta.mul(xfor)).sum())
    return fcst

# Load the data
tickers = ['SPY', 'XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU', 'XLRE']
data_close = yf.download(tickers, period='max', interval='1d')['Adj Close'].dropna()
# Select a starting date
start_from = '2023-01-01'
if start_from is not None:
    data_close = data_close.loc[start_from:]

# Sector names
sector_names = ['XLC', 'XLY', 'XLP', 'XLE', 'XLF', 'XLV', 'XLI', 'XLB', 'XLK', 'XLU', 'XLRE']
# Index names
index_names = ['SPY']
# Sector closing names
sector_close_names = [x+str('_C') for x in sector_names]
# Index closing names
index_close_names = [x+str('_C') for x in index_names]

# Get the daily returns
sector_close = data_close[sector_names].pct_change().dropna()
index_close = data_close[index_names].pct_change().dropna()

# The trading exercise is straightforward, with the signals coming from a multiple regression (see the blog post for theoretical details)
set_factor_X = sector_close.apply(np.sign)
set_index_Y = index_close.apply(np.sign)
factor_data = pd.concat([set_index_Y, set_factor_X], axis=1)
roll = 14
store = pd.DataFrame(data=None, index=index_close['SPY'].index, columns=['SPY', 'Factors'])
nobs = index_close['SPY'].shape[0]
# Set the parameters for model reduction and estimation
set_alpha = [0.9, 0.8]
set_lag = 1
use_robust = True

# Do the trading evaluation in a loop
for i in np.arange(0, nobs-roll, 1):
    i_data = factor_data.iloc[i:i+roll]
    fcst = srp(i_data, set_alpha, set_lag, use_robust)
    bench = index_close['SPY'].iloc[i+roll]
    store.iloc[i] = np.hstack([bench, bench*fcst])

# Compute the cumulative returns, print and plot
store = store.dropna()
store_cr = ((store+1).cumprod()-1)*100
print(store_cr.iloc[-1])
#
store_cr.plot(grid='both', title='The Speculative Sectors Strategy for SPY, daily data', xlabel='Date', ylabel='total return in percent')
plt.show()
