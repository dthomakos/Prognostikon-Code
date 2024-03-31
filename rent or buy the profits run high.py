#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/rent-or-buy-the-profits-run-high/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import pandas_datareader as pdr
import statsmodels.api as sm
import yfinance as yf

# A function to compute the maximum drawdown, input is a dataframe of cumulative returns
def max_dd(crets):
    maxcret = (crets+1).cummax(axis=0)
    drawdowns = ((crets + 1) / maxcret) - 1
    return drawdowns.min(axis=0)

# Another function to collect performance measures, input is a dataframe of returns
def performance_measures(rets, f_factor, target_r=0):
    mu = rets.mean() * f_factor
    sd = rets.std() * np.sqrt(f_factor)
    sr = mu / sd
    er = target_r - rets
    cr = (rets+1).cumprod(axis=0) - 1
    md = max_dd(cr)
    stats = pd.DataFrame([mu, sd, sr, cr.iloc[-1], md])
    stats.index = ['Mean', 'Std. Dev.', 'Sharpe', 'TR', 'MaxDD']
    return stats.transpose(), cr

# Read the economic data
data1 = pdr.fred.FredReader(['CUSR0000SEHA', 'CSUSHPISA', 'CPIAUCSL'], start='2000-01-01', end='2024-01-31').read()

# Read the financial data
ticker = 'XLK'
data2 = yf.download(ticker, start='2000-01-01', end='2024-01-31', interval='1mo').dropna()['Adj Close']

# Merge
data =  pd.concat([data1, data2], axis=1).dropna()
data.columns = ['RentPI', 'HousePI', 'CPI', ticker]

# Compute the explanatory variable
data['RentPI/HousePI'] = data['RentPI']/data['HousePI']

# Compute growth rates
use_data = data[[ticker, 'RentPI/HousePI']].pct_change().dropna()*100
nobs = use_data.shape[0]

# Select ols or robust estimation
do_robust = False

# Select a rolling window and the delay - I am using a direct search for the rolling window, you
# can change this below
set_roll = np.arange(6, 16, 1)
delay = 2

for roll in set_roll:

    # Initialize storage
    store = pd.DataFrame(data=None, index=use_data.index, columns=[ticker, 'Rent & Buy #1', 'Rent & Buy #2'])

    # Roll over
    for i in np.arange(0, nobs-roll-delay, 1):
        # Split the data
        di = use_data.iloc[i:(i+roll+delay):]

        # Compute the lags and align
        w = di['RentPI/HousePI'].shift(periods=delay).iloc[delay:]
        Iw = (w <= 0).astype(float)
        if all(Iw == 1):
            x = w
        else:
            x = pd.concat([w, Iw], axis=1)
        x = sm.add_constant(x)
        y = di[ticker].iloc[delay:]

        # Estimate and forecast with first model
        if do_robust:
            out = sm.RLM(endog=y, exog=sm.add_constant(w), hasconst=True, M=sm.robust.norms.TukeyBiweight()).fit()
        else:
            out = sm.OLS(endog=y, exog=sm.add_constant(w), hasconst=True).fit()
        bhat = out.params
        wf = di['RentPI/HousePI'].iloc[-delay]
        xhat = np.hstack([1, wf])
        fhat1 = np.sum(bhat*xhat)

        # Estimate and forecast with second model
        if do_robust:
            out = sm.RLM(endog=y, exog=x, hasconst=True, M=sm.robust.norms.TukeyBiweight()).fit()
        else:
            out = sm.OLS(endog=y, exog=x, hasconst=True).fit()
        bhat = out.params
        wf = di['RentPI/HousePI'].iloc[-delay]
        Iwf = (wf <= 0).astype(float)
        if all(Iw == 1):
            xhat = np.hstack([1, wf])
        else:
            xhat = np.hstack([1, wf, Iwf])
        fhat2 = np.sum(bhat*xhat)

        # Trade, these trades are long-only, switch to signs for long-short
        bench = use_data[ticker].iloc[i+roll+delay]/100
        s1 = bench*(fhat1 > 0) # np.sign(fhat1)
        s2 = bench*(fhat2 > 0) # np.sign(fhat2)
        store.iloc[i+roll+delay] = np.hstack([bench, s1, s2])

    store = store.dropna()
    stats, cr = performance_measures(store, 12, 0)
    print(roll)
    print(stats)
    if (ticker == 'SPY' and roll == 13) or (ticker == 'XLF' and roll == 11) or (ticker == 'XLE' and roll == 15) or (ticker == 'XLK' and roll == 11):
        cr.plot(title='The Rent & Buy Strategy vs the passive benchmark for '+ticker+', monthly data', xlabel='Date', ylabel='return in percent', grid='both')
        plt.show()
