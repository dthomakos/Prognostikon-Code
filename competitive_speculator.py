#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/12/07/the-competitive-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
import pandas_datareader as pdr

# Define the weighted mean
def weighted_mean(x, alpha=1.0):
    w = np.arange(1, x.shape[0]+1, 1) ** alpha
    m = (x * w).sum()/w.sum()
    return m

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
    er = er.clip(lower=0)
    l2 = (er ** 2).mean(axis=0)
    st = mu/np.sqrt(l2)
    cr = (rets+1).cumprod(axis=0) - 1
    md = max_dd(cr)
    stats = pd.DataFrame([mu, sd, sr, st, cr.iloc[-1], md])
    stats.index = ['Mean', 'Std. Dev.', 'Sharpe', 'Sortino', 'TR', 'MaxDD']
    return stats.transpose(), cr

# Select a ticker to examine - note that the frequency is set to monthly and the interest rate is a monthly one
ticker = 'DBC'
asset = yf.download(ticker, period='max', interval='1mo')['Adj Close'].dropna()
set_rate = 'TB4WK'
interest_rate = pdr.fred.FredReader(set_rate, start=asset.index[0], end=asset.index[-1]).read()/100 + 1
interest_rate = ((interest_rate ** (1/12)) - 1) # need to de-annualize the interest rate
d_interest_rate = interest_rate.diff().dropna()
d_asset = asset.pct_change().dropna()
all = pd.concat([asset, d_asset, interest_rate, d_interest_rate], axis=1).dropna()
all.columns = [ticker, 'D-'+ticker, 'R', 'DR']
nobs = all.shape[0]

# Set the rolling window from 3 to 6 months, in returns
nroll = np.arange(4, 8, 1)

# Set the scale factor from 0 to 2, in intervals of 0.2
nkappa = np.arange(0, 2.2, 0.2)

# Initialze storage to search
store_all = pd.DataFrame(data=None, index=nkappa, columns=nroll)

# Do the search loop
for roll in nroll:
    for kappa in nkappa:
        #
        # print(roll, kappa)
        store = pd.DataFrame(data=None, index=all.index, columns=[ticker, 'Speculation'])
        store_beta = pd.DataFrame(data=None, index=all.index, columns=['Discount'])
        count_nd = 0
        # Do the evaluation loop
        for i in np.arange(roll, nobs, 1):
            all_i = all.iloc[(i-roll):i]
            drift = weighted_mean(all_i['D-'+ticker], 0)
            kstd =  kappa*all_i['D-'+ticker].std()
            last_yi = all_i[ticker].iloc[-1]
            last_dri = all_i['DR'].iloc[-1]
            last_ri = all_i['DR'].iloc[-1]
            if last_ri < 0:
                last_ri = 0
                count_nd += 1
            beta_i = 1/(1+last_ri)
            store_beta.iloc[i] = beta_i
            spec = beta_i*(last_yi + drift  + kstd - last_dri) - last_yi
            benc = all['D-'+ticker].iloc[i]
            strg = np.sign(spec)*benc
            store.iloc[i,:] = np.hstack([benc, strg])
        #
        # print(count_nd/(nobs-roll))
        # Compute the excess returns and store
        crets = ((store+1).dropna().prod()-1)*100
        drets = crets['Speculation'] - crets[ticker]
        store_all.loc[kappa, roll] = drets

# Find the best performing parameter combination and redo the analysis
max_ret = store_all.max().max()
max_dim = np.where(store_all == max_ret)
roll_max = nroll[max_dim[1]][0]
kappa_max = nkappa[max_dim[0]][0]

store = pd.DataFrame(data=None, index=all.index, columns=[ticker, 'Speculation'])
store_beta = pd.DataFrame(data=None, index=all.index, columns=['Discount'])
count_nd = 0
#
for i in np.arange(roll_max, nobs, 1):
    all_i = all.iloc[(i-roll_max):i]
    drift = weighted_mean(all_i['D-'+ticker], 0)
    kstd =  kappa_max*all_i['D-'+ticker].std()
    last_yi = all_i[ticker].iloc[-1]
    last_dri = all_i['DR'].iloc[-1]
    last_ri = all_i['DR'].iloc[-1]
    if last_ri < 0:
        last_ri = 0
        count_nd += 1
    beta_i = 1/(1+last_ri)
    store_beta.iloc[i] = beta_i
    spec = beta_i*(last_yi + drift  + kstd - last_dri) - last_yi
    benc = all['D-'+ticker].iloc[i]
    strg = np.sign(spec)*benc
    # strg = (spec >= 0)*benc
    store.iloc[i,:] = np.hstack([benc, strg])
#
# print(count_nd/(nobs-roll))
crets = ((store+1).dropna().cumprod()-1)*100

# Done, plot and print
crets.plot(title='The competitive speculator strategy for '+ticker+', monthly rebalancing', xlabel='Date', ylabel='return in percent', grid='both')
plt.show()
#
stats = performance_measures(store, 12, 0)
print(roll_max, kappa_max)
print(stats[0])