#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-efficient-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Import a dependent and an explanatory assets
#
# For the post I used the following pairs: (QQQ, AAPL), (QQQ, MSFT), (SPY, EEM), (OIH, DBA), (OIH, GLD)
yname = 'GLD'
xname = 'OIH'
tickers = [xname, yname]

# Download the data, you can change the dates and the frequency
data = yf.download(tickers, start='2022-01-01', end='2024-01-31', interval='1d')['Adj Close'].dropna()
rets = data.pct_change().dropna()
nobs = rets.shape[0]

# Set rolling window and maximum cross-correlation lag
roll = 7
max_lag = 3
set_lags = np.arange(1, max_lag+1, 1)
# Prepare storage
store = pd.DataFrame(data=None, index=rets.index, columns=[yname, 'Corr(min). sign', 'd(min)-1', 'd(min)', 'd(min)+1', 'Corr(max). sign', 'd(max)-1', 'd(max)', 'd(max)+1'])

# Loop over the rolling windows, do the computations, trade and save results
for i in np.arange(0, nobs-roll, 1):
    ri = rets.iloc[i:(roll+i)]
    cor = pd.Series(data=None, index=[set_lags], name='IFE', dtype='float64')
    for j in set_lags:
       crj = pd.concat([ri[xname], ri[yname].shift(periods=j)], axis=1).dropna().corr().iloc[1, 0]
       cor.loc[j] = crj
    ife = 1 - np.sqrt(1 - cor**2)
    min_ife = ife.idxmin()[0]
    max_ife = ife.idxmax()[0]
    bench = rets[yname].iloc[roll+i]
    str1_max = bench*np.sign(cor.loc[max_ife])
    str2_max = bench*np.sign(rets[xname].iloc[roll+i-max_ife-1])
    str3_max = bench*np.sign(rets[xname].iloc[roll+i-max_ife])
    str1_min = bench*np.sign(cor.loc[min_ife])
    str2_min = bench*np.sign(rets[xname].iloc[roll+i-min_ife-1])
    str3_min = bench*np.sign(rets[xname].iloc[roll+i-min_ife])
    if min_ife != 1:
        str4_min = bench*np.sign(rets[xname].iloc[roll+i-min_ife+1])
    else:
        str4_min = bench
    if max_ife != 1:
        str4_max = bench*np.sign(rets[xname].iloc[roll+i-max_ife+1])
    else:
        str4_max = bench
    store.iloc[roll+i] = np.hstack([bench, str1_min, str2_min, str3_min, str4_min, str1_max, str2_max, str3_max, str4_max])

# Compute the combination strategy
store = store.dropna()
cr0 = (store+1).cumprod()
cr1 = cr0.shift(periods=1).fillna(value=0).apply(lambda x: np.argmax(x), axis=1)
cr2 = pd.Series(data=None, index=cr0.index, name='Combo', dtype='float64')
for i in range(cr2.shape[0]):
    cr2.iloc[i] = (store+1).iloc[i, cr1.iloc[i]]
cr2 = cr2.cumprod()
cr0['Wealth Rotation'] = cr2

# Do a nice plot, for the top 2 performers
str_idx = cr0.columns[(cr0.iloc[-1].rank() >= 10)].to_list()
str_idx = np.unique(np.hstack([str_idx, yname]))
tot_ret = (cr0[str_idx]-1)*100
tot_ret.plot(title='The efficient speculator strategy for '+yname+', weekly data', ylabel='total return in percent', xlabel='Date', grid='both')
plt.show()

# and print to discuss
print(tot_ret.iloc[-1])