#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/12/10/the-speculative-transform/
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

# Select ticker and frequency of data
ticker = 'TNA'
freq = '1d'
# Don't forget to adjust the frequency factor
ff = 260
# Set starting date
start_date = '2023-01-01'

# Get the data
data = yf.download(ticker, period='max', interval='1d').dropna().loc[start_date:]
price = data['Adj Close']
volume = data['Volume']

# Prepare the variables
#
# Smoothing volume
smooth = 7
V = volume.rolling(window=smooth).mean()
U = (volume.apply(np.log) - V.apply(np.log))
# Scale expanding or rolling
rho1 = U/U.abs().expanding().mean()
rho2 = U/U.abs().rolling(window=smooth).mean()

# Scale prices and returns
piso = 0.25*(100 ** 0.75)*(price ** 0.25)
diso = 0.031623*price.diff()/(price ** 0.75)
rets = price.pct_change()

# Put together, give names, prepare other parameters
all = pd.concat([price, piso, volume, V, U, rho1, rho2, rets, diso], axis=1).dropna()
all.columns = [ticker, ticker+'-iso', ticker+'-V', 'V', 'U', 'rho1', 'rho2', 'R', 'R-iso']
nobs = all.shape[0]
roll = 4
is_rolling = True

# Initialize storage
store1 = pd.DataFrame(data=None, index=all.index, columns=['vS1', 'vS2', 'vS3', 'vS4', ticker])
store2 = pd.DataFrame(data=None, index=all.index, columns=['vS1', 'vS2', 'vS3', 'vS4', ticker])

# The evulation loop
for i in np.arange(roll, nobs, 1):
    if is_rolling:
        all_i = all.iloc[(i-roll):i]
    else:
        all_i = all.iloc[:i]

    # First, with rho1
    riso_mean = all_i['R-iso'].mean()
    rho1_mean = all_i['rho1'].mean()
    riso_last = all_i['R-iso'].iloc[-1]
    rho1_last = all_i['rho1'].iloc[-1]
    ra = all['R'].iloc[i]
    if np.sign(riso_mean) == np.sign(rho1_last):
        store1['vS1'].iloc[i] = ra
    else:
        store1['vS1'].iloc[i] = -ra
    if np.sign(riso_last) == np.sign(rho1_mean):
        store1['vS2'].iloc[i] = -ra
    else:
        store1['vS2'].iloc[i] = ra
    r = all_i[['rho1', 'R-iso']].corr().iloc[1,0]
    r1 = (all_i['rho1'].shift(periods=1)).corr(all_i['R-iso'])
    store1['vS3'].iloc[i] = np.sign(r)*ra
    store1['vS4'].iloc[i] = np.sign(r1)*ra
    store1[ticker].iloc[i] = ra

    # Second, with rho2
    riso_mean = all_i['R-iso'].mean()
    rho1_mean = all_i['rho2'].mean()
    riso_last = all_i['R-iso'].iloc[-1]
    rho1_last = all_i['rho2'].iloc[-1]
    ra = all['R'].iloc[i]
    if np.sign(riso_mean) == np.sign(rho1_last):
        store2['vS1'].iloc[i] = ra
    else:
        store2['vS1'].iloc[i] = -ra
    if np.sign(riso_last) == np.sign(rho1_mean):
        store2['vS2'].iloc[i] = -ra
    else:
        store2['vS2'].iloc[i] = ra
    r = all_i[['rho2', 'R-iso']].corr().iloc[1,0]
    r1 = (all_i['rho2'].shift(periods=1)).corr(all_i['R-iso'])
    store2['vS3'].iloc[i] = np.sign(r)*ra
    store2['vS4'].iloc[i] = np.sign(r1)*ra
    store2[ticker].iloc[i] = ra

# Done, print performance and plot
out1 = performance_measures(store1.dropna(), ff, 0)
out2 = performance_measures(store2.dropna(), ff, 0)
#
print(out1[0])
print(out2[0])

(out2[1]*100)[[ticker, 'vS4']].plot(title='The speculative transform strategy for '+ticker+', daily data', xlabel='Date', ylabel='return in percent', grid='both')
plt.show()