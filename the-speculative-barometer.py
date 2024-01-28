#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2024/01/28/the-speculative-barometer/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import scipy as sp
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
    er = er.clip(lower=0)
    l2 = (er ** 2).mean(axis=0)
    st = mu/np.sqrt(l2)
    cr = (rets+1).cumprod(axis=0) - 1
    md = max_dd(cr)
    stats = pd.DataFrame([mu, sd, sr, st, cr.iloc[-1], md])
    stats.index = ['Mean', 'Std. Dev.', 'Sharpe', 'Sortino', 'TR', 'MaxDD']
    return stats.transpose(), cr

# Import the data from FRED
tickers1 = ['PCEPI', 'CPIAUCSL', 'TOTALSA', 'HOUST', 'BUSLOANS', 'LOANINV', 'TOTALSL']
tickers2 = ['T10Y2YM', 'UNRATE']
data1 = pdr.fred.FredReader(tickers1, start='1970-01-01', end='2023-12-31').read().dropna()
data2 = pdr.fred.FredReader(tickers2, start='1970-01-01', end='2023-12-31').read().dropna()
# Compute the two indices
x1 = (data1['PCEPI']/data1['CPIAUCSL']) # ratio of PCE to CPI
x2 = (data1['TOTALSL']/data1['LOANINV']) # ratio of total consumer credit to total credit
x3 = (data1['BUSLOANS']/data1['LOANINV']) # ratio of business loads to total credit
# First index
d1 = pd.concat([x1, x2], axis=1).mean(axis=1)*100
# Second, index is simpler, the unemployment rate minus the 10-2 year spread
d2 = data2['UNRATE']-data2['T10Y2YM']

# Put together
all_data = pd.concat([d1, d2], axis=1).dropna()
# Downsample with their mean to quarterly
qrt_data = all_data.resample('Q').mean()
qrt_data.columns = ['I', 'II']

# Download real GDP
gdp = pdr.fred.FredReader('GDPC1', start='1970-01-01', end='2023-12-31').read().dropna()
gdp = gdp.loc['1975-04-01':].apply(np.log).diff(periods=4).dropna()*100
gdp.index = qrt_data.index

# Merge with the two indices
qrt_data1 = pd.concat([qrt_data['I'], gdp], axis=1)
qrt_data1.columns = ['I', 'GDP']
#
qrt_data2 = pd.concat([qrt_data['II'], gdp], axis=1)
qrt_data2.columns = ['II', 'GDP']
#
qrt_data3 = pd.concat([qrt_data.mean(axis=1), gdp], axis=1)
qrt_data3.columns = ['III', 'GDP']

# Remove the outliers? leave at false, this was for another idea!!!
set_outliers = False
if set_outliers:
    qrt_data1 = qrt_data1.loc[qrt_data1['GDP'] > -8.0]
    qrt_data1 = qrt_data1.loc[qrt_data1['GDP'] <  6.0]

    qrt_data2 = qrt_data2.loc[qrt_data2['GDP'] > -8.0]
    qrt_data2 = qrt_data2.loc[qrt_data2['GDP'] <  6.0]

    qrt_data3 = qrt_data3.loc[qrt_data3['GDP'] > -8.0]
    qrt_data3 = qrt_data3.loc[qrt_data3['GDP'] <  6.0]

# Compute the cross cross-correlation
seq_lags = np.arange(0, 13, 1)
corr = pd.DataFrame(data=None, index=seq_lags, columns=['GDP & lagged I', 'GDP & lagged II', 'GDP & lagged III'])
for i in seq_lags:
    c1 = qrt_data1['GDP'].corr(qrt_data1['I'].shift(periods=i))
    c2 = qrt_data2['GDP'].corr(qrt_data2['II'].shift(periods=i))
    c3 = qrt_data3['GDP'].corr(qrt_data3['III'].shift(periods=i))
    corr.loc[i] = np.hstack([c1, c2, c3])

# Print the cross-correlation and plot it
print(corr)
corr.plot()
plt.show()

# Plot the indices and real GDP growth
# qrt_data1.plot(secondary_y='GDP')
# qrt_data2.plot()
ax1 = qrt_data3['III'].plot(title='US real GDP growth (blue line, right axis) \n & the speculative barometer (black line, left axis)', xlabel='Date', ylabel='Percent', color='black')
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
ax2 = ax1.twinx()
qrt_data3['GDP'].plot(ylabel='Percent', color='blue')
plt.show()

# Compute a rolling cross-correlation with real GDP
roll_corr = pd.DataFrame(data=None, index=qrt_data3.index, columns=['Corr[GDP, III(-4)]'])
roll = 32
nobs = qrt_data3.shape[0]
for i in np.arange(0, nobs-roll, 1):
    xi = qrt_data3.iloc[i:(i+roll+1)]
    ci = xi['GDP'].corr(xi['III'].shift(periods=4))
    roll_corr.iloc[i+roll] = ci*100

# and plot it
z = pd.concat([roll_corr, qrt_data3['GDP']], axis=1).dropna()
z.plot(secondary_y='GDP')
plt.show()

# and now do the speculative evaluation from 2001 and 2008
ticker = '^GSPC'
sp500 = yf.download(ticker, start='1985-01-01', end='2023-11-30', interval='1mo')['Adj Close'].dropna()
# start from 2001, 2008 (start from August of previous year!)
sp500 = sp500.pct_change().dropna().loc['2007-08-01':]
sp500.name = ticker

# Compute the signals
signal1 = all_data.iloc[:,0].loc[sp500.index].pct_change().shift(periods=4)
signal2 = -all_data.iloc[:,1].loc[sp500.index].pct_change().shift(periods=2)
signal3 = -all_data.mean(axis=1).loc[sp500.index].pct_change().shift(periods=2)
signal4 = np.sign(signal1)+np.sign(signal2)
str1 = np.sign(signal1)*sp500
str2 = np.sign(signal2)*sp500
str3 = np.sign(signal3)*sp500
str4 = np.sign(signal4)*sp500
all = pd.concat([sp500, str1, str2, str3, str4], axis=1).dropna()

# Print performance statistics and plot cumulative returns
stats, cr = performance_measures(all, 12)
stats = stats.T
stats.columns = ['S&P500', 'I', 'II', 'III', 'I and II']
print(stats)
cr.columns = stats.columns
cr[['S&P500', 'II', 'III', 'I and II']].plot(title='The speculative barometer strategies for the S&P500', xlabel='Date', ylabel='return in decimal', grid='both')
plt.show()

# and the performance by year
by_year = all.groupby(by=[all.index.year]).apply(lambda x: (x+1).prod()-1)
by_year.columns = cr.columns

# and a nice bar plot of the annual performance
by_year.plot(kind='bar', grid='both', xlabel='Year', ylabel='Total return in decimal', title='Annual Total Return for the Speculative Barometer Strategies')
plt.show()

