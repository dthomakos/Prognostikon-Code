#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/currency-and-money-market-funds-as-unemployment-proxies/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from scipy import signal, stats
from statsmodels.tsa.stattools import acf
from statsmodels.sandbox.stats import runs
from statsmodels.stats.descriptivestats import sign_test
import yfinance as yf

# Get the weekly data
tickers = ['WCURRNS', 'WRMFNS']
raw_data = pdr.fred.FredReader(tickers, start='1980-01-01', end='2024-10-01').read()
raw_data['WMR'] = np.log(raw_data['WCURRNS']/raw_data['WRMFNS'])*100

# Transform the main series into monthly
wmr_monthly = raw_data['WMR'].resample('MS').mean()

# Get the unemployment data
tickers = ['UNRATE']
raw_data = pdr.fred.FredReader(tickers, start='1980-01-01', end='2024-10-01').read()
# Add the main series
raw_data['AMR'] = wmr_monthly
# plot the original data (commented out)
# ax1 = raw_data['UNRATE'].plot(title='Unemployment and the WMR Indicator', xlabel='Date', ylabel='Unemployment Rate', color='black')
# ax1.xaxis.grid(True, which='major')
# ax1.yaxis.grid(True, which='major')
# ax1.legend(loc='lower left')
# ax2 = ax1.twinx()
# raw_data['WMR'].plot(ax=ax2,color=['blue'], ylabel='Currency in M1/Retail Money Market Funds Growth')
# ax2.legend(loc='upper left')
# plt.show()

# Compute the lagged-cross correlation between the main series and the unemployment rate
#
start_from = '2019-01-01' # Use '1990-01-01', '2000-01-01' and '2019-01-01'
if start_from is not None:
    x = raw_data.loc[start_from:].dropna()
else:
    start_from = str(raw_data.index.year[0])+'-01-01'
    x = raw_data.dropna()
#
store_ccor = pd.DataFrame(data=None, index=np.arange(0, 49, 1), columns=['WMR-UNRATE'])
for i in np.arange(0, 49, 1):
    store_ccor.iloc[i, 0] = x['UNRATE'].corr(x['AMR'].shift(periods=i))*100
# plot the lagged-cross correlations
store_ccor.plot(title='Cross-correlations between lagged AMR and the Unemployment Rate \n from '+start_from, xlabel='Lag', ylabel='Correlation', grid='both')
plt.show()
# extract max cross-correlation
max_ccor = store_ccor.apply(lambda x: np.argmax(np.abs(x)), axis=0)-1

# Next plot the aligned main series with the unemployment rate
#
# Expand the sample
extra = pd.date_range(start='2024-09-01', periods=max_ccor.iloc[0], freq='MS')
extra_sample = pd.date_range(start=x.index[0], end=extra[-1], freq='MS')
z = pd.DataFrame(data=None, index=extra_sample, columns=['UNRATE', 'WMR'])
z['UNRATE'] = x['UNRATE']
z['AMR'] = x['AMR']
z['AMR'] = z['AMR'].shift(periods=max_ccor.iloc[0])
z = z.iloc[max_ccor.iloc[0]:]
#
ax1 = z['UNRATE'].plot(title='Unemployment and the AMR Indicator lagged '+str(max_ccor.iloc[0])+' months \n from '+start_from, xlabel='Date', ylabel='Unemployment Rate', color='black')
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
ax1.legend(loc='lower left')
ax2 = ax1.twinx()
z['AMR'].plot(ax=ax2,color=['blue'], ylabel='Currency in M1/Retail Money Market Funds Growth')
ax2.legend(loc='upper left')
plt.show()

# Now for the trading exercise, a very simple one
ticker = 'TNA'
if ticker == 'SPY':
    d_delay = 2
elif ticker == 'QQQ':
    d_delay = 1
elif ticker == 'TNA':
    d_delay = 2
#
fin_data = yf.download(ticker, period='max', interval='1mo')['Adj Close'].dropna()
rets = fin_data.pct_change().dropna()
amr_rets = pd.concat([x, rets], axis=1).dropna()
amr_rets.columns = ['UNRATE', 'AMR', ticker]
for jj in np.arange(1, 13, 1):
    # the strategy is based just on the sign of the AMR indicator before 2019
    # and the negative of the sign of the difference of the AMR indicator after 2019
    if start_from < '2019-01-01':
        str = amr_rets[ticker]*(amr_rets['AMR'].shift(periods=jj).apply(np.sign))
    else:
        str = -amr_rets[ticker]*(amr_rets['AMR'].diff(periods=d_delay).shift(periods=jj).apply(np.sign))
    both = pd.concat([amr_rets[ticker], str], axis=1).dropna()
    both.columns = [ticker, 'AMR-'+ticker]
    tr = ((both + 1).cumprod()-1)*100
    # plot
    tr.plot(title='Total return of the AMR strategy for '+ticker+' from '+start_from, xlabel='Date', ylabel='return in percent', grid='both')
    plt.show()
    print(tr.iloc[-1])