#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-predictive-nature-of-deposits-sales-and-prices-the-dscp-strategy/
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
import scipy as sp
import statsmodels.api as sm
import yfinance as yf

# Parametrization
weekly_ticker = ['DPSACBW027SBOG']
monthly_tickers = ['PPIACO', 'CPIAUCSL', 'MRTSSM44000USS']
asset_ticker = 'TNA' # SPY QQQ IWF TNA DBC DBA XLF XLE
start_from = '2000-01-01' # 2000-01-01 or earliest time and 2018-01-01
max_delay = 12

# Import monthly economic data
monthly_data = pdr.fred.FredReader(monthly_tickers, start=start_from, end='2024-03-01').read()
monthly_data.columns = ['PPI', 'CPI', 'Sales']
monthly_data = monthly_data.dropna()

# Import weekly deposits and convert to monthly
weekly_data = pdr.fred.FredReader(weekly_ticker, start=start_from, end='2024-03-01').read()
weekly_to_monthly = weekly_data.resample('M').apply(lambda x: x.iloc[-1])
weekly_to_monthly = weekly_to_monthly.loc[start_from:'2024-02-01']
weekly_to_monthly.index = monthly_data.index
monthly_data['Deposits'] = weekly_to_monthly

# Import the financial returns
asset = yf.download(asset_ticker, start=start_from, end='2024-02-01', interval='1mo')['Adj Close'].dropna()
monthly_data[asset_ticker] = asset

# Create the new variables
monthly_data['Sales/Deposits'] = monthly_data['Sales']/monthly_data['Deposits']
monthly_data['CPI/PPI'] = monthly_data['CPI']/monthly_data['PPI']
monthly_data['Deposits/PPI'] = monthly_data['Deposits']/monthly_data['PPI']
monthly_data['Sales/CPI'] = monthly_data['Sales']/monthly_data['CPI']

# Take monthly growth rates
data = monthly_data.pct_change().dropna()

# Set search delay sequence and storage
seq_delay = np.arange(1, max_delay+1, 1)
store_cr = pd.DataFrame(data=None, index=seq_delay, columns=['Bench', 'S1', 'S2', 'S3', 'S1ns', 'S2ns', 'S3ns'])

# Do the loop over all delay values
for delay in seq_delay:

    # Compute the strategies directly
    bench = data[asset_ticker].iloc[delay:]
    s1 = data[asset_ticker]*np.sign((data['Deposits/PPI']-data['Sales/CPI']).shift(periods=delay))
    s2 = data[asset_ticker]*np.sign((data['Deposits']-data['CPI']).shift(periods=delay))
    s3 = data[asset_ticker]*np.sign((data['Deposits']).shift(periods=delay))
    s1ns = data[asset_ticker]*((data['Deposits/PPI']-data['Sales/CPI']).shift(periods=delay) >= 0)
    s2ns = data[asset_ticker]*((data['Deposits']-data['CPI']).shift(periods=delay) >= 0)
    s3ns = data[asset_ticker]*((data['Deposits']).shift(periods=delay) >= 0)
    cr_bench = ((bench+1).cumprod()-1)*100
    cr_s1 = ((s1+1).cumprod()-1)*100
    cr_s2 = ((s2+1).cumprod()-1)*100
    cr_s3 = ((s3+1).cumprod()-1)*100
    cr_s1ns = ((s1ns+1).cumprod()-1)*100
    cr_s2ns = ((s2ns+1).cumprod()-1)*100
    cr_s3ns = ((s3ns+1).cumprod()-1)*100
    #
    store_cr.loc[delay] = np.hstack([cr_bench.iloc[-1], cr_s1.iloc[-1], cr_s2.iloc[-1], cr_s3.iloc[-1], cr_s1ns.iloc[-1], cr_s2ns.iloc[-1], cr_s3ns.iloc[-1]])

# Find the max over all delay periods, print and then adjust lines 82 and 83 manually to produce the plots!
cr_max = store_cr.max(axis=0)
print(cr_max)
print(store_cr.idxmax(axis=0))
#
best_delay = 10
best_str = 'S1ns'

# Plotting
if best_str == 'S1':
    bench = data[asset_ticker].iloc[best_delay:]
    best = data[asset_ticker]*np.sign((data['Deposits/PPI']-data['Sales/CPI']).shift(periods=best_delay))
    cr_bench = ((bench+1).cumprod()-1)*100
    cr_best = ((best+1).cumprod()-1)*100
    merge = pd.concat([cr_bench, cr_best], axis=1).dropna()
    merge.columns = [asset_ticker, 'S1']
elif best_str == 'S2':
    bench = data[asset_ticker].iloc[best_delay:]
    best = data[asset_ticker]*np.sign((data['Deposits']-data['CPI']).shift(periods=best_delay))
    cr_bench = ((bench+1).cumprod()-1)*100
    cr_best = ((best+1).cumprod()-1)*100
    merge = pd.concat([cr_bench, cr_best], axis=1).dropna()
    merge.columns = [asset_ticker, 'S2']
elif best_str == 'S1ns':
    bench = data[asset_ticker].iloc[best_delay:]
    best = data[asset_ticker]*((data['Deposits/PPI']-data['Sales/CPI']).shift(periods=best_delay) >= 0)
    cr_bench = ((bench+1).cumprod()-1)*100
    cr_best = ((best+1).cumprod()-1)*100
    merge = pd.concat([cr_bench, cr_best], axis=1).dropna()
    merge.columns = [asset_ticker, 'S1ns']
elif best_str == 'S2ns':
    bench = data[asset_ticker].iloc[best_delay:]
    best = data[asset_ticker]*((data['Deposits']-data['CPI']).shift(periods=best_delay) >= 0)
    cr_bench = ((bench+1).cumprod()-1)*100
    cr_best = ((best+1).cumprod()-1)*100
    merge = pd.concat([cr_bench, cr_best], axis=1).dropna()
    merge.columns = [asset_ticker, 'S2ns']

# Done!
merge.plot(title='The DSCP strategy for '+asset_ticker, xlabel='Date', ylabel='return in percent', grid='both')
plt.show()
