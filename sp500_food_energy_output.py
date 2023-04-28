#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/04/29/the-market-as-predictor-of-relative-economic-activity/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yahoo
import datetime as dt
from dateutil.relativedelta import relativedelta

# Tickers, start date, end date and data extraction
tickers = ['INDPRO', 'MCUMFN', 'CPIUFDSL', 'CPIENGSL']
start_date = '2016-03-01'
end_date = '2023-03-31'
# Do the download
econdata = pdr.fred.FredReader(tickers, start=start_date, end=end_date).read()
# Give nice column names
econdata.columns = ['Output', 'Capacity', 'Food', 'Energy']

# Get the two ratios, output/capacity and food/energy
econdata = econdata.assign(Index1 = econdata['Food']/econdata['Energy'],
Index2 = econdata['Output']/econdata['Capacity'])
# Make nice column names again
econdata.columns = ['Output', 'Capacity', 'Food', 'Energy', 'Food/Energy', 'Output/Capacity']

# Specify just the two ratios in a new dataframe, take annual log-growth
indices = econdata[['Output/Capacity', 'Food/Energy']].apply(np.log).diff(periods=12).dropna()*100

# Next get the SP500, compute log-growth too!
sp500 = yahoo.download('^GSPC', period='max', interval='1mo')['Adj Close'].dropna()
sp500 = sp500.apply(np.log).diff(periods=12).dropna()*100
sp500.name = 'S&P500'

# Merge the data, drop the NA
data = pd.concat([indices, sp500], axis=1).dropna()

# Put together the two series for comparison; first the relative real output, the CARRO
y = data['Output/Capacity']-data['Food/Energy']
# then the S&P500 with the desired lead time - in a loop, find the lead time with max cross-correlation
lead_time = np.arange(0, 13, 1)
store_corr = pd.DataFrame(data=None, index=lead_time, columns=['CCorr'])
for i in lead_time:
    x = data['S&P500'].shift(periods=i)
    new_data = pd.concat([y, x], axis=1).dropna()
    new_data.columns = ['Relative Real Output', 'S&P500']
    store_corr.iloc[i] = new_data.corr().iloc[1,0]

# Nice plot for the CARRO and S&P500
plot_data = pd.concat([y, data['S&P500']], axis=1)
plot_data.columns = ['CARRO', 'S&P500']
plot_data.plot(title='Capacity-adjusted relative real output - CARRO and the S&P500', xlabel='Year', ylabel='annual log-growth, percent', grid='both', color=['blue', 'green'],
figsize=[13, 8])
plt.show()

# Plot the lead time vs. the cross-correlation
(store_corr*100).plot(title='Cross-correlation vs. lead time - S&P500 leading CARRO',
xlabel='Lead Time, months', ylabel='Cross-correlation, percent', grid='both', figsize=[13, 8])
plt.show()
print(store_corr)

# and then the series with the optimal lead time
istar = store_corr.sort_values(by='CCorr', ascending=False).index[0]
# Fix the dates for the effective sample
eff_start = dt.datetime.strptime(start_date, '%Y-%m-%d').date()+relativedelta(months=12+istar)
sample = str(eff_start)+' to '+end_date
# and do the rest of the computations
x = data['S&P500'].shift(periods=istar)
new_data = pd.concat([y, x], axis=1).dropna()
new_data.columns = ['CARRO(+'+str(istar)+')', 'S&P500']
new_data.plot(title='S&P500 leading CARRO for a lead time of '+str(istar)+' months for '+sample,
xlabel='Year', ylabel='annual log-growth, percent', grid='both', color=['blue', 'green'], figsize=[13, 8])
plt.show()
# and a scatterplot to show the relationship from a different perspective
new_data.plot(kind='scatter', x='S&P500', y='CARRO(+'+str(istar)+')',
title='S&P500 leading CARRO for a lead time of '+str(istar)+' months for '+sample,
xlabel='S&P500'', percent', ylabel='CARRO(+'+str(istar)+'), percent', grid='both',
c='green', s=50, figsize=[13, 8])
plt.show()