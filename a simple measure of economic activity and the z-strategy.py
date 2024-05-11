#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/a-simple-measure-of-real-economic-activity-the-z-strategy/
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

# Get the data
tickers = ['CUMFN', 'MDSP', 'GDPC1']
raw_data = pdr.fred.FredReader(tickers, start='1980-01-01', end='2024-10-01').read()

# Get the two variables, the quarterly change in the contrast between capacity utilization and
# mortage payments as % of disposable income
y0 = (raw_data['CUMFN']-raw_data['MDSP']).diff(periods=1)
# and the quarterly real GDP growth
y1 = raw_data['GDPC1'].pct_change(periods=1)*100

# Put together, give column names
data = pd.concat([y0, y1], axis=1).dropna()
data.columns = ['Capacity minus Mortgage Payments', 'Real GDP Growth']

# A full sample plot of the data
data.plot(secondary_y='Capacity minus Mortage Payments', title='', xlabel='Date', grid='both')
plt.show()

# Select dates for doing the comparisons
dates = ['1981-01-01', '1985-01-01', '1990-01-01', '1995-01-01', '2000-01-01', '2005-01-01', '2010-01-01', '2015-01-01', '2019-01-01']

# Import the financial data
sp500 = yf.download('^GSPC', period='max', interval='1mo')['Adj Close'].dropna()
rets = sp500.pct_change().dropna()
rets = rets.resample('QS-OCT').apply(lambda x: x.iloc[-1])

# Prepare dataframe to save the results
results = pd.DataFrame(data=None, index=dates, columns=['Corr(y, x)', 'Corr(fy, fx)', 'Sbar', 'Corr(ry, rx)', 'Normality', 'Sign', 'ER'])

# Loop over the dates and compute what you need
for d in dates:
    # Section the data
    x = data.loc[d:]
    r = rets.loc[d:'2023-10-01']

    # Save correlation of the data
    results.loc[d, 'Corr(y, x)'] = x.corr().iloc[1,0]

    # Scale the data for computing their spectral density and coherence
    sx = x/x.std(axis=0)
    f0, Pxx_den0 = signal.welch(sx.iloc[:,0].to_numpy(), fs=1, nperseg=36, scaling='spectrum')
    f1, Pxx_den1 = signal.welch(sx.iloc[:,1].to_numpy(), fs=1, nperseg=36, scaling='spectrum')
    ff = pd.DataFrame(data=np.hstack([Pxx_den0.reshape(-1, 1), Pxx_den1.reshape(-1, 1)]), index=f0, columns=data.columns)
    # ff.plot(title='Spectral densities from '+d, xlabel='Frequency', ylabel='Power', grid='both')
    # plt.show()

    # Save the correlation of the spectra
    results.loc[d, 'Corr(fy, fx)'] = ff.corr().iloc[1,0]

    # Compute and plot the coherence
    f, Cxy = signal.coherence(sx.iloc[:,0], sx.iloc[:, 1], nperseg=36)
    # plt.plot(f, Cxy)
    # plt.title('Squared coherence from '+d)
    # plt.grid(which='both')
    # plt.xlabel('Frequency')
    # plt.ylabel('Squared coherence')
    # plt.show()

    # Save the average coherence
    results.loc[d, 'Sbar'] = np.mean(Cxy)

    # Compute and plot the ACF
    a0 = acf(x.iloc[:,0]).reshape(-1, 1)
    a1 = acf(x.iloc[:,1]).reshape(-1, 1)
    aa = np.hstack([a0, a1])
    # plt.plot(aa)
    # plt.title('Autocorrelations from '+d)
    # plt.grid(which='both')
    # plt.xlabel('Lag')
    # plt.ylabel('Autocorrelation')
    # plt.show()

    # Save the correlation of the ACFs
    results.loc[d, 'Corr(ry, rx)'] = np.corrcoef(aa.T)[1,0]

    # Compute the two tests and save their pvalues, for the difference of the two series
    z = x.iloc[:,1] - x.iloc[:,0]
    results.loc[d, 'Normality'] = stats.normaltest(z).pvalue
    results.loc[d, 'Sign'] = sign_test(z, mu0=z.mean())[1]

    # Trade the index
    u = r*(z.shift(periods=2).apply(np.sign))
    both = pd.concat([r, u], axis=1).dropna()
    tr = (both+1).prod()
    results.loc[d, 'ER'] = tr.iloc[1] - tr.iloc[0]
    if d == dates[0]:
        cr = ((both+1).cumprod()-1)*100
        cr.plot(title='Cumulative return of the Z strategy from '+d+', d=2 quarters', xlabel='Date', ylabel='return in percent', grid='both')
        plt.show()

# OK, just print the results now
print(results)


