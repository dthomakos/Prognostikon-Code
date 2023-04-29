#
# Based on an idea and the assets from Kurtis Hemmerling, see this LinkedIn post
#
# https://www.linkedin.com/posts/hemmerlingkurtis_etfs-investingstrategy-longshort-activity-7057062449618817024-RdoE?utm_source=share&utm_medium=member_desktop
#
# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yahoo

# Download data, using monthly returns for a quick evaluation...
data = yahoo.download(['USMV', 'SPHB', 'SPY'], period='max', interval='1mo')['Adj Close'].dropna()
# Compute percent returns and cumulative returns
rets = data.pct_change().dropna()
cret = (rets+1).cumprod()

# Compute the difference of cumulative returns
dret = cret['USMV']-cret['SPHB']
# The signal is based on the sign of the previous month's rolling mean
signal = dret.rolling(window=12).mean().shift(periods=1).apply(np.sign)

# Compute the strategy's returns: long USMV if the signal is negative, long SPHB otherwise
sret = rets['USMV']*(signal <= 0) + rets['SPHB']*(signal > 0)

# Crop returns to approximately match the published post in LinkedIn
new_sret = sret.loc['2020-05-01':]
# Compute cumulative returns for the strategy and the index
tret = (new_sret+1).cumprod()-1
tspy = (rets['SPY']+1).loc[tret.index].cumprod()-1
# Put together, nice names and plot!
tall = pd.concat([tspy, tret], axis=1)*100
tall.columns = ['SPY', 'USMV-SPHB']
#
tall.plot(title='A simple version of USMV-SPHB rotation, monthly data - based on an idea from Kurtis Hemmerling',
xlabel='Time', ylabel='percent', grid='both')
plt.show()
#
print(tall)
