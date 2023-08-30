#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/08/31/the-speculative-rotation/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import yfinance as yf

# Define the weighted mean
def weighted_mean(x, alpha=1.0):
    w = np.arange(1, x.shape[0]+1, 1) ** alpha
    m = (x * w).sum()/w.sum()
    return m

# Download some data
ticker1 = 'BTC-USD'
ticker2 = 'SPY'
tickers = [ticker1, ticker2]
data = yf.download(tickers, period='max', interval='1d')['Adj Close'].dropna()
# Get percent returns, select period of estimation/evaluation
r = data.pct_change().dropna().loc['2022-08-01':]
# Define the difference-in-returns
y = r[ticker1]-r[ticker2]

# Selec the rolling window
roll = 63
# Set the exponent
set_alpha = 7
# Short? set to -1 for shorting the second asset
set_short = 1
# Compute the rolling mean and the weighted rolling means
mu0 = y.rolling(window=roll).mean()
mu1 = y.rolling(window=roll).apply(weighted_mean, args=(1.0,))
mu2 = y.rolling(window=roll).apply(weighted_mean, args=(set_alpha,))

# Compute the signs and trade
s0 = mu0.apply(np.sign)
s1 = mu1.apply(np.sign)
s2 = mu2.apply(np.sign)
# Get the benchmark right
bench = r.iloc[roll:]
# And do the rotation below
if set_short == 1:
    r0 = (r[ticker1]*(s0.shift(periods=1) >= 0) + r[ticker2]*(s0.shift(periods=1) < 0)).dropna()
    r1 = (r[ticker1]*(s1.shift(periods=1) >= 0) + r[ticker2]*(s1.shift(periods=1) < 0)).dropna()
    r2 = (r[ticker1]*(s2.shift(periods=1) >= 0) + r[ticker2]*(s2.shift(periods=1) < 0)).dropna()
elif set_short == -1:
    r0 = (y*(s0.shift(periods=1) >= 0) - y*(s0.shift(periods=1) < 0)).dropna()
    r1 = (y*(s1.shift(periods=1) >= 0) - y*(s1.shift(periods=1) < 0)).dropna()
    r2 = (y*(s2.shift(periods=1) >= 0) - y*(s2.shift(periods=1) < 0)).dropna()

# Put together, cumulate and plot
all = pd.concat([bench, r0, r1, r2], axis=1).dropna()
all.columns = [ticker1, ticker2, 'Spec. Mean', 'Spec. WMean', 'Spec. Exponent']
#
tr_all = ((all+1).cumprod()-1)*100
#
set_title = 'Total trading return of the rotating strategy between '+ticker1+' and '+ticker2
set_assets = [ticker1, ticker2, 'Spec. Exponent']
tr_all[set_assets].plot(title=set_title, color=['orange', 'red', 'green'], ylabel='return in percent', xlabel='Date', fontsize=10)
plt.grid(visible=True, which='both')
plt.show()
# and print the total trading return
print(roll)
print(tr_all.iloc[-1])