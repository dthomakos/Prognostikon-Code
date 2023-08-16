#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/07/09/the-probable-speculative-constant/
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
ticker = 'VXX'
data = yf.download(ticker, period='max', interval='1d')['Adj Close'].dropna()
# Get percent returns, select period of estimation/evaluation
y = data.pct_change().dropna().loc['2022-01-01':]

# Selec the rolling window - note that this will be used twice
roll = 63
# Set the exponent
set_alpha = 7
# Compute the rolling mean and the weighted rolling means
mu0 = y.rolling(window=roll).mean()
mu1 = y.rolling(window=roll).apply(weighted_mean, args=(1.0,))
mu2 = y.rolling(window=roll).apply(weighted_mean, args=(set_alpha,))

# Compute the signs and trade
s0 = mu0.apply(np.sign)
s1 = mu1.apply(np.sign)
s2 = mu2.apply(np.sign)
#
bench = y.iloc[roll:]
r0 = (y.iloc[1:]*s0.shift(periods=1)).dropna()
r1 = (y.iloc[1:]*s1.shift(periods=1)).dropna()
r2 = (y.iloc[1:]*s2.shift(periods=1)).dropna()

# Put together, cumulate and plot
all = pd.concat([bench, r0, r1, r2], axis=1)
all.columns = [ticker, 'Spec. Mean', 'Spec. WMean', 'Spec. Exponent']
#
tr_all = ((all+1).cumprod()-1)*100
#
tr_all.plot(title='Total trading returns of the speculative constant & exponent strategies in '+ticker, color=['red', 'blue', 'green', 'orange'], ylabel='return in percent')
plt.grid(visible=True, which='both')
plt.show()
# and print the total trading return
print(roll)
print(tr_all.iloc[-1])