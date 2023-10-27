#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/10/28/the-speculative-correlation/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

# Download some data, for the post I used
ticker = 'GLD'
data = yf.download(ticker, period='max', interval='1wk')['Adj Close'].dropna()
r = data.pct_change().dropna().loc['2020-01-01':] # you can change this of course
dr = r.diff()
# Put together returns and difference in returns
z = pd.concat([r, dr], axis=1).dropna()
z.columns = [ticker, 'Δ-'+ticker]

# Set observations and rolling window
nobs = z.shape[0]
roll = 2

# and initialize storage
store = pd.DataFrame(data=None, index=z.index, columns=[ticker, 'Speculative Correlation', 'Signal'])

# Run a simple loop to get the signals and the strategy returns
for i in np.arange(0, nobs-roll, 1):
    z_i = z.iloc[i:(i+roll),:]
    bnh = z.iloc[i+roll, 0]
    z_f = z_i.corr().iloc[1, 0]
    stg = bnh*np.sign(z_f)
    store.iloc[i+roll, :] = np.hstack([bnh, stg, z_f])

# Compute the cumulative return and plot performance
cret = ((store.iloc[:,[0,1]] + 1).cumprod() - 1)*100
cret.plot(grid='both', title='The speculative correlation strategy for '+ticker+' using weekly returns', xlabel='Date', ylabel='return in percent')
plt.savefig(ticker+'.png')
plt.show()
#
print(cret.iloc[-1])
