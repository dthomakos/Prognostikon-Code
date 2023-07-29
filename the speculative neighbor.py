#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/07/29/the-speculative-neighbor/
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
import nn_tools as nn

# Download some data
ticker = 'VXX'
data = yf.download(ticker, period='max', interval='1wk')['Adj Close'].dropna()
# Get percent returns, select period of estimation/evaluation
y = data.pct_change().dropna()#.loc['2018-01-01':]

# Compute the recursive NN forecast
ini_obs = 3
nobs = y.shape[0]
set_p = 1
set_alpha = 1
set_beta = 1
set_nn_type = 'forward'
#
rr = pd.DataFrame(data=None, index=y.index, columns=['Buy & Hold ', 'The Speculative Neighbor'])

#
for i in np.arange(ini_obs, nobs, 1):
    yi = y.iloc[:i]
    id = nn.get_nn(yi.to_numpy(), p=set_p, alpha=set_alpha, step=1)
    fi = nn.get_nn_forecast(yi.to_numpy(), id, beta=set_beta, nn_type=set_nn_type)[0]
    rr.iloc[i] = np.c_[y.iloc[i], np.sign(fi)*y.iloc[i]]

# Do a nice plot
tr = ((rr+1).cumprod()-1)*100
tr.plot(title='Total trading returns of the speculative neighbor strategy in '+ticker, color=['red', 'blue'], ylabel='return in percent')
plt.grid(visible=True, which='both')
plt.show()
# and print the total trading return
print(tr.iloc[-1])