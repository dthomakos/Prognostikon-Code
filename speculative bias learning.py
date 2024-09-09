#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/speculative-bias-learning/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance

# Select ticker and frequency of rebalancing
ticker = 'BTC-USD'
freq = '1d'
# Select cut-off date
start_from = '2022-01-01'
# Get data and returns
data = yfinance.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
data = data.loc[start_from:]
rets = data.pct_change().dropna()
y = rets.apply(np.sign)

# Prepare storage
e_store_reg = pd.DataFrame(data=None, index=rets.index, columns=['ex', 'ey'])
e_store_bin = pd.DataFrame(data=None, index=rets.index, columns=['ex', 'ey'])
rets_reg = pd.DataFrame(data=None, index=rets.index, columns=['B&H', 'Mean', 'Bias Learning'])
rets_bin = pd.DataFrame(data=None, index=rets.index, columns=['B&H', 'Mean', 'Bias Learning'])

# Number of observations and printing tracker
nobs = rets.shape[0]
try_max_reg = 1
try_max_bin = 1

# A double loop for searching
for yroll in np.arange(3, 12, 1):
    for broll in np.arange(2, 6, 1):
        # Gamma parameter is fixed, you can experiment on it!
        gamma = -1.0

        # A loop over the observations
        for i in np.arange(0, nobs-yroll, 1):
            ri = rets.iloc[i:(i + yroll)]
            ra = rets.iloc[i + yroll]
            yi = y.iloc[i:(i + yroll)]
            ya = y.iloc[i + yroll]

            # Compute the input forecasts
            xf_reg = np.sign(ri.mean())
            xf_bin = np.sign(yi.mean())
            ex_reg = ra - xf_reg
            ex_bin = ya - xf_bin

            # and do the updating
            if i == 0:
                yf_reg = xf_reg
                yf_bin = xf_bin
                ey_reg = 0
                ey_bin = 0
                e_store_reg.iloc[i + yroll] = np.hstack([ex_reg, ey_reg])
                e_store_bin.iloc[i + yroll] = np.hstack([ex_bin, ey_bin])
            else:
                if (i <= broll):
                    bias_reg = e_store_reg.iloc[:(i + yroll), 0].mean()
                    bias_bin = e_store_bin.iloc[:(i + yroll), 0].mean()
                else:
                    bias_reg = e_store_reg.iloc[(i + yroll - broll):(i + yroll), 1].mean()
                    bias_bin = e_store_bin.iloc[(i + yroll - broll):(i + yroll), 1].mean()
                #
                yf_reg = np.sign(xf_reg + gamma*bias_reg)
                yf_bin = np.sign(xf_bin + gamma*bias_bin)
                ey_reg = ra - yf_reg
                ey_bin = ya - yf_bin
                #
                e_store_reg.iloc[i + yroll] = np.hstack([ex_reg, ey_reg])
                e_store_bin.iloc[i + yroll] = np.hstack([ex_bin, ey_bin])

            # Compute the per period trading returns
            rets_reg.iloc[i + yroll] = ra*np.hstack([1, xf_reg, yf_reg])
            rets_bin.iloc[i + yroll] = ra*np.hstack([1, xf_bin, yf_bin])

        # Compute total trading returns
        tr_reg = (rets_reg + 1).prod()
        tr_bin = (rets_bin + 1).prod()

        # Do selecting printing and plotting
        if tr_reg['Bias Learning'] > try_max_reg:
            print('Regular returns with yroll =',yroll,'and broll =',broll)
            print(tr_reg)
            try_max_reg = tr_reg['Bias Learning']
            (rets_reg + 1).cumprod().plot(title='NAV of $1 for the bias learning strategy for '+ticker+', continuous data', ylabel='NAV of $1', xlabel='Date', grid='both')
            plt.show()
        if tr_bin['Bias Learning'] > try_max_bin:
            print('Binary returns with yroll =',yroll,'and broll =',broll)
            print(tr_bin)
            try_max_bin = tr_bin['Bias Learning']
            (rets_bin + 1).cumprod().plot(title='NAV of $1 for the bias learning strategy for '+ticker+', binary data', ylabel='NAV of $1', xlabel='Date', grid='both')
            plt.show()
