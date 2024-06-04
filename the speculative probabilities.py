#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/speculative-probabilities/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import os

import sys
file = open('output.txt', 'w')
sys.stdout = file

import warnings
warnings.filterwarnings("ignore")

# Set tickers and frequency of trading
set_tickers = ['DBB'] #['SPY', 'TNA', 'DBC', 'DBA', 'DBB', 'DBP', 'OIH', 'BTC-USD']
set_freq = '1mo'

# Set optional starting date
set_start = None #'2023-01-01'

# Set short, 0 or -1
set_short = -1.0

# Set rolling window for benchmark
set_roll = [13]# [3, 4, 5, 6, 9, 11, 12, 13]

# and whether you want recursive estimation of historical probabilities
set_recursive = False

# Finally, set the threshold for trading
set_thresh = [0.45] #np.arange(0.1, 0.55, 0.05)

for roll in set_roll:
    for thresh in set_thresh:
        # Set storage
        set_store = pd.DataFrame(data=None, index=['Rets', 'Bench', 'SP-I', 'SP-II', 'SP-III'], columns=set_tickers)

        # Now, do a loop over the tickers and over the observations
        for ticker in set_tickers:
            # Prepare the data
            data = yf.download(ticker, period='max', interval=set_freq)['Adj Close'].dropna()
            rets = data.pct_change().dropna()
            if set_start is not None:
                rets = rets.loc[set_start:]
            x = (rets > 0).astype(float)
            nobs = x.shape[0]

            # Initialize storage per ticker
            store = pd.DataFrame(data=None, index=x.index, columns=['Rets', 'Actual', 'BF', 'XF0', 'XF1', 'BE', 'XE', 'ell-0', 'ell-1'])
            trade = pd.DataFrame(data=None, index=x.index, columns=['Rets', 'Bench', 'SP-I', 'SP-II', 'SP-III'])
            signs = pd.DataFrame(data=None, index=x.index, columns=['Sign Actual', 'Sign SP-I', 'Sign SP-II', 'Sign SP-III'])

            # Loop over all observations
            for i in np.arange(roll, nobs-1, 1):
                # Split data and initialize probabilities
                last = x.iloc[i-1]
                actual = x.iloc[i]
                if set_recursive:
                    lagged = x.iloc[:i].mean()
                else:
                    lagged = x.iloc[(i-roll):i].mean()
                if i == roll:
                    pi_0 = lagged

                # Compute forecast errors and do probability updating
                ei_b = actual - lagged
                ei_0 = actual - pi_0
                # Compute the mean lambda from its range
                ell_0 = np.mean([(pi_0+0.5)/(pi_0-1), (pi_0+0.5)/pi_0])
                ell_1 = np.mean([(pi_0-0.5)/(pi_0-1), (pi_0-0.5)/pi_0])
                #ell_11 = np.mean([(pi_0-0.5)/(pi_0-1), (pi_0-0.5)/pi_0])
                #ell_12 = np.mean([(pi_0-0.5)/pi_0, (pi_0-0.5)/(pi_0-1)])
                # Select appropriate lambda
                if last > 0:
                    pi_1 = pi_0 + ell_1*(ei_0 - 0.5)
                    # if pi_0 > 0.5:
                    #     pi_1 = pi_0 + ell_11*(ei_1 - 0.5)
                    # else:
                    #     pi_1 = pi_0 + ell_12*(ei_1 - 0.5)
                else:
                    pi_1 = pi_0 + ell_0*(ei_0 - 0.5)
                # In case you get off the bounds revert to benchmark
                if (pi_1 < 0) or (pi_1 > 1):
                    pi_1 = lagged
                    #print('Probability correction!')

                # Compute trading signals and trade accordingly
                bs = (lagged >= thresh).astype(float) + set_short*(lagged < thresh).astype(float)
                ps1 = (pi_1 >= thresh).astype(float) + set_short*(pi_1 < thresh).astype(float)
                ps2 = (pi_1 >= lagged).astype(float) + set_short*(pi_1 < lagged).astype(float)
                ps3 = (lagged >= thresh and pi_1 > thresh).astype(float) + set_short*(lagged <= thresh or pi_1 < thresh).astype(float)
                # Store forecasts and trading returns per ticker
                store.iloc[i+1] = np.hstack([rets.iloc[i], actual, lagged, pi_0, pi_1, ei_b, ei_0, ell_0, ell_1])
                trade.iloc[i+1] = np.hstack([rets.iloc[i+1], rets.iloc[i+1]*bs, rets.iloc[i+1]*ps1, rets.iloc[i+1]*ps2, rets.iloc[i+1]*ps3])
                signs.iloc[i+1] = np.hstack([np.sign(rets.iloc[i+1]), ps1, ps2, ps3])
                # Update the current probability
                pi_0 = pi_1

            # Done, print and plot per ticker
            set_store[ticker] = (trade.dropna()+1).prod()
            #(trade+1).cumprod().plot()
            #plt.show()

        # Plot overall performance
        print('Roll=',roll)
        print('Threshold=',thresh)
        print(set_store.T)

file.close()
