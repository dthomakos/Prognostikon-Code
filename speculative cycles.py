#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2024/01/19/the-speculative-cycles/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
# Code for the training of the ESN adapted from:
#
# """
# A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data
# in "plain" scientific Python.
# from https://mantas.info/code/simple_esn/
# (c) 2012-2020 Mantas Lukoševičius
# Distributed under MIT license https://opensource.org/licenses/MIT
# """
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Here is a selection of tickers that I used for the post
tickers = ['SPY', 'QQQ', 'IWF', 'SSO', 'SH', 'HYG', 'TLT', 'EEM', 'GLD', 'OIH', 'TNA', 'DBA', 'DBC', 'UNG', 'WEAT', 'DBB', 'DBP', 'XLE', 'XLF', 'XLP', 'XLI', 'XLK', 'XLU', 'XLB', 'XLV', 'XLY', 'XLRE']

# Search for cycles within 5 to 30 days - clearly you can increase this!!
roll_seq = np.arange(10, 61, 1)

# Initialize storage for the optimal cycle lengths and cycle profits
store_cycles = pd.DataFrame(data=None, index=np.arange(1, len(roll_seq)+1), columns=tickers)
store_profits = pd.DataFrame(data=None, index=np.arange(1, len(roll_seq)+1), columns=tickers)

# A loop around all tickers, daily data used only
for ticker in tickers:
    # Download the data and compute returns, leave the last 3 years out!!
    data = yf.download(ticker, interval='1d', start='1995-01-01', end='2020-12-31')['Adj Close'].dropna()
    rets = data.pct_change().dropna()
    nobs = rets.shape[0]

    # Initialize within loop storage around the cycle days
    store_ex = pd.DataFrame(data=None, index=roll_seq, columns=['Excess'])

    # Then an inner loop around the cycle days
    for roll in roll_seq:
        seqr = np.arange(0, nobs, roll)
        store = pd.DataFrame(data=None, index=rets.index[seqr[:-1]], columns=['Bench'])

        # And another inner loop to compute the non-overlapping returns
        for i in range(len(seqr)):
            if seqr[i] >= seqr[-1]:
                break
            else:
                xi = rets.iloc[seqr[i]:seqr[i+1]]
                mi = (xi+1).prod()-1
                store.iloc[i] = mi

        # Once you have these non-overlapping returns compute the signal variable
        store_signal = (store.apply(np.sign)).rolling(window=5).quantile(quantile=0.75)
        # and trade...
        store['Cycles'] = store.multiply(store_signal.shift(periods=1), axis='index')
        store = store.dropna()
        # Compute the total returns and the excess return
        cret = ((store+1).cumprod()-1)*100
        store_ex.loc[roll] = cret.iloc[-1,1] - cret.iloc[-1, 0]

    # Now, sort the cycles by excess return, find profitable cycles
    sorted = store_ex.sort_values(by='Excess', ascending=False)
    profitable_cycles = sorted.iloc[np.where(sorted.values > 0)[0]]

    # Store the profitable cycles and the corresponding excess return
    store_cycles[ticker].iloc[:len(profitable_cycles)] = profitable_cycles.index.to_numpy()
    store_profits[ticker].iloc[:len(profitable_cycles)] = profitable_cycles.values.flatten()

# Save first batch of results
store_cycles.to_csv('store-cycles-2020.csv')
store_profits.to_csv('store-profits-2020.csv')

# Done, now for the post processing
#
# Lets look at the top three performing cycle lengths
top3_cycles = store_cycles.iloc[:3,:]
top3_profits = store_profits.iloc[:3,:]

# Compute the quartiles of cycle lengths
top3_cycles_q1 = top3_cycles.T.apply(lambda x: np.quantile(x.dropna(),q=0.25))
top3_cycles_q2 = top3_cycles.T.median()
top3_cycles_q3 = top3_cycles.T.apply(lambda x: np.quantile(x.dropna(),q=0.75))

# Put this cycles together and order them
top3_cycles_dist = np.unique(np.hstack([top3_cycles_q1, top3_cycles_q2, top3_cycles_q3]).astype(int))
top3_cycles_median = np.median(top3_cycles_dist)
if top3_cycles_median not in top3_cycles_dist:
    top3_cycles_dist = np.hstack([top3_cycles_dist, int(top3_cycles_median)])

# OK, now lets repeat the experiment with the given cycles only for the test data of the past 3 years!!

# Search for cycles within 5 to 30 days - clearly you can increase this!!
roll_seq = top3_cycles_dist

# Initialize storage for the optimal cycle lengths and cycle profits
store_cycles = pd.DataFrame(data=None, index=np.arange(1, len(roll_seq)+1), columns=tickers)
store_profits = pd.DataFrame(data=None, index=np.arange(1, len(roll_seq)+1), columns=tickers)

# A loop around all tickers, daily data used only
for ticker in tickers:
    # Download the data and compute returns, leave 5 years out
    data = yf.download(ticker, interval='1d', start='2021-01-01', end='2023-12-31')['Adj Close'].dropna()
    rets = data.pct_change().dropna()
    nobs = rets.shape[0]

    # Initialize within loop storage around the cycle days
    store_ex = pd.DataFrame(data=None, index=roll_seq, columns=['Excess'])

    # Then an inner loop around the cycle days
    for roll in roll_seq:
        seqr = np.arange(0, nobs, roll)
        store = pd.DataFrame(data=None, index=rets.index[seqr[:-1]], columns=['Bench'])

        # And another inner loop to compute the non-overlapping returns
        for i in range(len(seqr)):
            if seqr[i] >= seqr[-1]:
                break
            else:
                xi = rets.iloc[seqr[i]:seqr[i+1]]
                mi = (xi+1).prod()-1
                store.iloc[i] = mi

        # Once you have these non-overlapping returns compute the signal variable
        store_signal = (store.apply(np.sign)).rolling(window=5).quantile(quantile=0.75)
        # and trade...
        store['Cycles'] = store.multiply(store_signal.shift(periods=1), axis='index')
        store = store.dropna()
        # Compute the total returns and the excess return
        cret = ((store+1).cumprod()-1)*100
        store_ex.loc[roll] = cret.iloc[-1,1] - cret.iloc[-1, 0]

    # Now, sort the cycles by excess return, find profitable cycles
    sorted = store_ex.sort_values(by='Excess', ascending=False)
    profitable_cycles = sorted.iloc[np.where(sorted.values > 0)[0]]

    # Store the profitable cycles and the corresponding excess return
    store_cycles[ticker].iloc[:len(profitable_cycles)] = profitable_cycles.index.to_numpy()
    store_profits[ticker].iloc[:len(profitable_cycles)] = profitable_cycles.values.flatten()

# Save second batch of results
store_cycles.to_csv('store-cycles-2023.csv')
store_profits.to_csv('store-profits-2023.csv')

# Collect the top performers
top = pd.concat([store_cycles.iloc[0,:].T, store_profits.iloc[0,:].T], axis=1)

# Done, now for the post processing, here we care for performance and the choice of the cycle
median_cycle = store_cycles.median(axis=1)
q1_cycles = store_cycles.apply(lambda x: np.quantile(x,q=0.25), axis=1)
q3_cycles = store_cycles.apply(lambda x: np.quantile(x,q=0.75), axis=1)

# and a nice plot with a single ticker
ticker = 'WEAT'

data = yf.download(ticker, interval='1d', start='2021-01-01', end='2023-12-31')['Adj Close'].dropna()
rets = data.pct_change().dropna()
nobs = rets.shape[0]

# Initialize within loop storage around the cycle days
store_ex = pd.DataFrame(data=None, index=roll_seq, columns=['Excess'])
roll = 22

seqr = np.arange(0, nobs, roll)
store = pd.DataFrame(data=None, index=rets.index[seqr[:-1]], columns=[ticker])

# And another inner loop to compute the non-overlapping returns
for i in range(len(seqr)):
    if seqr[i] >= seqr[-1]:
        break
    else:
        xi = rets.iloc[seqr[i]:seqr[i+1]]
        mi = (xi+1).prod()-1
        store.iloc[i] = mi

# Once you have these non-overlapping returns compute the signal variable
store_signal = (store.apply(np.sign)).rolling(window=5).quantile(quantile=0.75)
# and trade...
store['Speculative Cycle'] = store.multiply(store_signal.shift(periods=1), axis='index')
store = store.dropna()
# Compute the total returns and the excess return
cret = ((store+1).cumprod()-1)*100
cret.plot(grid='both', title='The speculative cycle strategy for '+ticker+' for a cycle of '+str(roll)+' days', xlabel='Date', ylabel='return in percent')
plt.show()