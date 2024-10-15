#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-speculative-switch/
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

# Select ticker and frequency of rebalancing
ticker = 'ETH-USD'
freq = '1d'
# Get the data and crop appropriately
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
rets = data.pct_change().dropna()
rets = rets.loc['2022-01-01':]

# Initialize parameters storage
nobs = rets.shape[0]
switch = 1
counter = 0
store = None
ch = 0

# A triple loop over rolling windows, delay values and the evaluation over observations
for roll in np.arange(3, 23, 1):
    for delay in np.arange(0, 4, 1):

        # Initialize storage
        strategy = pd.DataFrame(data=None, index=rets.index, columns=[ticker, 'S'])

        for i in np.arange(roll, nobs, 1):
            # Get the trading sign
            mu = rets.iloc[(i-roll-delay):(i-delay)].mean()
            if switch > 0:
                if np.sign(mu) > 0:
                    trade = 1
                else:
                    trade = -1
            else:
                if np.sign(mu) > 0:
                    trade = -1
                else:
                    trade = 1
            # Store performances
            strategy.iloc[i, :] = np.hstack([rets.iloc[i], rets.iloc[i]*trade])
            # Apply the switching rule, note the use of the last 3 observations!!
            if i > 0:
                tr = (strategy.iloc[(i-2):i]+1).prod()
                if ((tr[ticker] > tr['S']) and (trade > 0)) or ((tr[ticker] < tr['S']) and (trade < 0)):
                    switch = -switch

        ch = ch + 1
        # Compute performance and save accordingly
        tr_all = (strategy+1).prod()
        if (tr_all[ticker] < tr_all['S']) and (tr_all['S'] > 1):
            excess_return = tr_all['S'] - tr_all[ticker]
            new = pd.DataFrame(data=np.hstack([roll, delay, excess_return]).reshape(1, 3), index=[counter], columns=['roll', 'delay', 'excess return'])
            store = pd.concat([store, new], axis=0)
            # print('roll=',roll,'delay=',delay)
            # print(tr_all)
            # Update the counter
            counter = counter + 1

# Compute the average excess return per delay value
m0 = store.loc[store['delay']==0].mean()
m1 = store.loc[store['delay']==1].mean()
m2 = store.loc[store['delay']==2].mean()
m3 = store.loc[store['delay']==3].mean()
#
m_all = np.hstack([m0.iloc[2], m1.iloc[2], m2.iloc[2], m3.iloc[2]])
m_all = pd.DataFrame(data=m_all, index=np.arange(0, 4, 1), columns=['Average Excess Return'])
# Print
print(store)
#
print(store.shape[0]/ch)
#
print(m_all)
#
print(store.loc[store['excess return']==store['excess return'].max()])
print(store.loc[store['excess return']==store['excess return'].min()])