#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/summer-speculations/
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

# Select ticker and frequency of trading
ticker = 'GLD'
freq = '1d'

# Get the data, section if needed
data = yf.download(ticker, period='max', interval=freq)['Adj Close']
rets = data.pct_change().dropna()
start_from = '2022-01-01'
if start_from is not None:
    rets = rets.loc[start_from:]

# Select maximum for the parameters
delay_max = 6
ind_roll_max = 14
est_roll_max = 21

# Initialize a counter and storage of results
counter = 0
store = pd.DataFrame(data=np.zeros([1, 7]), index=[counter], columns=['Delay', 'Ind Roll', 'Est Roll', 'S00', 'S01', 'S02', 'S03'])

# A triple search loop
for delay in np.arange(2, delay_max+1, 1):
    for ind_roll in np.arange(-1, ind_roll_max+1, 1):
        for est_roll in np.arange(2, est_roll_max+1, 1):

            # Compute the indicator
            if ind_roll > 0:
                indicator = rets.rolling(window=ind_roll).sum().shift(periods=delay)
            else:
                indicator = rets.shift(periods=delay)

            # Compute the signals as per the blog
            signal_00 = 1 - 2*(rets.shift(periods=delay) < 0)
            signal_01 = 1 - 2*(indicator < 0)
            prob_top = (rets < 0).rolling(window=est_roll).mean().shift(periods=1)
            prob_bot = prob_top.shift(periods=delay)
            signal_02 = 1 - 2*(prob_top/prob_bot)*(rets.shift(periods=delay) < 0)
            prob_bot = (indicator < 0).rolling(window=est_roll).mean().shift(periods=delay)
            signal_03 = 1 - 2*(prob_top/prob_bot)*(indicator.shift(periods=delay) < 0)

            # Compute the returns of each strategy
            s00 = rets*(signal_00.apply(np.sign))
            s01 = rets*(signal_01.apply(np.sign))
            s02 = rets*(signal_02.apply(np.sign))
            s03 = rets*(signal_03.apply(np.sign))

            # Put everything together, store and continue
            all = pd.concat([rets, s00, s01, s02, s03], axis=1).dropna()

            # Compute the cumulative and the excess returns over the B&H
            cr = ((all + 1).cumprod()-1)*100
            er = cr.iloc[-1,1:] - cr.iloc[-1, 0]

            # If any strategy has positive excess returns store it
            if any(er > 0):
                counter = counter + 1
                tt = np.hstack([delay, ind_roll, est_roll, er.to_numpy().flatten()]).reshape(1, 7)
                tt = pd.DataFrame(data=tt, index=[counter], columns=store.columns)
                store = pd.concat([store, tt], axis=0)

# Find the best performers
max_id = store.apply(np.argmax, axis=0)
er00 = store.iloc[max_id['S00'],[0, 1, 2, 3]].to_numpy().reshape(-1, 1)
er01 = store.iloc[max_id['S01'],[0, 1, 2, 4]].to_numpy().reshape(-1, 1)
er02 = store.iloc[max_id['S01'],[0, 1, 2, 5]].to_numpy().reshape(-1, 1)
er03 = store.iloc[max_id['S01'],[0, 1, 2, 6]].to_numpy().reshape(-1, 1)

# Put together and print
er_all = pd.DataFrame(np.c_[er00, er01, er02, er03], index = ['d', 'M', 'R', 'ER'], columns = ['S1', 'S2', 'S3', 'S4'])
#
print(round(er_all,3))