#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-lazy-moments-strategy/
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
ticker = 'DBC'
freq = '1wk'
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
rets = data.pct_change().dropna()
# Crop data if desired
start_date = '2018-01-01'
if start_date is not None:
    rets = rets.loc[start_date:]

# Select range of rolling windows and delay of signal
roll_range = np.arange(3, 25, 1)

# Roll over the delays
for delay in np.arange(1, 7, 1):

    # Initialize storage
    store = pd.DataFrame(data=None, index=rets.index, columns=roll_range)

    # Compute strategy returns for all rolling windows
    for roll in roll_range:
        signal = rets.rolling(window=roll).apply(lambda x: x.mean()/x.median()).shift(periods=delay).apply(np.sign)
        strategy = rets*signal
        store[roll] = strategy

    # Crop returns, compute cumulative returns, find the best rolling window each period
    store = store.dropna()
    cumret = ((store+1).cumprod()-1)*100
    idx_col = cumret.apply(np.argmax, axis=1).to_numpy()
    # Initialize storage for wealth rotation
    rotation = pd.DataFrame(data=None, index=cumret.index, columns=['Rotation'])
    rotation.loc[cumret.index[0]] = store.iloc[0, idx_col[0]]

    # Compute the rotation's returns, carefull to use store and not cumret here!
    for i in range(cumret.shape[0]-1):
        rotation.loc[cumret.index[i+1]] = store.iloc[i+1, idx_col[i]]

    # add to all results
    cr_rotation = ((rotation+1).cumprod()-1)*100
    cumret = pd.concat([cumret, cr_rotation], axis=1)

    # Add the benchmark, plot and print
    cumret[ticker] = ((rets.loc[store.index]+1).cumprod()-1)*100
    tr = cumret.iloc[-1]
    to_plot = cumret[[roll_range[tr[roll_range].argmax()], 'Rotation', ticker]]
    to_plot.plot(title='The lazy moments strategy for '+ticker+', weekly rebalancing for delay d='+str(delay), xlabel='Date', ylabel='returns in percent', grid='both')
    plt.show()
    #
    print('Delay =', delay)
    print(to_plot.iloc[-1])