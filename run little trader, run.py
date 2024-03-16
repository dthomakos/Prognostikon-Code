#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/run-little-trader-run/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plot
import numpy as np
import pandas as pd
import yfinance as yf

# Define a function to compute runs and reversals and associated probabilities
def runs_to_reversals(x):
    n = len(x)
    runs = 1

    for i in np.arange(1, n, 1):
        if x.iloc[i] != x.iloc[i-1]:
            runs = runs + 1

    pruns = runs/n
    prevr = 1 - pruns

    return pruns/prevr, pruns, prevr

# Select a ticker to analyze, starting and ending dates, frequency of rebalancing,the rolling window and the alpha for the runs-to-reversals ratio
ticker = 'OIH'
start_date = '2010-01-01'
end_date = '2024-02-29'
freq = '1d'
# roll = 12
# alpha = 0.75
if freq == '1d':
    freq_name = 'daily'
    roll_set = [14, 21, 30, 63]
elif freq == '1wk':
    freq_name = 'weekly'
    roll_set = [12, 26, 52]
elif freq == '1mo':
    freq_name = 'monthly'
    roll_set = [12, 24, 36]

# I made a loop to find the best values of the pair (roll, alpha) below and you can easily adapt the code to your liking!
for roll in roll_set:
    for alpha in [0.25, 0.5, 0.75, 1.0, 1.05, 1.15, 1.25, 2]:
        # Download the data, compute returns and signs, compute the rolling signal variable
        data = yf.download(ticker, start=start_date, end=end_date, interval=freq)['Adj Close'].dropna()
        y = data.pct_change().dropna()
        s = np.sign(y)
        rr = s.rolling(window=21).apply(lambda x: runs_to_reversals(x)[0])
        signal = (rr > alpha).astype(float) - (rr <= alpha).astype(float)

        # Compute the strategy's returns, the cumulative returns, plot and print - that's it!
        r2r = y*(signal.shift(periods=1).apply(np.sign))
        both = pd.concat([y, r2r], axis=1).iloc[(roll+1):]
        both.columns = [ticker, 'R2R strategy']
        #
        cr = ((both+1).cumprod()-1)*100
        # Print only the meaningful cases
        if cr.iloc[-1,1] > cr.iloc[-1,0]:
            print('roll = ',roll,'alpha = ',alpha)
            print(cr.iloc[-1])
            # and plot if desired!
            # if (roll == 63) and (alpha == 1.05):
            #     cr.plot(title='The Runs-to-Reversals Strategy for '+ticker+' for '+freq_name+' data', xlabel='Date', ylabel='return in percent', grid='both')
            #     plot.show()

