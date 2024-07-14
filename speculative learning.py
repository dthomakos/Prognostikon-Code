#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/speculative-learning/
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

# Define the lazy moments signal
def mean_to_median(x):
    if x.median() != 0:
        return x.mean()/x.median()
    else:
        return x.mean()

# Input the data
ticker = 'SPY'
freq = '1d'
start_from = '2022-01-01'
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
rets = data.pct_change().dropna()
if start_from is not None:
    rets = rets.loc[start_from:]

# Input the parameters for rolling windows and resetting of probabilities
max_roll = 14
set_roll = np.arange(3, max_roll+1, 1)
nroll = len(set_roll)
delay = 1
reset = 6

# Compute upfront the signals and the corresponding errors across all rolling windows
signals = pd.DataFrame(data=None, index=rets.index, columns=set_roll)
errors = pd.DataFrame(data=None, index=rets.index, columns=set_roll)
#
for roll in set_roll:
    signals[roll] = rets.rolling(window=roll).apply(mean_to_median).apply(np.sign).shift(periods=delay)
    errors[roll] = rets.apply(np.sign) - signals[roll]

# Trim
signals = signals.dropna()
errors = errors.dropna()
rets = rets.loc[signals.index]
nobs = rets.shape[0]

# You can easily see the total returns of the separate rolling windows
tr_roll_windows = ((signals.mul(rets, axis='index') + 1).prod()-1)*100
print(tr_roll_windows)

# Initialize sign errors and loss holders
e_minus = np.zeros([nroll, 1])
e_plus = np.zeros([nroll, 1])
l_minus = np.ones([nroll, 1])
l_plus = np.ones([nroll, 1])
count = 0

# Initialize storage of strategies' returns
store_avg = pd.DataFrame(data=None, index=rets.index, columns=[np.hstack([ticker, 'Avg Sign & Wealth Loss', 'Avg Sign Loss', 'Avg Wealth Loss'])])
store1 = pd.DataFrame(data=None, index=rets.index, columns=[np.hstack([ticker, set_roll.astype(str)])])
store2 = pd.DataFrame(data=None, index=rets.index, columns=[np.hstack([ticker, set_roll.astype(str)])])
store3 = pd.DataFrame(data=None, index=rets.index, columns=[np.hstack([ticker, set_roll.astype(str)])])

# and perform the computations in a loop
for i in np.arange(0, nobs-1, 1):
    # Accumulate errors
    ei = errors.iloc[i].to_numpy().reshape(-1, 1)
    e_plus = e_plus + (ei == 2)
    e_minus = e_minus + (ei == -2)
    # Compute actual losses
    ra = rets.iloc[i]
    l_plus = (1 + np.abs(ra)*(ei == 2))*l_plus
    l_minus = (1 + np.abs(ra)*(ei == -2))*l_minus
    # Compute the three weighting schemes
    v_plus = 2 ** (-e_plus)
    v_minus = 2 ** (-e_minus)
    h_plus = v_plus/l_plus
    h_minus = v_minus/l_minus
    w_plus = h_plus/(h_minus + h_plus)
    w_minus = h_minus/(h_minus + h_plus)
    # Get the forward signal for averaging
    signal = signals.iloc[i+1].to_numpy().reshape(-1, 1)
    # Compute the average signal
    weights1 = 2 ** (h_plus + h_minus)
    weights1 = weights1/weights1.sum()
    avg_signal1 = np.sign(np.sum(weights1*signal))
    #
    weights2 = 2 ** (-e_plus-e_minus)
    weights2 = weights2/weights2.sum()
    avg_signal2 = np.sign(np.sum(weights2*signal))
    #
    weights3 = 2 ** (-l_plus-l_minus)
    weights3 = weights3/weights3.sum()
    avg_signal3 = np.sign(np.sum(weights3*signal))
    # and finally compute the signals based on the losses
    bench = rets.iloc[i+1]
    trade1 = np.sign(w_plus  - w_minus)*bench
    trade2 = np.sign(v_plus  - v_minus)*bench
    trade3 = np.sign(-l_plus + l_minus)*bench
    trade_avg1 = avg_signal1*bench
    trade_avg2 = avg_signal2*bench
    trade_avg3 = avg_signal3*bench
    store_avg.iloc[i+1] = np.hstack([bench, trade_avg1, trade_avg2, trade_avg3])
    store1.iloc[i+1] = np.hstack([bench, trade1.flatten()])
    store2.iloc[i+1] = np.hstack([bench, trade2.flatten()])
    store3.iloc[i+1] = np.hstack([bench, trade3.flatten()])
    # Add to the counter and reset the holders if needed
    count = count + 1
    if (count > 0) and (count%reset == 0):
        e_minus = np.zeros([nroll, 1])
        e_plus = np.zeros([nroll, 1])
        l_minus = np.ones([nroll, 1])
        l_plus = np.ones([nroll, 1])

# Compute the cumulative returns
cr_avg = (store_avg.dropna() + 1).cumprod()
cr1 = (store1.dropna() + 1).cumprod()
cr2 = (store2.dropna() + 1).cumprod()
cr3 = (store3.dropna() + 1).cumprod()

# Compute and print the final returns
tr_avg = cr_avg.iloc[-1]
tr1 = cr1.iloc[-1]
tr2 = cr2.iloc[-1]
tr3 = cr3.iloc[-1]
tr_all = pd.concat([tr1, tr2, tr3], axis=1)
tr_max = tr_all.apply(np.argmax, axis=0)

# Print
print(tr_avg)
print(tr_all)

# Prepare for a plot of the best performers across strategy types and rolling windows
cr_max1 = cr1[set_roll[tr_max.iloc[0]-1].astype(str)]
cr_max2 = cr2[set_roll[tr_max.iloc[1]-1].astype(str)]
cr_max3 = cr3[set_roll[tr_max.iloc[2]-1].astype(str)]
#
cr_all = pd.concat([cr_avg, cr_max1, cr_max2, cr_max3], axis=1)
cr_all.columns = [ticker, 'Avg Sign & Wealth Loss', 'Avg Sign Loss', 'Avg Wealth Loss', 'Sign & Wealth Loss', 'Sign loss', 'Wealth Loss']

# Plot and print the final result
cr_rets = (cr_all - 1)*100
cr_rets.plot(grid='both', xlabel='Date', ylabel='returns in percent', title='The Speculative Learning Strategy')
plt.show()
#
print(cr_rets.iloc[-1])
cr_rets.iloc[-1].to_csv('xxx.csv')
#
cr_rets[[ticker, 'Avg Sign & Wealth Loss', 'Avg Sign Loss', 'Avg Wealth Loss']].plot(grid='both', xlabel='Date', ylabel='returns in percent', title='The Speculative Learning Strategy')
plt.show()


