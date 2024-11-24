#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/speculative-sign-smoothing-and-nearest-neighbors/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

## Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import yfinance as yf

## Define some functions
#
def get_trajectory(x, k):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    if k == 1:
        return x
    elif k > 1:
        y = np.zeros([x.shape[0] - k + 1, k * x.shape[1]])
        for i in range(x.shape[0] - k + 1):
            y[i, :] = np.hstack(x[i:(i + k), :])
        return y
    else:
        raise ValueError('In function get_trajectory the memory order must be >= 1')

##
def get_nn(x, p=2, alpha=0.8, step=1):
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    # Extract the last row from the trajectory matrix
    x_last = x[-1, :]
    # Compute all the distances, sort and extract the indices
    if p < 0:
        similarity_score = np.sum(np.logical_and(x[:-1, :], x_last), axis=1)
        winner_pos = np.argwhere(similarity_score >= -p).flatten()
        if len(winner_pos) == 0:
            return None
        _aux = np.argsort(similarity_score[winner_pos])
        isd = winner_pos[_aux][::-1]
    else:
        if p == 0:
            dxt = np.sum(np.abs(x[:-1, :] - x_last), axis=1)
        else:
            dxt = np.sum((np.abs(x[:-1, :] - x_last) ** p), axis=1) ** (1 / p)
        isd = dxt.argsort()
        isd = isd[np.where(isd < len(x))]
    isd = isd[:int(len(isd) * alpha) + (1 if p < 0 else 0)]
    # Subsample?
    if step > 1:
        ss = np.arange(0, len(isd), step)
        isd = isd[ss]
    # Done - note that the index of the closest NN is the first in isd!
    return isd

##
def get_nn_forecast(x, isd):
    isd = isd[isd < x.shape[0]]
    if len(x.shape) == 1:
        x = x.reshape((-1, 1))
    z = x[isd, :]
    # The forecast is about discrete values, use the mode
    return sp.stats.mode(z, axis=0)

##
def get_trained_gamma(s):
    s2 = s[2:]
    s1 = s[1:-1]
    s0 = s[:-2]
    ds = s1 - s0
    gamma = (s2 - ds)/s0
    return gamma

##
def get_combinator(r, set_k=10, set_prop=0.25):
    s = np.sign(r.values)
    s[s == 0.0] = 1.0
    gamma = get_trained_gamma(s)
    tmat = get_trajectory(gamma, k=set_k)
    target = gamma[(set_k-1):].reshape(-1, 1)
    nn = get_nn(tmat, p=-1, alpha=set_prop)
    fnn = get_nn_forecast(target, nn)[0][0]
    gammaf = get_nn_forecast(target, nn)[0][0]
    fnn = np.sign((s[-1]-s[-2]) + gammaf*s[-2])
    return np.sign(fnn)

## Now for the analysis

# Get some data
ticker = 'WEAT'
freq = '1d'
start_from = '2022-01-01'
data = yf.download(ticker, period='max', interval=freq)['Adj Close']
rets = data.pct_change().dropna()
rets = rets.loc[start_from:]
y = np.sign(rets.values)
# Set the parameters
max_roll = 30
set_nn = np.arange(3, (max_roll/2.0)+1, 1).astype(int)
set_prop = np.arange(0.1, 0.55, 0.05)
set_delay = 1
# Counters and storage
tot_count = 0
pos_count = 0
store_er = pd.DataFrame(data=np.zeros([1, 3]), index=[0], columns=['Embedding', '%NN', 'ER'])

# The main double loop, over the embedding dimension and the % of NN used forming the forecast
for nn in set_nn:
    for prop in set_prop:
        tot_count = tot_count + 1
        signal = rets.rolling(window=max_roll).apply(get_combinator, args=(nn, prop, ))
        strategy = rets*signal.shift(periods=set_delay)
        both = pd.concat([rets, strategy], axis=1).dropna()
        both.columns = [ticker, 'Smoothed NN']
        tr = ((both + 1).cumprod()-1)*100
        # tr.plot()
        # plt.show()
        er = tr.iloc[-1,1] - tr.iloc[-1,0]
        # print(nn, prop, er)
        if er > 0:
            pos_count = pos_count + 1
            # print(nn, round(prop, 3), round(er, 3))
            to_store = pd.DataFrame(data=np.array([nn, prop, er]).reshape(1, 3), index=[pos_count], columns=store_er.columns)
            store_er = pd.concat([store_er, to_store], axis=0)

# Trim the storage dataframe
store_er = store_er.iloc[1:]
# Compute the succesful proportion of trades across combinations
pos_prop = pos_count/tot_count
# Compute the descriptive statistics of the successful trades
stats = store_er['ER'].describe()
# Print and you are done
print(pos_prop)
print(stats)