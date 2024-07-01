#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-random-enforcer/
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

# Define the random enforcer, note the use of the global variable _W
def random_enforcer(x, W=None, B=10, interval=(-1, 1)):
    nobs = x.shape[0]
    if W is None:
        W = np.random.uniform(size=(B, nobs), low=interval[0], high=interval[1])
    z = W@(x.to_numpy().reshape(nobs, 1))
    mu = np.mean(z)
    md = np.median(z)
    global _W
    _W =+ W
    return np.sign(mu/md)

# Train the random enforcer
def train_random_enforcer(data, R, delay, to_discrete=True, threshold=1, repl=10, bounds=(-1, 1)):
    rets = data
    if to_discrete:
        y = rets.apply(np.sign)
    else:
        y = rets
    do = True
    while do:
        signal = y.rolling(window=R).apply(random_enforcer, args=(None, repl, bounds, )).shift(periods=delay)
        both = pd.concat([rets, rets*signal], axis=1).dropna()
        tr = (both+1).prod()
        if (tr.iloc[1] - tr.iloc[0]) > threshold:
            do = False
            break
    return _W, tr

# Select ticker, period of computation and frequency
ticker = 'OIH'
start_date = '2019-01-01'
freq = '1wk'
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
if start_date is not None:
    data = data.loc[start_date:]
#
rets = data.pct_change().dropna()
y = rets.apply(np.sign)
nobs = y.shape[0]

# Set the parameters of the evaluation
set_repl = 3
set_tt = [0.65, 0.70, 0.75, 0.80]
set_roll = [2, 3, 4, 6, 12]

# Initialize storage
store_in = pd.DataFrame(data=None, index=set_tt, columns=set_roll)
store_out = pd.DataFrame(data=None, index=set_tt, columns=set_roll)

# Set the parameters of training
set_discrete = True
set_threshold = 0
set_d = 2
set_B = 5
set_interval = (-1, 1)

# Now, loop over the different periods of evaluation and produce results!
for tt in set_tt:
    y_in = y.iloc[:int(tt*nobs)]
    y_out = y.iloc[int(tt*nobs):]
    rets_in = rets.iloc[:int(tt*nobs)]
    rets_out = rets.iloc[int(tt*nobs):]

    for roll in set_roll:
        print('Now doing tt=',tt,'and roll=',roll)
        #
        W_all = None
        for repl in range(set_repl):
            W, tr_in = train_random_enforcer(rets_in, roll, set_d, set_discrete, set_threshold, set_B, set_interval)
            # Average the W matrix
            W = W/(nobs - roll+1)
            W_all =+ W
        # Average the average W matrix
        W_all = W_all/set_repl
        signal_out = rets_out.rolling(window=roll).apply(random_enforcer, args=(W_all, )).shift(periods=set_d)
        both_out = pd.concat([rets_out, rets_out*signal_out], axis=1).dropna()
        tr_out = (both_out+1).prod()
        # Store in-sample (training) and out-of-sample (evaluation) performance as excess return
        store_in.loc[tt, roll] = tr_in.iloc[1]-tr_in.iloc[0]
        store_out.loc[tt, roll] = tr_out.iloc[1]-tr_out.iloc[0]

# Print results for examination
print(store_in)
print(store_out, 3)
ps = (store_out > 0).mean()
print(ps)
print(store_out.max())
print(store_out.mean())
