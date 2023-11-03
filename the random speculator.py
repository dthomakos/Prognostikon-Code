#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/11/04/the-lazy-random-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# A function to compute the maximum drawdown, input is a dataframe of cumulative returns
def max_dd(crets):
    maxcret = (crets+1).cummax(axis=0)
    drawdowns = ((crets + 1) / maxcret) - 1
    return drawdowns.min(axis=0)


# Another function to collect performance measures, input is a dataframe of returns
def performance_measures(rets, f_factor, target_r=0):
    mu = rets.mean() * f_factor
    sd = rets.std() * np.sqrt(f_factor)
    sr = mu / sd
    er = target_r - rets
    er = er.clip(lower=0)
    l2 = (er ** 2).mean(axis=0)
    st = mu/np.sqrt(l2)
    cr = (rets+1).cumprod(axis=0) - 1
    md = max_dd(cr)
    stats = pd.DataFrame([mu, sd, sr, st, cr.iloc[-1], md])
    stats.index = ['Mean', 'Std. Dev.', 'Sharpe', 'Sortino', 'TR', 'MaxDD']
    return stats.transpose(), cr

# Select an ETF ticker, for the post I used SPY, TNA and LQD
ticker = 'LQD'

# Select frequency of trading
freq = '1d'

# Get some data after 2022
data = yf.download(ticker, period='max', interval=freq)['Adj Close']
r = data.pct_change().dropna().loc['2022-01-01':].dropna()
nobs = r.shape[0]

# Select replications and minimum leverage level
repl = 100
minlev = 1

# Select threshold and step
tt = -0.1
step = 0.01

# Initialize storage across replications
store_repl = np.zeros([nobs, repl])

# Loop across replications
for j in range(repl):
    print('Now running simulation = ', j)

    # Re-initialize capital
    K0 = 100
    B0 = K0

    # Initialize storage and tracking
    x0 = np.random.uniform(low=-minlev, high=+minlev, size=1)
    store = pd.DataFrame(data=None, index=r.index, columns=['x', ticker, 'TLP'])
    track = 0

    # Loop over the observations
    for i in range(nobs):

        ri = r.iloc[i]
        K1 = K0 + x0*K0*ri
        B1 = B0*(1 + ri)

        if np.sign(ri) != np.sign(x0):
            if x0 > 0:
                x1 = np.random.uniform(low=x0, high=minlev+x0, size=1)
            else:
                x1 = np.random.uniform(low=-minlev-x0, high=x0, size=1)
        else:
            if x0 < 0:
                x1 = np.random.uniform(low=x0, high=minlev+x0, size=1)
            else:
                x1 = np.random.uniform(low=-minlev-x0, high=x0, size=1)

        store.iloc[i] = np.hstack([x1, B1, K1])

        if (K1 - K0)/K0 > tt:
            if i > 1:
                x0 = store['x'].iloc[(i-1):(i+1)].mean()
            else:
                x0 = x1
            tt = tt + step
        else:
            tt = tt - step
        K0 = K1
        B0 = B1

    # Store and continue
    store_repl[:,j] = store['TLP'].values

# Compute the average across replications
store_avg = np.mean(store_repl, axis=1)

# Add to dataframe
store['TLP-avg'] = store_avg

# Set the frequency factor for performance evaluation
if freq == '1d':
    f_factor = 262
elif freq == '1wk':
    f_factor = 52
elif freq == '1mo':
    f_factor = 12

# Get the statistics, print and plot!
stats, cret = performance_measures(store[[ticker, 'TLP-avg']].pct_change().dropna(), f_factor)
print(round(stats, 3))
(cret *100).plot(grid='both', title='The (lazy) random speculator strategy for '+ticker+', daily trading', ylabel='return in percent', xlabel='Date')
plt.show()