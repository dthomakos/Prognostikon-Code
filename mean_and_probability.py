#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/07/09/the-probable-speculative-constant/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import yfinance as yf

# A function to compute the trajectory matrix for the ecdf calculation
def get_trajectory(x, k):
    """
    Compute the trajectory matrix of a matrix given a memory parameter

    :param x:           array of data
    :param k:           scalar, memory order
    :return:            the trajectory matrix
    """
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

# Download some data
ticker = 'BTC-USD'
data = yf.download(ticker, period='max', interval='1d')['Adj Close'].dropna()
# Get percent returns, select period of estimation/evaluation
y = data.pct_change().dropna().loc['2022-01-01':]

# Selec the rolling window - note that this will be used twice
roll = 4
# Compute the rolling mean and the predictive errors
mu = y.rolling(window=roll).mean()
e = (y - mu.shift(periods=1)).dropna()

# Use the same rolling window to compute the trajectory matrix of the predictive errors
zmat = get_trajectory(e.to_numpy(), roll)
# Carefully align the rolling mean with the values of the trajectory matrix
mmat = mu.iloc[(2*roll-2):-1].to_numpy().reshape(-1, 1)
# Compute the ecdf easily
pmat = np.apply_along_axis(np.mean, 1, (zmat <= -mmat))

# Align the evaluation returns
ymat = y.iloc[2*roll:-1].to_numpy().reshape(-1, 1)
# and, again, carefully align the rolling mean and ecdf for the signal (note the renaming)
mmat = mu.iloc[(2*roll-1):-2].to_numpy().reshape(-1, 1)
pmat = pmat[:-2].reshape(-1, 1)

# Put everything together, index correctly
together = pd.DataFrame(data=np.c_[ymat, mmat, pmat], index=y.iloc[2*roll:-1].index, columns=['Actual', 'Mean', 'Prob'])

# Initialize the dataframe for the strategy returns
rr = pd.DataFrame(data=None, index=together.index, columns=['Buy & Hold', 'Probable Speculative Constant AND', 'Probable Speculative Constant OR', 'Speculative Constant'])

# Set the probability for the ecdf
prob = 0.5

# Get the strategy via a  loop
for i in range(together.shape[0]):
    xi = together.iloc[i, :]
    mi = xi['Mean']
    pi = xi['Prob']
    # The probable speculative constant - and
    if (mi > 0) and (pi <= prob):
        rr.iloc[i, 1] = xi['Actual']
    elif (mi < 0) and (pi > prob):
        rr.iloc[i, 1] = -xi['Actual']
    else:
        rr.iloc[i, 1] = 0
    # The probable speculative constant - or
    if (mi > 0) or (pi <= prob):
        rr.iloc[i, 2] = xi['Actual']
    elif (mi < 0) or (pi > prob):
        rr.iloc[i, 2] = -xi['Actual']
    else:
        rr.iloc[i, 2] = 0
    # The speculative constant alone
    if (mi > 0):
        rr.iloc[i, 3] = xi['Actual']
    elif (mi < 0):
        rr.iloc[i, 3] = -xi['Actual']
    # The benchmark
    rr.iloc[i, 0] = xi['Actual']

# Do a nice plot
tr = ((rr+1).cumprod()-1)*100
tr.plot(title='Total trading returns of the speculative constant strategies in '+ticker, color=['red', 'blue', 'green', 'orange'], ylabel='return in percent')
plt.grid(visible=True, which='both')
plt.show()
# and print the total trading return
print(roll)
print(tr.iloc[-1])