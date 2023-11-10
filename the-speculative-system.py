#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/11/11/the-speculative-system/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tools.sm_exceptions import ValueWarning
import warnings
warnings.simplefilter('ignore', ValueWarning)

# Download some data, for the post I used
#
# GLD (1, True, 12) and set_fcst_type = 0
# DBA (18, True, 6) and set_fcst_type = 2
# WEAT (12, True, 12) and set_fcst_type = 0
# SH (1, False, 1) and set_fcst_type = 2
# SPY (3, False, 36) and set_fcst_type = 0
# FXE (2, True, 24) and set_fcst_type = 2
ticker = 'FXE'
data = yf.download(ticker, period='max', interval='1mo')['Adj Close'].dropna()
r = data.pct_change().dropna()#.loc['2020-01-01':] # you can change this of course
dr = r.diff()
# Get the threshold variables
z1p = ((dr > 0) & (r > 0)).astype(float)
z2p = ((dr <= 0) & (r > 0)).astype(float)
z1n = ((dr <= 0) & (r <= 0)).astype(float)
z2n = ((dr > 0) & (r <= 0)).astype(float)
# Convert the zeroes to -ones, this is important!!
z1p.iloc[z1p == 0] = -1.0
z2p.iloc[z2p == 0] = -1.0
z1n.iloc[z1n == 0] = -1.0
z2n.iloc[z2n == 0] = -1.0
# Put together
z = pd.concat([r, dr, z1p, z2p, z1n, z2n], axis=1).dropna()
z.columns = [ticker, 'D-'+ticker, 'Z1+', 'Z2+', 'Z1-', 'Z2-']

# Number of observations and initial window
nobs = z.shape[0]
ini_wind = 60
set_lag = 2
use_exog = True
set_fcst_type = 2
train_every = 24

# and initialize storage
store = pd.DataFrame(data=None, index=z.index, columns=[ticker, 'Speculative System'])

# Run a simple loop to get the signals and the strategy returns
for i in np.arange(0, nobs-ini_wind, 1):
    z_i = z.iloc[:(i+ini_wind),:] # this is recursive estimation, change to i:(i+ini_wind) for rolling!!
    # Use the threshold variables?
    if use_exog:
        y_i = z_i.iloc[1:,[0,1]]
        x_i = z_i.iloc[:-1,2:].apply(np.sign)
        x_i.index = y_i.index
        if (i%train_every) == 0:
            model = VAR(endog=y_i, exog=x_i).fit(set_lag)
        x_f = z_i.iloc[-1,2:].values.reshape(1, 4)
        fcst = model.forecast(y=y_i.values[-set_lag:], steps=1, exog_future=x_f)[0]
    # Or not?
    else:
        y_i = z_i.iloc[:,[0,1]]
        if (i%train_every) == 0:
            model = VAR(endog=y_i).fit(set_lag)
        fcst = model.forecast(y=y_i.values[-set_lag:], steps=1)[0]
    # Comptue the forecast
    if set_fcst_type == 0:
        if (fcst[0] > 0) | (fcst[1] > 0):
            z_f = 1.0
        elif (fcst[0] < 0) | (fcst[1] < 0):
            z_f = -1.0
    elif set_fcst_type == 1:
        z_f = np.sign(np.mean(fcst))
    elif set_fcst_type == 2:
        z_f = np.sign(fcst[0])
    #
    bnh = z.iloc[i+ini_wind, 0]
    stg = bnh*z_f
    store.iloc[i+ini_wind, :] = np.hstack([bnh, stg])

# Compute the cumulative return and plot
cret = ((store + 1).cumprod() - 1)*100
cret.plot(grid='both', title='The speculative system strategy for '+ticker+' using monthly returns', xlabel='Date', ylabel='return in percent')
plt.savefig(ticker+'.png')
plt.show()
#
print(cret.iloc[-1])