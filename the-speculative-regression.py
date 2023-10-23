#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/10/24/the-speculative-regression/
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

# Linear model, least squares or robust, from statsmodels with sequential eliminination based on p-values
def sequential_elimination_lm(set_Y, set_X, set_alpha, robust=False):
    if robust:
        out = sm.RLM(endog=set_Y, exog=set_X).fit()
    else:
        out = sm.OLS(endog=set_Y, exog=set_X, hasconst=True).fit()
    pv_old = out.pvalues
    ip_old = pv_old[pv_old <= set_alpha[0]].index

    # and with a simple loop remove the rest in the proper way with diminishing p-values
    for aa in np.arange(1, len(set_alpha)):
        xa = set_X[ip_old]
        ya = set_Y
        if robust:
            out = sm.RLM(endog=set_Y, exog=xa).fit()
        else:
            out = sm.OLS(endog=ya, exog=xa, hasconst=True).fit()
        pv_new = out.pvalues
        ip_new = pv_new[pv_new <= set_alpha[aa]].index
        if len(ip_new) > 0:
            pv_old = pv_new
            ip_old = ip_new

    # and this is the final model
    xa = set_X[ip_old]
    ya = set_Y
    out = sm.OLS(endog=ya, exog=xa, hasconst=True).fit()

    # Done!
    return out

# A simple regression predictor based on the above function and the data structure of the post
def srp(data, alpha, robust):
    y = data.iloc[:, 0]
    x = sm.add_constant(data.iloc[:, 1:])
    model = sequential_elimination_lm(y.iloc[1:], x.shift(periods=1).iloc[1:], alpha, robust)
    beta = model.params
    xfor = x.iloc[-1]
    fcst = (beta.mul(xfor)).sum()
    return np.sign(fcst)

# Download some data, for the post I used SSO, USO, DBB, LQD, WEAT, IYR - please see the note on the
# setting of the diminishing p-values later in the code!!
ticker = 'IYR'
data = yf.download(ticker, period='max', interval='1d')['Adj Close'].dropna()
r = data.pct_change().dropna().loc['2022-01-01':] # you can change this of course
r.name = ticker

# Compute the necessary variables
dr = r.diff()
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
z_all = pd.concat([r, dr, z1p, z2p, z1n, z2n], axis=1).dropna()
z_all.columns = [ticker, 'D-'+ticker, 'Z1+', 'Z2+', 'Z1-', 'Z2-']

# Number of observations and initial window
nobs = z_all.shape[0]
ini_wind = 21

# Select a sequence of p-values; this can be changed or fine-tuned
set_alpha = [0.9, 0.7, 0.5, 0.25, 0.125, 0.0625] # used the full sequence for WEAT & IYR only, else used [0.9, 0.7]

# Use robust estimation?
use_robust = False

# and initialize storage
store = pd.DataFrame(data=None, index=z_all.index, columns=[ticker, 'Speculative Regression'])

# Run a simple loop to get the signals and the strategy returns
for i in np.arange(0, nobs-ini_wind, 1):
    z_i = z_all.iloc[:(i+ini_wind),:] # this is recursive estimation, change to i:(i+ini_wind) for rolling!!
    z_f = srp(z_i, set_alpha, use_robust)
    bnh = z_all.iloc[i+ini_wind, 0]
    stg = bnh*z_f
    store.iloc[i+ini_wind, :] = np.hstack([bnh, stg])

# Compute the cumulative return and plot
cret = ((store + 1).cumprod() - 1)*100
cret.plot(grid='both', title='The speculative regression strategy for '+ticker+' using daily returns', xlabel='Date', ylabel='return in percent')
plt.savefig(ticker+'.png')
plt.show()
#
print(cret.iloc[-1])
print(store.mean(axis=0)/store.std(axis=0))

#SSO                       -23.656021
#Speculative Regression    122.367302
#USO                        26.183619
#Speculative Regression    140.869522
#DBB                      -21.432536
#Speculative Regression    34.503916
#LQD                      -17.359407
#Speculative Regression    33.838079
#WEAT                     -18.844565
#Speculative Regression    49.009478
#IYR                      -27.650459
#Speculative Regression     17.98597
