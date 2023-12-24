#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/12/24/the-speculative-bayesian/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import yfinance as yf

# Define the trading mean
def trade_the_mean(y, rwind, alpha=1, beta=1):
    z = (y >= 0)
    sum = z.rolling(window=rwind).sum()
    if (alpha == 0.0) or (beta == 0.0):
        prior_mean = 0
    else:
        prior_mean = alpha/(alpha+beta)
    posterior_mean = (sum + alpha)/(alpha + beta + rwind)
    frc = (posterior_mean >= 0.5).apply(float) - (posterior_mean < 0.5).apply(float)
    ret = (frc.shift(periods=1))*y
    return ret.dropna(), frc.iloc[-1], prior_mean, posterior_mean.iloc[-1]

# Compute the relative wealth
def relative_wealth(theta, y, rwind):
    alpha = theta[0]
    beta = theta[1]
    ret, frc, prior_mean, posterior_mean = trade_the_mean(y, rwind, alpha, beta)
    crt = (ret+1).prod()
    brt = (y.loc[ret.index]+1).prod()
    return brt/crt, frc, prior_mean, posterior_mean

# Optimize the parameters by direct search
def optimize_params(y, roll_range=(2, 12), alpha_range=(1, 10), beta_range=(1, 10), step=1):
    wealth_old = 1
    set_roll = roll_range[0]
    set_alpha = alpha_range[0]
    set_beta = beta_range[0]
    set_frc = 1
    set_prior_mean = 0
    set_posterior_mean = 0
    for r in np.arange(roll_range[0], roll_range[1]+1, 1):
        for i in np.arange(alpha_range[0], alpha_range[1]+1, step):
            for j in np.arange(beta_range[0], beta_range[1]+1, step):
                wealth_new, frc, prior_mean, posterior_mean = relative_wealth((i, j), y, r)
                if wealth_new < 1 and wealth_new < wealth_old:
                    wealth_old = wealth_new
                    set_roll = r
                    set_alpha = i
                    set_beta = j
                    set_frc  = frc
                    set_prior_mean = prior_mean
                    set_posterior_mean = posterior_mean
                else:
                    if wealth_new < wealth_old:
                        wealth_old = wealth_new
                        set_roll = r
                        set_alpha = i
                        set_beta = j
                        set_frc  = frc
                        set_prior_mean = prior_mean
                        set_posterior_mean = posterior_mean
    #
    return set_roll, set_alpha, set_beta, set_frc, set_prior_mean, set_posterior_mean

#
# Get some data, monthly, full-sample in all cases, parametrizations below
#
# GLD, (3, 14), (1, 12), (1, 12), 2, 12 - sparse
#
# SPY, (3, 14), (1, 12), (1, 12), 2, 3 - sparse
# SPY, (3, 14), (1, 3), (1, 3), 1, 3 - dense
#
# DBC, (3, 14), (1, 12), (1, 12), 2, 3 - sparse
# DBC, (3, 14), (1, 3), (1, 3), 1, 3 - dense
#
# HYG, (3, 14), (1, 12), (1, 12), 2, 3 - sparse
# HYG, (3, 14), (1, 3), (1, 3), 1, 3 - dense
#
# TNA, (3, 14), (1, 3), (1, 3), 1, 3 - dense
#
# WEAT, (3, 14), (1, 3), (1, 3), 1, 12/24 - dense
#
# DBA, (3, 14), (1, 3), (1, 3), 1, 12/24 - dense
#
# UNG, (3, 14), (1, 3), (1, 3), 1, 12/24 - dense
#
# and some daily data, starting from 2023, parametrizations below
#
# WEAT, (2, 5), (1, 3), (1, 3), 1, 1 - dense
#
# DBC, (2, 5), (1, 3), (1, 3), 1, 1 - dense
#
# DBB, (2, 5), (1, 3), (1, 3), 1, 1 - dense
#
ticker = 'DBB'
data = yf.download(ticker, period='max', interval='1d')['Adj Close'].dropna().loc['2023-01-01':]
rets = data.pct_change().dropna()
nobs = rets.shape[0]

# Set the parameters, rolling windows, alpha range and beta range
r_range = (2, 5)
a_range = (1, 3)
b_range = (1, 3)
set_step = 1

# Set sparse or dense forecasting
set_sparse = False

# Set training interval
train_every = 1

# Set initial number of days
ini_wind = r_range[1]

# Initialize storage
store_ret = pd.DataFrame(data=None, index=rets.index, columns=[ticker, 'Speculative Mean', 'Speculative Bayesian', 'Average'])
store_means = pd.DataFrame(data=None, index=rets.index, columns=['prior mean', 'posterior mean'])

# Initialize wealth-based rotation probabilities
w0 = 0.5
w1 = 0.5

# and the evaluation loop
for i in np.arange(ini_wind, nobs, 1):

    # Section the data
    yi = rets.iloc[:i]

    # Select sparse or dense forecasting
    if set_sparse:
        # Train the parameters
        if i == ini_wind:
            # Train the two means, first the plain one
            set_roll0, set_alpha0, set_beta0, set_frc0, set_prior_mean0, set_posterior_mean0 = optimize_params(yi, r_range, (0, 0), (0, 0))
            # Then the lazy Bayesian one
            set_roll1, set_alpha1, set_beta1, set_frc1, set_prior_mean1, set_posterior_mean1 = optimize_params(yi, r_range, a_range, b_range, set_step)
        elif i%train_every == 0:
            # Train the two means, first the plain one
            set_roll0, set_alpha0, set_beta0, set_frc0, set_prior_mean0, set_posterior_mean0 = optimize_params(yi, r_range, (0, 0), (0, 0))
            # Then the lazy Bayesian one
            set_roll1, set_alpha1, set_beta1, set_frc1, set_prior_mean1, set_posterior_mean1 = optimize_params(yi, r_range, a_range, b_range, set_step)
    else:
        # Train the parameters
        if i == ini_wind:
            # Train the two means, first the plain one
            set_roll0, set_alpha0, set_beta0, set_frc0, set_prior_mean0, set_posterior_mean0 = optimize_params(yi, r_range, (0, 0), (0, 0))
            # Then the lazy Bayesian one
            set_roll1, set_alpha1, set_beta1, set_frc1, set_prior_mean1, set_posterior_mean1 = optimize_params(yi, r_range, a_range, b_range, set_step)
        elif i%train_every == 0:
            # Train the two means, first the plain one
            set_roll0, set_alpha0, set_beta0, set_frc0, set_prior_mean0, set_posterior_mean0 = optimize_params(yi, r_range, (0, 0), (0, 0))
            # Then the lazy Bayesian one
            set_roll1, set_alpha1, set_beta1, set_frc1, set_prior_mean1, set_posterior_mean1 = optimize_params(yi, r_range, a_range, b_range, set_step)
        else:
            # If dense forecasting then update the sign
            bret, set_frc0, set_prior_mean0, set_posterior_mean0 = trade_the_mean(yi, set_roll0, 0, 0)
            sret, set_frc1, set_prior_mean1, set_posterior_mean1 = trade_the_mean(yi, set_roll1, set_alpha1, set_beta1)

    # Trade in real-time
    yb = rets.iloc[i]
    y0 = yb*set_frc0
    y1 = yb*set_frc1
    y2 = yb*np.sign(w0*set_frc0 + w1*set_frc1)

    # Store
    store_ret.iloc[i] = np.hstack([yb, y0, y1, y2])
    store_means.iloc[i] = np.hstack([set_prior_mean1, set_posterior_mean1])

    # Compute wealth so far
    cret_i = (store_ret.iloc[:(i+1)]+1).dropna().prod()

    # Keep relative wealth for updating the rotation
    w0 = cret_i['Speculative Mean']
    w1 = cret_i['Speculative Bayesian']
    ws = w0+w1
    w0 = w0/ws
    w1 = w1/ws

    # Print progress
    if i%train_every == 0:
        print('Now training at i =', i, 'of', nobs)
        print('Wealth progress')
        print(cret_i)

# Done, compute cumulative returns and plot
store_ret = store_ret.dropna()
cret = ((store_ret+1).cumprod()-1)*100
print(cret.iloc[-1])
cret[[ticker, 'Speculative Mean', 'Speculative Bayesian']].plot(title='The speculative Bayesian strategy vs. the speculative mean strategy for '+ticker+', daily data', xlabel='Date', ylabel='return in percent', grid='both')
plt.show()