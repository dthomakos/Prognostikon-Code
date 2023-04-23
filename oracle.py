#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/04/22/the-oracle-as-risk-predictability/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from scipy import stats
from scipy import optimize as opt

# Get some data
data = yf.download('FXE', period='max', interval='1d').dropna()['Adj Close']
# Convert to log returns, get the right dates
rets = data.apply(np.log).diff().dropna()
rets = rets.loc['2022-01-01':'2023-03-31']

# Compute the oracle returns
ro = rets.apply(np.abs)
# and the oracle signs
so = rets.apply(np.sign)

# Fit the exponential distribution
out1 = stats.expon.fit(ro)

# Fit the cumulative oracle returns line
x = sm.add_constant(np.arange(0, ro.shape[0]))
out2 = sm.OLS(ro.cumsum(), x).fit()

# Perform a KS test of goodness of fit for the exponential distribution
out3 = stats.kstest(ro, 'expon', args=(out1))

# Write a function to find the parameters that would have maximized the p-value of the KS test
def min_expon_pvalue(theta, data, fix_mean=None):
    if fix_mean == None:
        mu = theta[0]
        sigma = theta[1]
    else:
        mu = fix_mean
        sigma = theta[0]
    out = stats.kstest(data, 'expon', args=(mu, sigma))
    return 1/out[1]

# and apply the function
set_fix_mean = True

if set_fix_mean:
    out4 = opt.minimize(min_expon_pvalue, out1[1], args=(ro, out1[0]), method='SLSQP', bounds=((0, None),))
    out5 = stats.kstest(ro, 'expon', args=(out1[0], out4.x[0]))
else:
    out4 = opt.minimize(min_expon_pvalue, out1, args=(ro,), method='SLSQP', bounds=((0, None), (0, None)))
    out5 = stats.kstest(ro, 'expon', args=(out4.x))

# Raw print results...
print(ro.mean())
print(out1)
print(out2.summary())
print(out3)
print(out4)
print(out5)