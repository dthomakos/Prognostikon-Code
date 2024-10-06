#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/money-demand-and-inflation-forecasting/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

# Get the data and starting date
tickers = ['CPIAUCSL', 'CURRSL', 'DEMDEPSL', 'DPSACBM027SBOG']
raw_data = pdr.fred.FredReader(tickers, start='1992-01-01', end='2024-10-01').read()
data = (raw_data/raw_data.iloc[0])
start_from = '2017-01-01'

# Select the deposit measure
deposits = 'DEMDEPSL'
#
if deposits == 'DEMDEPSL':
    deposit_type = 'Demand'
else:
    deposit_type = 'Total'
yy = data['CPIAUCSL']
xx = data['CURRSL']/data[deposits]
y = yy.pct_change(periods=12).dropna()*100
x = xx.pct_change(periods=12).dropna()*100
xx = pd.concat([y, x], axis=1)
xx.columns = ['Inflation', 'Currency-to-'+deposit_type+' Deposits']
xx = xx.dropna()
# Crop the data
xx = xx.loc[start_from:]

# Prepare to find the cross-correlations
store_ccor_inf = pd.DataFrame(data=None, index=np.arange(0, 25, 1), columns=['Cross-Correlation'])
#
for i in np.arange(0, 25, 1):
    store_ccor_inf.iloc[i, 0] = xx['Inflation'].corr(xx['Currency-to-'+deposit_type+' Deposits'].shift(periods=i))*100
# plot the lagged-cross correlations
store_ccor_inf.plot(title='Cross-correlations of lagged Currency-to-'+deposit_type+' Deposits and inflation', xlabel='Lag', ylabel='Correlation', grid='both')
plt.show()

# extract max cross-correlation
max_ccor_inf = store_ccor_inf.apply(lambda x: np.argmax(np.abs(x)), axis=0)

# Estimate a delay regression - make sure to use the xx dataframe
yt = xx['Inflation'].iloc[max_ccor_inf.iloc[0]:]
xt = sm.add_constant(xx['Currency-to-'+deposit_type+' Deposits'].shift(periods=max_ccor_inf.iloc[0])).dropna()
mod = sm.OLS(endog=yt, exog=xt).fit()
xf = sm.add_constant(xx['Currency-to-'+deposit_type+' Deposits'].iloc[-max_ccor_inf.iloc[0]:])
frc = mod.predict(exog=xf)
# Print the estimated delay regression and corresponding forecasts
print(mod.summary())
print(frc)

# Expand the sample, required for the dynamic part of the forecast and the plotting
extra = pd.date_range(start='2024-09-01', periods=max_ccor_inf.iloc[0], freq='MS')
extra_sample = pd.date_range(start=xx.index[0], end=extra[-1], freq='MS')

# You can optionally adjust the forecasts for the dynamic term - I did this
# in the most simplistic of ways, by an AR(1) model on the residuals of the
# delay regression above
do_dynamic = False
if do_dynamic:
    res = AutoReg(endog=mod.resid, lags=1).fit()
    res_frc = res.predict(start='2024-09-01', end=extra.to_flat_index()[-1])
    if do_dynamic:
        frc.index = res_frc.index
        frc = frc + res_frc

# Next plot the aligned series plus the corresponding forecasts
#
# Expand the sample
extra = pd.date_range(start='2024-09-01', periods=max_ccor_inf.iloc[0], freq='MS')
extra_sample = pd.date_range(start=xx.index[0], end=extra[-1], freq='MS')
z = pd.DataFrame(data=None, index=extra_sample, columns=['Inflation', 'Currency-to-'+deposit_type+' Deposits'])
z['Inflation'] = xx['Inflation']
z['Currency-to-'+deposit_type+' Deposits'] = xx['Currency-to-'+deposit_type+' Deposits']
z['Currency-to-'+deposit_type+' Deposits'] = z['Currency-to-'+deposit_type+' Deposits'].shift(periods=max_ccor_inf.iloc[0])
z = z.iloc[max_ccor_inf.iloc[0]:]
z.loc['2024-09-01':,'Inflation'] = frc.to_numpy()
#
ax1 = z['Currency-to-'+deposit_type+' Deposits'].plot(title='Inflation and Currency-to-'+deposit_type+' Deposits lagged '+str(max_ccor_inf.iloc[0])+' months', xlabel='Date', ylabel='Currency-to-'+deposit_type+' Deposits', color=['green'])
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
ax1.legend(loc='lower left')
ax2 = ax1.twinx()
z['Inflation'].plot(ax=ax2,color=['blue'], ylabel='Inflation')
ax2.legend(loc='lower right')
plt.show()

