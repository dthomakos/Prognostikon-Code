#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/05/06/peaks-and-troughs-forecasting-us-real-gdp-growth/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from dateutil.relativedelta import relativedelta

# Get the real GDP
realgdp = pdr.fred.FredReader('GDPC1', start='1947-01-01', end='2023-07-01').read()
# Compute the annual growth rate
growthrate = realgdp.apply(np.log).diff(periods=4).dropna()*100
# Give nice column names
growthrate.columns = ['Real GDP Growth Rate']

# Given the most recent highest peak find the nearest neighbors (NN)
set_peak = 8
# Fix target and training sets
set_last = growthrate.iloc[-set_peak:]
set_rest = growthrate.iloc[:-set_peak]

# Compute the magnitude and positions of the NN
mod = set_rest.shape[0]%set_peak
set_rest = set_rest.iloc[mod:]
set_M = set_rest.shape[0]-set_peak+1
store_dates = pd.DataFrame(data=np.zeros([set_peak, set_M]))
store_values = pd.DataFrame(data=np.zeros([set_peak, set_M]))
store_distances = pd.DataFrame(data=np.zeros([set_M, 1]), columns=['Distances'])

# Carefull in the NN - the first position must be a peak!!
for i in range(set_rest.shape[0]-set_peak+1):
    xi = set_rest.iloc[i:(i+set_peak)]
    store_dates.iloc[:,i] = xi.index
    store_values.iloc[:, i] = xi.to_numpy()
    test = (xi.iloc[0] > xi.iloc[1:]).all().to_numpy()[0]
    if test:
        dist = np.sqrt(((xi.to_numpy()[0] - set_last.to_numpy()[0]) ** 2).sum())
    else:
        dist = 999
    store_distances.iloc[i,:] = dist

# Now, get the dates and values for the ordered distances
idx = store_distances.sort_values(by='Distances').index
store_dates = store_dates.loc[:,idx]
store_values = store_values.loc[:, idx]

# By inspection now, select the first 3 NN by magnitude of their first value
# and by non-overlapping periods: [6, 40, 139, 67, 95] and [6, 40, 95]
look_back = [6, 40, 139, 67, 95]
look_back = [6, 40, 95]
frc1 = pd.DataFrame(data=np.zeros([2*set_peak, len(look_back)]), columns=look_back)

# For each period we must also find the next set_peak values for the forecast
for j in look_back:
    idx = store_dates.loc[:,j]
    frc1.loc[:,j] = growthrate.loc[idx[0]:(idx[set_peak-1]+relativedelta(months=3*set_peak))].to_numpy()

# Compute the mean over the NN
mean1 = frc1.mean(axis=1)
# and the standard deviation of this mean
std = frc1.std(axis=1)/np.sqrt(len(look_back))
# bounds...
mean1_lb = mean1 - 2*std
mean1_ub = mean1 + 2*std
mean1_all = pd.concat([mean1_lb, mean1, mean1_ub], axis=1)

# Let us add a standard NN forecast for comparison based on the last observation
last_obs = growthrate.iloc[-1]
dist = np.sqrt(((growthrate - growthrate.iloc[-1])**2)).sort_values(by='Real GDP Growth Rate').iloc[1:]
# select number of NN to use
set_NN = 2
frc2 = pd.DataFrame(data=np.zeros([set_peak, set_NN]), columns=range(set_NN))
for j in range(set_NN):
    frc2.iloc[:,j] = growthrate.loc[dist.index[j]+relativedelta(months=3):(dist.index[j]+relativedelta(months=3*set_peak+1))].to_numpy()
# get the forecast and bounds
mean2 = frc2.mean(axis=1)
std = frc2.std(axis=1)/np.sqrt(set_NN)
mean2_lb = mean2 - 2*std
mean2_ub = mean2 + 2*std
mean2_all = pd.concat([mean2_lb, mean2, mean2_ub], axis=1)

# Done, fix nicely for a plot
actual = pd.DataFrame(data=np.vstack([set_last.to_numpy(), np.repeat(np.nan, set_peak).reshape(-1, 1)]),
index=pd.date_range(start='2021-04-01', periods=2*set_peak, freq='Q-DEC'))
mean1_all.iloc[:set_peak] = np.nan
mean2_all = pd.concat([pd.DataFrame(np.repeat(np.nan, 3*set_peak).reshape(set_peak, 3), columns=range(3)), mean2_all], axis=0)
mean1_all.index = actual.index
mean2_all.index = actual.index
#
actual_frc1 = pd.concat([actual, mean1_all], axis=1)
actual_frc1.columns = ['Real GDP growth rate', 'NNP-forecast lower bound',
'NNP-forecast', 'NNP-forecast upper bound']
actual_frc2 = pd.concat([actual, mean2_all], axis=1)
actual_frc2.columns = ['Real GDP growth rate', 'NN-forecast lower bound',
'NN-forecast', 'NN-forecast upper bound']
actual_frc = pd.concat([actual, mean1_all.iloc[:,1], mean2_all.iloc[:,1]], axis=1)
actual_frc.columns = ['Real GDP growth rate', 'NNP-forecast', 'NN-forecast']
#
ax1 = actual_frc1.plot(grid='both', color=['black', 'red', 'blue', 'red'], style=['-', ':', '--', ':'],
title='US real GDP growth rate and NN peak-based forecast with 95% bounds using '+str(len(look_back))+' NN',
figsize=[13, 8], xlabel='Date', ylabel='percent')
# plt.axvline(actual_frc1.index[set_peak-1],color='black', linestyle=':')
ax1.xaxis.grid(True, which='minor')
ax1.yaxis.grid(True, which='minor')
plt.show()
#
ax2 = actual_frc2.plot(grid='both', color=['black', 'red', 'blue', 'red'], style=['-', ':', '--', ':'],
title='US real GDP growth rate and standard NN forecast with 95% bounds using '+str(set_NN)+' NN',
figsize=[13, 8], xlabel='Date', ylabel='percent')
ax2.xaxis.grid(True, which='minor')
ax2.yaxis.grid(True, which='minor')
plt.show()
#
# for this last plot insert the last actual value in the forecasts to look nicer
actual_frc.loc['2023-03-31',['NNP-forecast', 'NN-forecast']] = actual_frc.loc['2023-03-31', 'Real GDP growth rate']
ax3 = actual_frc.plot(grid='both', color=['black', 'blue', 'green'], style=['-', '--', ':'],
title='US real GDP growth rate and both NN-type forecasts using '+str(len(look_back))+' NNP and '+str(set_NN)+' NN', figsize=[13, 8], xlabel='Date', ylabel='percent')
ax3.xaxis.grid(True, which='minor')
ax3.yaxis.grid(True, which='minor')
plt.show()