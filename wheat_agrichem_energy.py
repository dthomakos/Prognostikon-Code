#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/09/04/trading-wheat-inflation/
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

# Download the data from FRED
data = pdr.fred.FredReader(['DNRGRC1M027SBEA', 'PCU32533253', 'WPU0121'], start='1990-01-01').read()#
data.columns = ['Energy', 'AgriChem', 'Wheat']

# Select whether to use returns (True) or levels (False)
do_pct_change = True

# Prepare the data and the parametrization accordingly - NOTE: changing the rolling window, max lag and starting date
# will produce different results; you have to experiment with all of them!!!
if do_pct_change:
    data['AgriChem-Energy'] = (data['AgriChem']/data['Energy']).pct_change()
    roll = 24
    max_lag = 12
    data = data.loc['2005-01-01':]
else:
    data['AgriChem-Energy'] = (data['AgriChem']/data['Energy'])
    roll = 12
    max_lag = 6
    data = data.loc['2007-01-01':]
# Get sample size and the benchmark wheat inflation
nobs = data.shape[0]
data['Wheat-Returns'] = data['Wheat'].pct_change()

# Initialize storage
rets = pd.DataFrame(data=None, index=data.index, columns = ['Benchmark', 'AgriChem-Energy'])

# Do the rolling window loop for trading
for i in np.arange(0, nobs-roll, 1):
    # Crop the data
    use_data = data[['Wheat', 'AgriChem-Energy', 'Wheat-Returns']].iloc[i:i+roll]

    # Compute the cross-correlations
    cross_corr = pd.DataFrame(data=None, index=np.arange(0, max_lag+1, 1), columns=['Wheat-AgriChem-Energy'])
    for s in np.arange(0, max_lag+1, 1):
        if do_pct_change:
            cross_corr.iloc[s, 0] = use_data['Wheat-Returns'].corr(use_data['AgriChem-Energy'].shift(periods=s))
        else:
            cross_corr.iloc[s, 0] = use_data['Wheat'].corr(use_data['AgriChem-Energy'].shift(periods=s))

    # Find the maximum absolute cross-correlation
    imax = cross_corr.apply(np.abs).apply(np.argmax, axis=0).iloc[0]
    smax = cross_corr.iloc[imax, 0]

    # Trade with the sign of the cross-correlation times the sign of the data at the optimal lag
    # (note that for levels the data are always positive so you trade the sign of smax essentially)
    tmax = np.sign(use_data['AgriChem-Energy'].iloc[-imax+1]*smax)

    # Find the next value and evaluate your trade
    nextr = data['Wheat-Returns'].iloc[roll+i]
    rets.iloc[roll+i] = np.c_[nextr, nextr*tmax]

# Drop the missing values from the dataframe of the strategy
rets = rets.dropna()
crets = ((rets+1).cumprod()-1)*100

# and plot...
if do_pct_change:
    crets.plot(title='Total Return from Trading Wheat Inflation with AgriChem & Energy \n (trading signals are from growth rates)', ylabel='return in percent', xlabel='Time', color=['red', 'green'], grid='both', fontsize=12)
else:
    crets.plot(title='Total Return from Trading Wheat Inflation with AgriChem & Energy \n (trading signals are from levels)', ylabel='return in percent', xlabel='Time', color=['red', 'green'], grid='both', fontsize=12)
#
plt.show()
