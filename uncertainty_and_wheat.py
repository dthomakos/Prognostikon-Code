#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/05/14/economic-policy-uncertainty-and-long-term-wheat-forecasting/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
import pandas as pd
import statsmodels.api as sm

# Get the global economic policy uncertainty and global price of wheat
data = pdr.fred.FredReader(['GEPUCURRENT', 'PWHEAMTUSDM'], start='1990-01-10').read()
data.columns = ['GEPU', 'Wheat']
# Keep last available value for wheat of April 2023
keep_last = data['Wheat'].iloc[-1]
# and drop NAs
data = data.dropna()

# Plot the original data
data.plot(title='Global Economic Policy Uncertainty & Global Price of Wheat',
xlabel='Date', ylabel='Index', grid='both', color=['blue', 'green'])
plt.show()

# Apply an xx-month moving average smoother, add to the data
smooth = 12
data['GEPU-MA('+str(smooth)+')'] = data['GEPU'].rolling(window=smooth).mean()
data['Wheat-MA('+str(smooth)+')'] = data['Wheat'].rolling(window=smooth).mean()
data = data.dropna()

# Using a simple loop find the lag with maximum cross-correlation (CC) between
# lagged uncertainty and current price of wheat
store_correlation = pd.DataFrame(data=None, index=range(37),
columns=['Raw Corr', 'Smoothed Corr'])
for i in range(37):
    # print("Lag order = ", i)
    c1 = data['Wheat'].corr(data['GEPU'].shift(periods=i))
    c2 = data['Wheat-MA('+str(smooth)+')'].corr(data['GEPU-MA('+str(smooth)+')'].shift(periods=i))
    # print("Correlations = ", c1, c2)
    store_correlation.iloc[i, :] = np.c_[c1, c2][0]

# Print the stored CC and then plot it
print(store_correlation)
store_correlation.plot(title='Cross-correlation between lagged GEPU and Wheat', xlabel='lead time', ylabel='correlation', grid='both')
plt.show()

# Find the lag with maximum cross-correlation (CC)
max_lag = store_correlation.apply(np.argmax, axis=0)

# Create two plots of the original and smoothed data aligned by the max CC
lag1 = max_lag[0]
new_data_raw = pd.concat([data['Wheat'], data['GEPU'].shift(periods=lag1)], axis=1).dropna()
new_data_raw.columns = ['Wheat', 'GEPU(-'+str(lag1)+')']
#
lag2 = max_lag[1]
new_data_smoothed = pd.concat([data['Wheat-MA('+str(smooth)+')'], data['GEPU-MA('+str(smooth)+')'].shift(periods=lag2)], axis=1).dropna()
new_data_smoothed.columns = ['Wheat-MA('+str(smooth)+')', 'GEPU(-'+str(lag2)+')-MA('+str(smooth)+')']

# Here you have the plot from a starting date of your choice: from 2014 they appear
# that the two series move in tandem...
set_start_date = '2014-01-01'
set_title = 'Global Economic Uncertainty leading Global Price of Wheat by '+str(lag1)+' months'
new_data_raw.loc[set_start_date:].plot(title=set_title, xlabel='Date', ylabel='Index', grid='both', color=['green', 'blue'])
plt.show()
#
new_data_smoothed.loc[set_start_date:].plot(title=set_title, xlabel='Date', ylabel='Index', grid='both', color=['green', 'blue'])
plt.show()

# Estimated a delay-based regression for the series in order to generate the
# forecast for wheat prices - careful to use the chosen start date!!
est_data_raw = new_data_raw.loc[set_start_date:]
x_raw = sm.add_constant(est_data_raw.iloc[:, 1])
out_raw = sm.OLS(est_data_raw.iloc[:, 0], x_raw).fit()
# Print the estimation results
print(out_raw.summary())

# OK, now let's make the prediction!
new_dates_raw = pd.date_range(start='2023-04-01', periods=lag1, freq='MS')
xfrc_raw = data['GEPU'].iloc[-lag1:]
xfrc_raw.index = new_dates_raw
# add the constant
xfrc_raw = sm.add_constant(xfrc_raw)
# and create the forecast
frc_raw = out_raw.predict(exog=xfrc_raw)

# Let's repeat for the smoothed series
est_data_smoothed = new_data_smoothed.loc[set_start_date:]
x_smoothed = sm.add_constant(est_data_smoothed.iloc[:, 1])
out_smoothed = sm.OLS(est_data_smoothed.iloc[:, 0], x_smoothed).fit()
# Print the estimation results
print(out_smoothed.summary())

# and the smoothed prediction
new_dates_smoothed = pd.date_range(start='2023-04-01', periods=lag1, freq='MS')
xfrc_smoothed = data['GEPU-MA('+str(smooth)+')'].iloc[-lag2:]
xfrc_smoothed.index = new_dates_smoothed
# add the constant
xfrc_smoothed = sm.add_constant(xfrc_smoothed)
# and create the forecast
frc_smoothed = out_smoothed.predict(exog=xfrc_smoothed)

# Make a nice plot of the actual vs. the forecasted values by filling in...
add_actual = pd.Series(data=np.nan, index=new_dates_raw)
actual = pd.concat([data['Wheat'].loc[set_start_date:], add_actual], axis=0)
actual.name = 'Wheat'
actual.loc[new_dates_raw[0]] = keep_last
#
add_frc_raw = pd.Series(data=np.nan, index=data.loc[set_start_date:].index)
forecast_raw = pd.concat([add_frc_raw, frc_raw], axis=0)
forecast_raw.name = 'Wheat Forecast'
#
add_frc_smoothed = pd.Series(data=np.nan, index=data.loc[set_start_date:].index)
forecast_smoothed = pd.concat([add_frc_smoothed, frc_smoothed], axis=0)
forecast_smoothed.name = 'Wheat Forecast (smoothed)'
#
all = pd.concat([actual, forecast_raw, forecast_smoothed], axis=1)

# Will plot the smoothed forecast - makes more sense for two years ahead...
ax = all[['Wheat', 'Wheat Forecast (smoothed)']].plot(title='Global Price of Wheat and Forecast up to March 2025', xlabel='Date', ylabel='Index', grid='both', color=['green', 'green'], style=['-', ':'])
ax.xaxis.grid(True, which='minor')
ax.yaxis.grid(True, which='minor')
plt.show()

# and with the raw data...
ax = all[['Wheat', 'Wheat Forecast']].plot(title='Global Price of Wheat and Forecast up to March 2025', xlabel='Date', ylabel='Index', grid='both', color=['green', 'green'], style=['-', ':'])
ax.xaxis.grid(True, which='minor')
ax.yaxis.grid(True, which='minor')
plt.show()