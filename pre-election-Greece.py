#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/05/21/economic-growth-and-relative-population-growth/
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

# Get the tickers with their explanations
t1 = 'POPTOTGRA647NWDB' # total population
t2 = 'SPPOP65UPTOZSGRC' # percent of total population over 65
t3 = 'SPPOP0014TOZSGRC' # percent of total population less than 14
t4 = 'RGDPNAGRA666NRUG' # real GDP at constant national prices, in 2017 US dollars
t5 = 'RKNANPGRA666NRUG' # capital stock at constant national prices, in 2017 US dollars

# Get the data
raw = pdr.fred.FredReader([t1, t2, t3, t4, t5], start='1960-01-10').read()
raw.columns = ['POP', 'POP > 65', 'POP < 14', 'RGDP', 'RKS']

# Assign to new dataset
data = raw

# Compute the young-to-old index
age_ratio = data['POP < 14']/data['POP > 65']
age_ratio.name = 'YOUNG/OLD'
data = pd.concat([data, age_ratio], axis=1)

# and plot it
ax = age_ratio.plot(grid='both', title='Ratio of young (less than 14 years) to old (greater than 65 years) \n (as percents of total population)', xlabel='Year', ylabel='Ratio')
ax.title.set_fontsize(10)
plt.show()

# Convert to log-growth rates
use_variables = ['RGDP', 'RKS', 'YOUNG/OLD']
data[use_variables] = data[use_variables].apply(np.log).diff().dropna()*100

# Lets plot the growth rates of the data one at a time # instructive to see how the details are made...
# fig, axes = plt.subplots(3,1)
# for i in range(data[use_variables].shape[1]):
#     xi = (data[use_variables].iloc[:,i])
#     if i <= 1:
#         set_xlabel = ''
#     else:
#         set_xlabel = 'Year'
#     set_title = 'Annual log-growth of '+xi.name
#     xi.plot(ax=axes.flatten()[i], title=set_title, xlabel=set_xlabel, ylabel='percent', grid='both')
#     axes.flatten()[i].title.set_size(10)
#     axes.flatten()[i].yaxis.label.set_size(10)
#     axes.flatten()[i].xaxis.label.set_size(10)
#     axes.flatten()[i].tick_params(labelsize=10)
# plt.show()

# Lets compute the cross-correlation (CC) between lagged economic growth and the growth of the young/old
store_cc = pd.DataFrame(data=None, index=range(16), columns=['CC-RGDP', 'CC-RKS'])

for i in range(16):
    df1 = pd.concat([data['YOUNG/OLD'], data['RGDP'].shift(periods=i)], axis=1).dropna()
    df2 = pd.concat([data['YOUNG/OLD'], data['RKS'].shift(periods=i)], axis=1).dropna()
    store_cc.iloc[i, 0] = df1.corr().iloc[1, 0]*100
    store_cc.iloc[i, 1] = df2.corr().iloc[1, 0]*100

# Plot the max CC for both series of growth
store_cc.plot(grid='both', title='Cross-correlation between YOUNG/OLD and economic growth', xlabel='Lag', ylabel='percent')
plt.show()

# Now plot the series aligned based on their max CC
#
# Prepare two data frames
max_lag = store_cc.apply(np.argmax)
plot_data1 = pd.concat([data['YOUNG/OLD'], data['RGDP'].shift(periods=max_lag[0])], axis=1).dropna()
plot_data1.columns = ['YOUNG/OLD', 'RGDP('+str(-max_lag[0])+')']
plot_data2 = pd.concat([data['YOUNG/OLD'], data['RKS'].shift(periods=max_lag[1])], axis=1).dropna()
plot_data2.columns = ['YOUNG/OLD', 'RKS('+str(-max_lag[1])+')']

# Plot them, and note the special usage for moving around the legend location!!
#
# First, for RGDP
#
ax1 = plot_data1.iloc[:,0].plot(label=plot_data1.columns[0])
ax2 = plot_data1.iloc[:,1].plot(label=plot_data1.columns[1], secondary_y=True)
#
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
#
plt.legend(h1+h2, l1+l2, loc='lower right')
plt.title('Young/Old growth vs. lagged real GDP growth')
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
plt.show()

# and then for RKS
#
ax1 = plot_data2.iloc[:,0].plot(label=plot_data2.columns[0])
ax2 = plot_data2.iloc[:,1].plot(label=plot_data2.columns[1], secondary_y=True)
#
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
#
plt.legend(h1+h2, l1+l2, loc='lower left')
plt.title('Young/Old growth vs. lagged real capital stock growth')
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
plt.show()

# Add a scatterplot for the association between lagged real GDP growth and relative population growth
plot_data1.plot(kind='scatter', x=1, y=0, s=np.abs(plot_data1.iloc[:,1])*20, color='red', grid='both', xlabel='Lagged real GDP growth', ylabel='Relative population growth', title='Young/Old growth vs. lagged real GDP growth')
plt.show()

# and finally one for the capital stock etc...
plot_data2.plot(kind='scatter', x=1, y=0, s=np.abs(plot_data2.iloc[:,1])*20, color='red', grid='both', xlabel='Lagged real capital stock growth', ylabel='Relative population growth', title='Young/Old growth vs. lagged real capital stock growth')
plt.show()
