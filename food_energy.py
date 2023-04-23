#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/04/16/long-term-bonds-food-energy/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import packages
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as pdr

# Tickers and data extraction
tickers = ['IRLTLT01USM156N', 'PFOODINDEXM', 'PNRGINDEXM']
data = pdr.fred.FredReader(tickers, start='2013-01-1').read()
data.columns = ['10Y Yield', 'Food', 'Energy']

# Optionally plot the data
# data.plot(secondary_y='10Y Yield', grid='both')
# plt.show()

# Add the price of food deflated by the energy index
data = data.assign(Food_Energy = data['Food']/data['Energy'])
data.columns = ['10Y Yield', 'Food', 'Food', 'Food/Energy']

# Need only two series to plot
data_plot = data[['10Y Yield', 'Food/Energy']]

# and plot them...
data_plot.plot(color=['green', 'blue'], title='10Y US Yield vs. Global Food Index deflated by Global Energy Index', xlabel='Date', secondary_y='Food/Energy')
ax1, ax2 = plt.gcf().get_axes()
ax1.set_ylabel('percent')
ax2.set_ylabel('ratio of indexes')
plt.show()

