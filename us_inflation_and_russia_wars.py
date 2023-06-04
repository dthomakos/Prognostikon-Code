#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/06/04/inflation-and-war/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas_datareader as pdr
import pandas as pd

# Get the data
data = pdr.fred.FredReader('CPIAUCSL', start='2006-01-01', end='2023-04-30').read()
infl = data.apply(np.log).diff(periods=12).dropna()*100

# Mark the 12-months prior and 1+12-months after the event, with center in the month after,
# and scale with the post-event inflation...dates are from the associated Wikipedia pages
georgia_war = infl.loc['2007-09-01':'2009-09-01']/infl.loc['2008-09-01']
georgia_war.index = np.arange(-12, 13, 1)
#
crimean_war = infl.loc['2013-04-01':'2015-04-01']/infl.loc['2014-04-01']
crimean_war.index = np.arange(-12, 13, 1)
#
ukraine_war = infl.loc['2021-03-01':'2023-03-01']/infl.loc['2022-03-01']
ukraine_war.index = np.arange(-12, 13, 1)

# Put together, with new index
all = pd.concat([georgia_war, crimean_war, ukraine_war], axis=1)*100
all.columns = ['Georgia War', 'Crimean War', 'Ukraine War']

# Now for the plots...first the whole period - note the split of columns for doing the
# legends and axes labels right!
ax1 = all[['Georgia War', 'Crimean War']].plot(title='Pre- and post-event US inflation vs. Russian Wars', xlabel='Months', ylabel='Inflation Index', color=['black', 'blue'])
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
all[['Ukraine War']].plot(ax=ax2, color=['red'], ylabel='Inflation Index')
ax2.legend(loc='upper right')
plt.show()

# and then at and after the event
ax1 = all.loc[0:,['Georgia War', 'Crimean War']].plot(title='Post-event US inflation vs. Russian Wars', xlabel='Months', ylabel='Inflation Index', color=['black', 'blue'])
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
ax1.legend(loc='lower left')
ax2 = ax1.twinx()
all.loc[0:,['Ukraine War']].plot(ax=ax2,color=['red'], ylabel='Inflation Index')
ax2.legend(loc='upper right')
plt.show()

# and let us repeat this for global food inflation, copy/paste and adapt from above
data = pdr.fred.FredReader('PFOODINDEXM', start='2006-01-01', end='2023-04-30').read()
infl = data.apply(np.log).diff(periods=12).dropna()*100
#
georgia_war = infl.loc['2007-09-01':'2009-09-01']/infl.loc['2008-09-01']
georgia_war.index = np.arange(-12, 13, 1)
#
crimean_war = infl.loc['2013-04-01':'2015-04-01']/infl.loc['2014-04-01']
crimean_war.index = np.arange(-12, 13, 1)
#
ukraine_war = infl.loc['2021-03-01':'2023-03-01']/infl.loc['2022-03-01']
ukraine_war.index = np.arange(-12, 13, 1)
# Put together, with new index
all = pd.concat([georgia_war, crimean_war, ukraine_war], axis=1)*100
all.columns = ['Georgia War', 'Crimean War', 'Ukraine War']

# Now for the plots...use only Georgia and Ukraine wars...and post event only
ax1 = all.loc[0:,['Georgia War']].plot(title='Post-event global food inflation vs. Russian Wars', xlabel='Months', ylabel='Food Inflation Index', color=['black'])
ax1.xaxis.grid(True, which='major')
ax1.yaxis.grid(True, which='major')
ax1.legend(loc='lower left')
ax2 = ax1.twinx()
all.loc[0:,['Ukraine War']].plot(ax=ax2,color=['red'], ylabel='Food Inflation Index')
ax2.legend(loc='upper right')
plt.show()
