#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/04/27/output-capacity-food-energy/
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
import yfinance as yahoo
from scipy import optimize as opt
from statsmodels.tsa.stattools import ccf

# Tickers, start date, end date and data extraction
tickers = ['INDPRO', 'MCUMFN', 'CPIUFDSL', 'CPIENGSL']
start_date = '2012-03-01'
end_date = '2023-03-31'
econdata = pdr.fred.FredReader(tickers, start=start_date, end=end_date).read()
econdata.columns = ['Output', 'Capacity', 'Food', 'Energy']

# Get the two ratios
econdata = econdata.assign(Index1 = econdata['Food']/econdata['Energy'],
Index2 = econdata['Output']/econdata['Capacity'])
econdata.columns = ['Output', 'Capacity', 'Food', 'Energy', 'Food/Energy', 'Output/Capacity']

# Do the plot
indices = econdata[['Output/Capacity', 'Food/Energy']].apply(np.log).diff(periods=12).dropna()*100
indices.plot(secondary_y='Food/Energy', color=['blue', 'green'], xlabel='Year',
title='Output-to-Capacity ratio vs. Food-to-Energy ratio, US monthly data')
ax1, ax2 = plt.gcf().get_axes()
ax1.set_ylabel('annual log-growth, percent')
ax2.set_ylabel('annual log-growth, percent')
plt.show()

# and compute the correlation
print(indices.corr())

