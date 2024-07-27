#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/speculative-climate-change/
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
import yfinance as yf

# Read the static data and complete with a download
temp = pd.read_csv('global temperature data.csv', header=0, index_col=0, date_format='%Y%m')
sp500 = pd.read_csv('SP500-monthly-from-1928.csv', header=0, index_col=0, date_format='%m/%d/%Y')
sp500.name = 'SP500'
sp500_add = yf.download('^GSPC', start='2022-06-30', end='2023-12-31', interval='1mo')['Adj Close']
sp500_add.name = 'SP500'
sp500 = pd.concat([sp500, sp500_add], axis=0)
# Crop the temperature from 1928
temp = temp.loc['1928-01-01':]
temp.columns = ['Temp']
# Merge the indices
temp.index = sp500.index
# add industrial production
indpro = pdr.fred.FredReader('INDPRO', start='1928-01-01', end='2023-12-01').read()
indpro.index = sp500.index
# Put together
sp500_temp_indpro = pd.concat([sp500.apply(np.log), temp, indpro], axis=1)
# and crop from 1950
sp500_temp_indpro = sp500_temp_indpro.loc['1950-01-01':]
sp500_temp_indpro.to_csv('sp500_temp_indpro.csv')

# Rolling or expanding computation of the mean temperature?
do_rolling = True

# Loop over all starting years
for year in ['1950-01-01', '1960-01-01', '1970-01-01', '1975-01-01', '1980-01-01', '1985-01-01', '1990-01-01', '1995-01-01', '2000-01-01', '2005-01-01', '2010-01-01', '2015-01-01', '2020-01-01']:

    # Compute the returns and the signal variables
    if do_rolling:
        rets = pd.concat([sp500.pct_change(), temp, temp.rolling(window=12).mean(), temp.diff(), temp.diff().rolling(window=12).mean()], axis=1).dropna()
    else:
        rets = pd.concat([sp500.pct_change(), temp, temp.expanding().mean(), temp.diff(), temp.diff().expanding().mean()], axis=1).dropna()
    # Give nice column names
    rets.columns = ['SP500', 'Temp', 'Temp-Mean', 'Δ-Temp', 'Δ-Temp-Mean']

    rets = rets.loc[year:]
    print('******************************************************')
    print('             ')
    print('Year is='+year)

    # Compute the signals for delays of 1 to 24 months
    for i in np.arange(1, 25, 1):
        signal_temp = rets['Temp'].shift(periods=i).apply(np.sign)
        signal_temp_mean = rets['Temp-Mean'].shift(periods=i).apply(np.sign)
        signal_dtemp = rets['Δ-Temp'].shift(periods=i).apply(np.sign)
        signal_dtemp_mean = rets['Δ-Temp-Mean'].shift(periods=i).apply(np.sign)

        # Compute the strategies
        benchmark = rets['SP500']
        strategy1 = benchmark*signal_temp
        strategy2 = benchmark*signal_temp_mean
        strategy3 = benchmark*signal_dtemp
        strategy4 = benchmark*signal_dtemp_mean

        # Put together and compute the total returns
        both = pd.concat([benchmark, strategy1, strategy2, strategy3, strategy4], axis=1).dropna()
        tr = ((both+1).prod()-1)*100

        # If the delays is profitable then print the excess returns
        if (tr.iloc[1] > tr.iloc[0]):
            print('Evaluation starts in '+format(both.index[0], '%Y-%m'))
            print('----------------------------')
            print('Strategy #1, Temp        with delay =',i,'and ER = {0:6.2f}'.format(tr.iloc[1]-tr.iloc[0]))
        #
        if (tr.iloc[2] > tr.iloc[0]):
            print('             ')
            print('Evaluation starts in '+format(both.index[0], '%Y-%m'))
            print('----------------------------')
            print('Strategy #2, Temp-Mean   with delay =',i,'and ER = {0:6.2f}'.format(tr.iloc[2]-tr.iloc[0]))
        #
        if (tr.iloc[3] > tr.iloc[0]):
            print('             ')
            print('Evaluation starts in '+format(both.index[0], '%Y-%m'))
            print('----------------------------')
            print('Strategy #3, Δ-Temp      with delay =',i,'and ER = {0:6.2f}'.format(tr.iloc[3]-tr.iloc[0]))
        #
        if (tr.iloc[4] > tr.iloc[0]):
            print('             ')
            print('Evaluation starts in '+format(both.index[0], '%Y-%m'))
            print('----------------------------')
            print('Strategy #4, Δ-Temp-Mean with delay =',i,'and ER = {0:6.2f}'.format(tr.iloc[4]-tr.iloc[0]))
        #
    print('             ')
    print('******************************************************')
