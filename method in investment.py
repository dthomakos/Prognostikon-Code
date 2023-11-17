#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/11/17/method-in-investment/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# A function to compute the maximum drawdown
def max_dd(wealth):
    maxwealth = wealth.cummax(axis=0)
    drawdowns = wealth/maxwealth - 1
    return drawdowns.min(axis=0)

# Select a ticker to analyze, always weekly frequency
ticker = 'QQQ'
data = yf.download(ticker, period='max', interval='1wk')['Adj Close'].dropna()
rets = data.pct_change().dropna().loc['2019-01-01':]

# Get the 3- and 10-week rolling momentum
m3 = rets.rolling(window=3).apply(lambda x: (x+1).prod()-1)
m10= rets.rolling(window=10).apply(lambda x: (x+1).prod()-1)
# Put data together
all = pd.concat([rets, m3, m10], axis=1).dropna()
all.columns = [ticker, ticker+'-M3', ticker+'+M10']
# Get number of observations
nobs = all.shape[0]

# Set parameters
alpha = 0.02
alpha2= alpha/2
#
iniK = 100
K0 = iniK*(1+all.iloc[0,0])

# Initialize storage
store = pd.DataFrame(data=None, index=all.index, columns=['The Method', ticker, 'AddFunds', 'TakeProfit'])
store.iloc[0,:] = np.hstack([K0, K0, 0, 0])
add = 0

# Loop over the observations
for i in range(nobs-1):
    # First, check for taking profit at the 6% return over a month
    if i > 3:
        r_i = (store.iloc[i,0]/store.iloc[i-3,0])-1
        if r_i >= 0.06 and K0 > 0:
            # Take half-of-alpha for profit
            TakeProfit = alpha2*K0
            # Redefine available capital
            K0 = K0 - TakeProfit
            # Store profit taken
            store.iloc[i, 3] = TakeProfit
        else:
            TakeProfit = 0
            store.iloc[i, 3] = TakeProfit

    # Next, check for adding capital when momentum is negative
    x_i = all.iloc[i,:]
    # This happens as an OR
    if x_i.iloc[1] < -0.03 or x_i.iloc[2] < -0.1:
        # Compute total profit from inception
        TotalProfit = K0 - iniK
        # If the profit is positive then add alpha fraction of new funds
        if TotalProfit > 0:
            K1 = ((1+alpha)*K0 - alpha*iniK)*(1+all.iloc[i+1,0])
            add = (K0 - iniK)*alpha
        else:
            K1 = K0*(1+all.iloc[i+1,0])
            add = 0
    # and when momentum is positive, short the market
    elif x_i.iloc[1] > 0.03 and x_i.iloc[2] > 0.1:
        # Compute total profit from inception
        TotalProfit = K0 - iniK
        # If the profit is positive then short the market
        if TotalProfit > 0:
            K1 = K0*(1-alpha*all.iloc[i+1,0])
        else:
            K1 = K0*(1+all.iloc[i+1,0])
            add = 0
    else:
        K1 = K0*(1+all.iloc[i+1,0])
        add = 0

    # Compute the benchmark return, store and move on
    benchK = store.iloc[i,1]*(1+all.iloc[i+1,0])
    store.iloc[i+1,:3] = np.hstack([K1, benchK, add])
    K0 = K1

# Lets compute the maximum drawdown
wealth = store[['The Method', ticker]]
maxDD = wealth.apply(max_dd)

# Do a performance plot
wealth.plot(title='Method-in-Investment strategy for '+ticker, xlabel='Date',
ylabel='Value of investment of '+str(iniK)+'$', grid='both')
plt.show()

# and another plot with the funds added and take profit taken
in_and_out = store[['AddFunds', 'TakeProfit']]
in_and_out.cumsum().plot(title='Method-in-Investment deposits & withdrawals for '+ticker, xlabel='Date',
ylabel='Value in $', grid='both')
plt.show()

# Compute the annual deposits and withdrawals
annual = in_and_out.groupby(by=in_and_out.index.year).sum()

# Print everything
print('TIcker:', ticker)
print('Total wealth of strategy  for initial investment of '+str(iniK)+'$ = ', wealth.iloc[-1, 0])
print('Total wealth of benchmark for initial investment of '+str(iniK)+'$ = ', wealth.iloc[-1, 1])
print('Maximum drawdown for strategy = ', maxDD[0])
print('Maximum drawdown for benchmark = ', maxDD[1])
#
print(annual)
print(annual.median())
print(in_and_out.cumsum())
