#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/10/15/tcb-with-automatic-window-length/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#


# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import yfinance as yf

# Define the function that finds the best window according to type and nroll
def automatic_window_length(x, nroll, type):
    y = x.rank()
    z = y == y.max()
    out = nroll[z]
    if len(out) == 0:
        return np.nan
    else:
        if type == 'min':
            return int(np.min(out))
        elif type == 'median':
            return int(np.median(out))
        elif type == 'mean':
            return int(np.mean(out))
        elif type == 'max':
            return int(np.max(out))

# Get some data, for the post I used the tickers below
tickers = ['EEM', 'DBA', 'DBC', 'USO', 'FXE']

# Forecasts are based on discrete or continuous data?
use_signs = True

# Set the rolling windows
set_R1 = 2
set_RM = 18
nroll = np.arange(set_R1, set_RM+1, 1)

# Set the rolling period for the computation of cumulative returns, or to -1 to use recursive computation
set_roll_wealth = -1

# Then, a loop over the tickers
for ticker in tickers:
    # Download the data, using monthly frequency below and starting from 2013
    data = yf.download(ticker, period='max', interval='1mo')['Adj Close'].dropna()
    r = data.pct_change().dropna().loc['2013-01-01':] # you can change this of course
    r.name = ticker
    y = r
    if use_signs:
        y = np.sign(r)

    # Nobs
    nobs = y.shape[0]

    # Initialize storage
    forecasts = pd.DataFrame(data=None, index=y.index, columns=nroll)
    rolreturn = pd.DataFrame(data=None, index=y.index, columns=nroll)

    # Loop over the rolling windows, the forecast is the plain sample mean
    for i in nroll:
        frc = y.rolling(window=i).mean().apply(np.sign)
        ret = r*(frc.shift(periods=1))
        if set_roll_wealth > 0:
            rrt = (ret + 1).rolling(window=set_roll_wealth).apply(np.prod) - 1
        else:
            rrt = (ret + 1).cumprod() - 1
        forecasts[i] = frc
        rolreturn[i] = rrt

    # Compute the combinations
    c_min = rolreturn.apply(automatic_window_length, args=(nroll, 'min'), axis=1)
    c_mdn = rolreturn.apply(automatic_window_length, args=(nroll, 'median'), axis=1)
    c_mnn = rolreturn.apply(automatic_window_length, args=(nroll, 'mean'), axis=1)
    c_max = rolreturn.apply(automatic_window_length, args=(nroll, 'max'), axis=1)

    # Based on the combination select the appropriate forecast each time
    new_forecasts = pd.DataFrame(data=None, index=y.index,
    columns=['AWL-min', 'AWL-median', 'AWL-mean', 'AWL-max'])
    for i in range(nobs):
        c_min_i = c_min.iloc[i]
        c_mdn_i = c_mdn.iloc[i]
        c_mnn_i = c_mnn.iloc[i]
        c_max_i = c_max.iloc[i]
        #
        c_min_i_check = np.isnan(c_min_i)
        c_mdn_i_check = np.isnan(c_mdn_i)
        c_mnn_i_check = np.isnan(c_mnn_i)
        c_max_i_check = np.isnan(c_max_i)
        #
        if not c_min_i_check:
            new_forecasts.iloc[i, 0] = forecasts.iloc[i, int(c_min_i-set_R1)]
        if not c_mdn_i_check:
            new_forecasts.iloc[i, 1] = forecasts.iloc[i, int(c_mdn_i-set_R1)]
        if not c_mnn_i_check:
            new_forecasts.iloc[i, 2] = forecasts.iloc[i, int(c_mnn_i-set_R1)]
        if not c_max_i_check:
            new_forecasts.iloc[i, 3] = forecasts.iloc[i, int(c_max_i-set_R1)]

    # and compute the new returns
    new_r = pd.DataFrame(data=np.repeat(r.to_numpy(), 4).reshape(-1, 4), index=r.index,
    columns=['AWL-min', 'AWL-median', 'AWL-mean', 'AWL-max'])
    new_ret = new_r*(new_forecasts.shift(periods=1))
    if set_roll_wealth > 0:
        new_rolreturn = (new_ret + 1).rolling(window=set_roll_wealth).apply(np.prod) - 1
    else:
        new_rolreturn = (new_ret + 1).cumprod() - 1

    # and the benchmark
    if set_roll_wealth > 0:
        bench = (r + 1).rolling(window=set_roll_wealth).apply(np.prod) - 1
    else:
        bench = (r + 1).cumprod() - 1

    # Collect the results, remove impact of maximum rolling window
    all = pd.concat([bench, new_rolreturn], axis=1).dropna()*100

    # Do the plot
    all[[ticker, 'AWL-min']].plot(title='Total trading returns of the AWL-min strategy for '+ticker, ylabel='return in percent')
    plt.grid(visible=True, which='both')
    plt.show()

    # and a plot of the AWL windows selected
    cmb_all = pd.concat([c_min, c_mdn, c_mnn, c_max], axis=1)
    cmb_all.columns = ['AWL-min', 'AWL-median', 'AWL-mean', 'AWL-max']
    cmb_all['AWL-min'].plot(title='AWL-min selections for '+ticker, ylabel='window length')
    plt.grid(visible=True, which='both')
    plt.show()

    # and print some statistics for the cumulative or average cumulative return
    if set_roll_wealth > 0:
        print(all.mean(axis=0))
    else:
        print(all.iloc[-1])

    # and cumulative return by year
    r_all = pd.concat([r, new_ret], axis=1)
    by_year = r_all.groupby(by=new_ret.index.year).apply(lambda x: (x+1).prod()-1)*100
    print(by_year)