#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/macroeconomic-uncertainty-currency-and-commodities-trading/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import scipy as sp
import yfinance as yf

# Input the the uncertainty indices
enames = ['USEPUINDXM', 'RUSEPUINDXM', 'EUEPUINDXM', 'INDEPUINDXM', 'UKEPUINDXM']
data = pdr.fred.FredReader(enames, start='1990-01-01', end='2024-01-31').read()
data.columns = ['UI-US', 'UI-RS', 'UI-EU', 'UI-IN', 'UI-UK']
# Now input the currency ETFs
fxnames = ['UUP', 'FXE', 'FXY', 'FXB', 'FXF', 'FXA', 'DBA', 'DBC', 'OIH', 'WEAT']
fx_data = yf.download(fxnames, start='2002-01-01', end='2024-01-31', interval='1mo')['Adj Close'].dropna()
# and merge them
all_data = pd.concat([data, fx_data], axis=1)

# Create the index and add it to the datasset
idx_data = all_data[['UI-US', 'UI-RS', 'UI-EU', 'UI-IN', 'UI-UK']]
all_data['GUI'] = (idx_data/idx_data.iloc[0]).mean(axis=1)

# All variables of interest together
all_names = np.hstack([fxnames, 'GUI']).tolist()

# Select the starting years for computing the cross-correlationis
years = pd.date_range(start='2007-03-01', end='2023-01-01', freq='AS-JAN')
store_rho = pd.DataFrame(data=None, index=years, columns=fxnames)
# Sequence of lags for the cross-correlations
seq_lags = np.arange(1, 13, 1)
# Makes sense to consider the monthly growth rates but you can change this
set_diff = 1

# Loop over the different starting years
for yy in years:
    # Compute the growth rates, start from each particular year
    new_d_data = all_data.pct_change(periods=set_diff).dropna().loc[yy:]
    store_cz = pd.DataFrame(data=None, index=seq_lags, columns=fxnames)

    # For each of the lags compute the cross-correlation
    for i in seq_lags:
        z = pd.concat([new_d_data[fxnames],new_d_data['GUI'].shift(periods=i)], axis=1).dropna()
        cz = z.corr()
        store_cz.loc[i] = (cz[fxnames].loc['GUI']).to_numpy()

    # For each year and for all lags compute and store the lag with max abs cross-correlation
    store_rho.loc[yy] = store_cz.apply(lambda x: np.argmax(np.abs(x)), axis=0)+1

# Done, print the optimal lags
print(store_rho)
store_rho.to_csv(str(set_diff)+'-rho.csv')

# Now for the forecasting stage - note that here we start from each particular year and
# then we make the out-of-sample
set_year = '2019-01-01'
d_data = all_data[all_names].pct_change(periods=set_diff).dropna()
d_data = d_data.loc[set_year:]
yname = 'WEAT'
# Use the in-sample mean lag
set_lag = int(np.floor(store_rho[yname].mean()))
# IMPORTANT: CHANGE TO 2024-01-01 FOR WHEAT, OIL, FOOD AND TO 2024-02-01 FOR THE FINANCIALS!!!
f_data = pd.DataFrame(data=None, index=pd.date_range(start='2024-02-01', periods=set_lag, freq='MS'), columns=[yname, 'GUI'])
# Compute the delay regression
py = pd.concat([d_data[yname], d_data['GUI'].shift(periods=set_lag)], axis=1).dropna()
pyc = py.corr().iloc[1,0]
sy = py.std()
my = py.mean()
beta = pyc*(sy[yname]/sy['GUI'])
alpha = my[yname] - beta*my['GUI']
f_data['GUI'] = (d_data['GUI'].iloc[-set_lag:]).to_numpy()
# Compute the out-of-sample forecast
f_data[yname] = alpha + (d_data['GUI'].iloc[-set_lag:]).to_numpy()*beta
both_data = pd.concat([py, f_data], axis=0)
# print and plot!
recent_both = both_data.loc['2023-10-01':]
recent_both.plot(grid='both')
plt.show()
print(recent_both)

# Finally, for the trading experiment using the same dependent variable as above
y = all_data[yname].pct_change() # this must be "pure" monthly returns
x = all_data['GUI'].pct_change(periods=set_diff) # but this can be anything in terms of periods
y = y.loc[x.index]
x = x.loc[x.index]

# and here we trade for each different lag of the index and compare with the optimal ones found before
store_xy = pd.DataFrame(data=None, index=seq_lags, columns=['Strategy', 'Benchmark'])
#
for i in seq_lags:
    xy = (y*(x.shift(periods=i).apply(np.sign))).dropna()
    rxy = pd.concat([xy, y.loc[xy.index]], axis=1)
    rxy.columns = ['GUI'+' on '+yname, yname]
    t = ((rxy+1).prod()-1)*100
    #t_by_year = rxy.groupby(by=rxy.index.year).apply(lambda x: (x+1).prod())
    #print(i)
    #print(t_by_year)
    store_xy.loc[i] = t.to_numpy()
    # Plot the best performers
    z_uup =  (i == 4)  and (yname == 'UUP')
    z_fxe =  (i == 12) and (yname == 'FXE')
    z_fxy =  (i == 2)  and (yname == 'FXY')
    z_fxb =  (i == 10) and (yname == 'FXB')
    z_fxf =  (i == 2)  and (yname == 'FXF')
    z_fxa =  (i == 3)  and (yname == 'FXA')
    z_dba =  (i == 12) and (yname == 'DBA')
    z_dbc =  (i == 11) and (yname == 'DBC')
    z_oih =  (i == 10) and (yname == 'OIH')
    z_weat = (i == 3)  and (yname == 'WEAT')
    cxy = (((rxy+1).cumprod()-1)*100)
    set_title = 'Global Uncertainty Index (GUI) strategy for '+yname
    if z_uup or z_fxe or z_fxy or z_fxb or z_fxf or z_fxa or z_weat or z_dba or z_dbc or z_oih:
        cxy.plot(title=set_title, xlabel='Date', ylabel='return in percent', grid='both')
        plt.show()

# Print the trading results
print(store_xy)