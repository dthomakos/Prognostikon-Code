#
# Python code for nearest neighbors forecasting, adapted from the following post
#
# https://prognostikon.wordpress.com/2023/05/06/peaks-and-troughs-forecasting-us-real-gdp-growth/
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
from dateutil.relativedelta import relativedelta

# Let's define the function that does the hard work for you
def fred_nn_forecaster(ticker, name=None, start_date=None, end_date=None, log=True, diff=1, peak_location=None, nn=10, h=10, freq='monthly', look_back=None, plot=True, save=False):
    """
    ...

    Out-of-sample forecasting of any monthly or quarterly data series from FRED database
    https://fred.stlouisfed.org/ using two types of nearest neighbors
    ...

    ticker: character string, ticker representation of FRED series

    name: character string, simplified variable name

    start_date: character string, in format 'YYYY-MM-DD'

    end_date: character string, in format 'YYYY-MM-DD'

    log: boolean, set to True if you wish to take natural logarithms

    diff: integer, order of regular (diff=1) or seasonal (diff=12 or diff=4) differencing

    peak_location: integer, position of last peak counting from the end of the series

    nn: integer, number of nearest neighbors to consider in averaging

    h: integer, steps-ahead of forecasting horizong

    freq: character string, frequency of series, either 'monthly' or 'quarterly'

    look_back: either None, or a list of particular NN to consider; if this is used the length
    of the list will overide the nn setting

    plot: boolean, set to True to produce a plot of of the forecasts

    save: boolean, set to True to save the forecasts in csv files

    return: a dictionary with components 'raw_data' which returns the original downloaded data, 'x' which returns the transformed series, 'NNP' which returns the peak-based forecasts with 2 s.e. bounds, 'NN' which returns the standard forecasts with 2 s.e. bounds, 'both' which returns both forecasts without s.e. bounds and 'peak_dates' which returns the peak-based dates of the nearest neighbors

    ...
    """
    # Check start and end dates
    if (start_date == None) or (end_date == None):
        raise ValueError('You must specify valid start and end dates!')
    elif (start_date != None) and (end_date != None):
        if start_date > end_date:
            raise ValueError('End date must be later than start date!')

    # Check frequency
    if (freq != 'monthly') and (freq != 'quarterly'):
        raise ValueError('This function works only for monthly and quarterly data!')

    # Check peak location
    if peak_location == None:
        raise ValueError('You must select a peak location!')

    # Conversion for relativedelta
    if freq == 'quarterly':
        cf = 3
        sf = 'Q-DEC'
    else:
        cf = 1
        sf = 'M'

    # Get the data series, transform if necessary
    raw_data = pdr.fred.FredReader(ticker, start=start_date, end=end_date).read()
    x = raw_data
    if log:
        log_data = raw_data.apply(np.log)
        x = log_data
        if diff > 0:
            dlog_data = log_data.diff(periods=diff).dropna()*100
            x = dlog_data
    else:
        if diff > 0:
            diff_data = raw_data.diff(periods=diff).dropna()
            x = diff_data

    # Give nice name if asked for
    if name != None:
        x.columns = [name]

    # Given pick location find the NNP forecast first
    set_peak = peak_location
    # Fix target and training sets
    set_last = x.iloc[-set_peak:]
    set_rest = x.iloc[:-set_peak]

    # Compute the magnitude and positions of the NNP
    mod = set_rest.shape[0]%set_peak
    set_rest = set_rest.iloc[mod:]
    set_M = set_rest.shape[0]-set_peak+1
    store_dates = pd.DataFrame(data=np.zeros([set_peak, set_M]))
    store_values = pd.DataFrame(data=np.zeros([set_peak, set_M]))
    store_distances = pd.DataFrame(data=np.zeros([set_M, 1]), columns=['Distances'])

    # Carefull in the NNP - the first position must be a peak!!
    for i in range(set_rest.shape[0]-set_peak+1):
        xi = set_rest.iloc[i:(i+set_peak)]
        store_dates.iloc[:,i] = xi.index
        store_values.iloc[:, i] = xi.to_numpy()
        test = (xi.iloc[0] > xi.iloc[1:]).all().to_numpy()[0]
        if test:
            dist = np.sqrt(((xi.to_numpy()[0] - set_last.to_numpy()[0]) ** 2).sum())
        else:
            dist = 999
        store_distances.iloc[i,:] = dist

    # Now, get the dates and values for the ordered distances
    idx = store_distances.sort_values(by='Distances').index
    store_dates = store_dates.loc[:,idx]
    store_values = store_values.loc[:, idx]

    # Select the first NNP
    if look_back == None:
        look_back = store_dates.columns[range(nn)]
    frc1 = pd.DataFrame(data=np.zeros([set_peak+h, len(look_back)]), columns=look_back)

    # For each period we must also find the next set_peak values for the forecast
    for j in look_back:
        idx = store_dates.loc[:,j]
        frc1.loc[:,j] = x.loc[idx[0]:(idx[set_peak-1]+relativedelta(months=cf*h))].to_numpy()

    # Compute the mean over the NNP
    mean1 = frc1.mean(axis=1)
    # and the standard deviation of this mean
    std = frc1.std(axis=1)/np.sqrt(len(look_back))
    # bounds...
    mean1_lb = mean1 - 2*std
    mean1_ub = mean1 + 2*std
    mean1_all = pd.concat([mean1_lb, mean1, mean1_ub], axis=1)

    # Let us add the standard NN forecast for comparison based on the last observation
    last_obs = x.iloc[-1]
    dist = np.sqrt(((x - x.iloc[-1])**2)).sort_values(by=x.columns[0]).iloc[1:]
    # and get the forecast
    frc2 = pd.DataFrame(data=np.zeros([h, nn]), columns=range(nn))
    for j in range(nn):
        z = x.loc[dist.index[j]+relativedelta(months=cf):(dist.index[j] +relativedelta(months=cf*h))].to_numpy()
        if len(z) == h:
            frc2.iloc[:,j] = z
        else:
            next
    # clean columns of all zeros
    check_frc2 = frc2.apply(lambda z: (z == 0).all())
    frc2 = frc2.drop(columns=frc2.columns[check_frc2])

    # get the forecast and bounds
    mean2 = frc2.mean(axis=1)
    std = frc2.std(axis=1)/np.sqrt(nn)
    mean2_lb = mean2 - 2*std
    mean2_ub = mean2 + 2*std
    mean2_all = pd.concat([mean2_lb, mean2, mean2_ub], axis=1)

    # Done, fix nicely for output, plot and saving
    actual = pd.DataFrame(data=np.vstack([set_last.to_numpy(), np.repeat(np.nan, h).reshape(-1, 1)]),
    index=pd.date_range(start=set_last.index[0], periods=set_peak+h, freq=sf))
    mean1_all.iloc[:set_peak] = np.nan
    mean2_all = pd.concat([pd.DataFrame(np.repeat(np.nan, 3*set_peak).reshape(set_peak, 3), columns=range(3)), mean2_all], axis=0)
    mean1_all.index = actual.index
    mean2_all.index = actual.index
    #
    actual_frc1 = pd.concat([actual, mean1_all], axis=1)
    actual_frc1.columns = [name, 'NNP-forecast lower bound',
    'NNP-forecast', 'NNP-forecast upper bound']
    actual_frc2 = pd.concat([actual, mean2_all], axis=1)
    actual_frc2.columns = [name, 'NN-forecast lower bound',
    'NN-forecast', 'NN-forecast upper bound']
    actual_frc = pd.concat([actual, mean1_all.iloc[:,1], mean2_all.iloc[:,1]], axis=1)
    actual_frc.columns = [name, 'NNP-forecast', 'NN-forecast']
    #
    if plot:
        ax1 = actual_frc1.plot(grid='both', color=['black', 'red', 'blue', 'red'], style=['-', ':', '--', ':'],
        title=x.columns[0]+' and NN peak-based forecast with 95% bounds using '+str(len(look_back))+' NN',
        figsize=[13, 8], xlabel='Date')
        ax1.xaxis.grid(True, which='minor')
        ax1.yaxis.grid(True, which='minor')
        plt.show()
        #
        ax2 = actual_frc2.plot(grid='both', color=['black', 'red', 'blue', 'red'], style=['-', ':', '--', ':'],
        title=x.columns[0]+' and standard NN forecast with 95% bounds using '+str(nn)+' NN',
        figsize=[13, 8], xlabel='Date')
        ax2.xaxis.grid(True, which='minor')
        ax2.yaxis.grid(True, which='minor')
        plt.show()
        #
        # for this last plot insert the last actual value in the forecasts to look nicer
        actual_frc.loc[actual.index[set_peak-1],['NNP-forecast', 'NN-forecast']] = actual_frc.loc[actual.index[set_peak-1], x.columns[0]]
        ax3 = actual_frc.plot(grid='both', color=['black', 'blue', 'green'], style=['-', '--', ':'],
        title=x.columns[0]+' and both NN-type forecasts using '+str(len(look_back))+' NNP and '+str(nn)+' NN', figsize=[13, 8], xlabel='Date')
        ax3.xaxis.grid(True, which='minor')
        ax3.yaxis.grid(True, which='minor')
        plt.show()
    #
    if save:
        actual_frc1.to_csv(x.columns[0]+'-NNP.csv')
        actual_frc2.to_csv(x.columns[0]+'-NN.csv')
    #
    output = { 'raw_data': raw_data, 'x': x, 'NNP': actual_frc1, 'NN': actual_frc2,
    'both': actual_frc, 'peak_dates': store_dates }
    return output

