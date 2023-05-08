#
# Python code for nearest neighbors forecasting, adapted from the following post
#
# https://prognostikon.wordpress.com/2023/05/06/peaks-and-troughs-forecasting-us-real-gdp-growth/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
# This file contains three examples on the usage of the function in fred_nn_forecaster.py
#

# Import packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
from dateutil.relativedelta import relativedelta
import fred_nn_forecaster as nn

# Example #1: replication of the results in the blog post, GDP growth forecasting, quarterly frequency
set_ticker = 'GDPC1'
set_name = 'Growth'
set_start = '1947-01-01'
set_end = '2023-03-31'
set_log = True
set_diff = 4
set_peak = 8
set_nn = 2
set_h = 8
set_freq = 'quarterly'
set_lookback = [6, 40, 95] # overwrite the set_nn=3 setting for the peak-based NNP
do_plot = True
do_save = False
out1 = nn.fred_nn_forecaster(set_ticker, set_name, set_start, set_end,
set_log, set_diff, set_peak, set_nn, set_h, set_freq, set_lookback, do_plot, do_save)

# Example #2: inflation forecasting, monthly frequency
set_ticker = 'CPIAUCSL'
set_name = 'Inflation'
set_start = '1947-01-01'
set_end = '2023-03-31'
set_log = True
set_diff = 12
set_peak = 10
set_nn = 4
set_h = 10
set_freq = 'monthly'
set_lookback = [404, 4, 328, 38, 406] # overwrite the set_nn=3 setting for the peak-based NNP
do_plot = True
do_save = False
out2 = nn.fred_nn_forecaster(set_ticker, set_name, set_start, set_end,
set_log, set_diff, set_peak, set_nn, set_h, set_freq, set_lookback, do_plot, do_save)

# Example #3: unemployment forecasting, monthly frequency
set_ticker = 'UNRATE'
set_name = 'Unemployment'
set_start = '1947-01-01'
set_end = '2023-04-30'
set_log = False
set_diff = 0
set_peak = 16
set_nn = 5
set_h = 9
set_freq = 'monthly'
set_lookback = None #[229, 208, 833, 612] #, 830] # overwrite the set_nn=5 setting for the peak-based NNP
do_plot = True
do_save = False
out3 = nn.fred_nn_forecaster(set_ticker, set_name, set_start, set_end,
set_log, set_diff, set_peak, set_nn, set_h, set_freq, set_lookback, do_plot, do_save)

