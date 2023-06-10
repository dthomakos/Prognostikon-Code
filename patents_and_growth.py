#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/06/10/patents-and-economic-growth/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr

# Download the data on total patents - convert to annual growth rates
data_patents = pdr.fred.FredReader(['PATENTUSALLTOTAL', 'PATENT4NCNTOTAL', 'PATENT4NILTOTAL',
'PATENT4NJPTOTAL', 'PATENT4NDETOTAL', 'PATENT4NKRTOTAL', 'PATENT4NTWTOTAL', 'PATENT4NINTOTAL',
'PATENT4NGBTOTAL'], start='1992-01-01', end='2022-12-01').read().pct_change().dropna()
data_patents.columns = ['Pat-US', 'Pat-CN', 'Pat-IL', 'Pat-JP', 'Pat-DE',
'Pat-KR', 'Pat-TW', 'Pat-IN', 'Pat-UK']

# Download the data on real GDP - convert to annual growth rates
data_growth = pdr.fred.FredReader(['RGDPNAUSA666NRUG', 'RGDPNACNA666NRUG', 'RGDPNAILA666NRUG', 'RGDPNAJPA666NRUG',   'RGDPNADEA666NRUG', 'RGDPNAKRA666NRUG', 'RGDPNATWA666NRUG', 'RGDPNAINA666NRUG', 'RGDPNAGBA666NRUG'], start='1992-01-01', end='2022-12-01').read().pct_change().dropna()
data_growth.columns = ['Grw-US', 'Grw-CN', 'Grw-IL', 'Grw-JP', 'Grw-DE',
'Grw-KR', 'Grw-TW', 'Grw-IN', 'Grw-UK']

# Quickly get and print the delays and cross-correlations between the variables for each country
print("-----")

# First from lagged patents to growth
for i in range(6):
    set_lag = i
    data_both = pd.concat([data_patents.shift(periods=set_lag), data_growth], axis=1).dropna()
    corr = data_both.corr().loc[data_growth.columns, data_patents.columns]
    grw_patents = pd.DataFrame(np.diag(corr), index=corr.columns, columns=[set_lag])
    print(grw_patents)

print("-----")

# Then from lagged growth to patents
for i in range(6):
    set_lag = i
    data_both = pd.concat([data_patents, data_growth.shift(periods=set_lag)], axis=1).dropna()
    corr = data_both.corr().loc[data_growth.columns, data_patents.columns]
    grw_patents = pd.DataFrame(np.diag(corr), index=corr.columns, columns=[set_lag])
    print(grw_patents)

# I looked at the data from their printouts and collected them into a new dataframe
countries = ['US', 'CN', 'IL', 'JP', 'DE', 'KR', 'TW', 'IN', 'UK']
# Patents-to-growth
ptg = np.array([[4, 0.2683], [4, 0.3768], [4, 0.1176], [4, 0.4384], [4, 0.2324],
[1, 0.3832], [4, 0.5923], [1, 0.2163], [4, 0.2116]])
# to dataframe...
ptg = pd.DataFrame(data=ptg, index=countries, columns=['Lag', 'Cross-Correlation'])
# Growth-to-patents
gtp = np.array([[0, 0.1245], [0, 0.0882], [3, 0.3966], [2, 0.1653], [2, 0.2908], [2, 0.3964],
[2, 0.3332], [3, 0.3711], [0, 0.0770]])
# to dataframe...
gtp = pd.DataFrame(data=gtp, index=countries, columns=['Lag', 'Cross-Correlation'])

# The figures in the post were then made with Libre Office...
