#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/06/19/midas-touch-gold-agricultural-prices-as-one/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import openpyxl as xl
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.api import ARDL

# Read the data, fix dates
data = pd.read_excel('gold_and_agriculture.xlsx', sheet_name='data', header=0, index_col=0)
data.index = pd.date_range('1960-01-01', periods=data.shape[0], freq='M')

# Set a sequence of starting dates for the analysis
set_start_date = ['1967-12-31', '1977-12-31', '1987-12-31', '1997-12-31', '2007-12-31', '2015-12-31']

# Loop over the starting dates
for sd in set_start_date:

    # Crop the data, convert into index for the plot
    data = data[['GOLD', 'iAGRICULTURE']].loc[sd:]
    data = (data/data.iloc[0])*100

    # Present the plot nicely
    ax1 = data['GOLD'].plot(color=['gold'], ylabel='Gold Index', title='Gold price index and agricultural prices index, monthly data', xlabel='Date')
    ax1.xaxis.grid(True, which='major')
    ax1.yaxis.grid(True, which='major')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    data['iAGRICULTURE'].plot(ax=ax2, color=['green'], ylabel='Agricultural Prices Index')
    ax2.legend(loc='lower right')
    plt.show()

    # Take log-growth rates
    diff_data = data.apply(np.log).diff().dropna()
    sel_gold = ar_select_order(diff_data['GOLD'], maxlag=13, ic='hqic', trend='c')
    fit_gold = sel_gold.model.fit()
    res_gold = fit_gold.resid

    # Pre-filter with a AR(HQ) model for both series
    #
    sel_agri = ar_select_order(diff_data['iAGRICULTURE'], maxlag=13, ic='hqic', trend='c')
    fit_agri = sel_agri.model.fit()
    res_agri = fit_agri.resid
    #

    # Store the cross-correlations of the residuals
    store_res_ccf = pd.DataFrame(data=None, index=range(23), columns=['A-->G', 'G-->A'])
    for i in range(23):
        res_data = pd.concat([res_gold, res_agri.shift(periods=i)], axis=1).dropna()
        store_res_ccf.iloc[i, 0] = res_data.corr().iloc[1,0]
        res_data = pd.concat([res_gold.shift(periods=i), res_agri], axis=1).dropna()
        store_res_ccf.iloc[i, 1] = res_data.corr().iloc[1,0]

    # Start printing the output...
    print('\n')
    print(sd)
    print(data.corr())
    print(store_res_ccf)

    # Now for the ARDL analysis...
    #
    # First gold on agriculture...we use model reduction once...
    ardl_agri = ARDL(diff_data['iAGRICULTURE'], sel_agri.ar_lags, pd.DataFrame(diff_data['GOLD']), 24)
    fit_ardl_agri = ardl_agri.fit()
    new_agri = pd.DataFrame(ardl_agri.exog_names).iloc[np.where(fit_ardl_agri.pvalues <= 0.075)[0]].index
    if any(new_agri < np.max(sel_agri.ar_lags)+1):
        if np.max(sel_agri.ar_lags) <= 1:
            setp = np.max(sel_agri.ar_lags)
        else:
            setp = np.max(sel_agri.ar_lags)+1
        xlags = new_agri[setp:] - np.max(sel_agri.ar_lags)-1
        ylags = np.max(sel_agri.ar_lags)
    else:
        xlags = new_agri - np.max(sel_agri.ar_lags)-1
        ylags = 0
    # Fit the final model, compute Wald test for long-run impact
    ardl_agri = ARDL(diff_data['iAGRICULTURE'], ylags, pd.DataFrame(diff_data['GOLD']), xlags)
    fit_ardl_agri = ardl_agri.fit()
    print(fit_ardl_agri.summary())
    if sd == '1967-12-31':
        wald_test = fit_ardl_agri.wald_test('GOLD.L0+GOLD.L8+GOLD.L9+GOLD.L16+GOLD.L23', scalar=True)
    if sd == '1977-12-31':
        wald_test = fit_ardl_agri.wald_test('GOLD.L0+GOLD.L2+GOLD.L4+GOLD.L8+GOLD.L9+GOLD.L16+GOLD.L21+GOLD.L24', scalar=True)
    if sd == '1987-12-31':
        wald_test = fit_ardl_agri.wald_test('GOLD.L0+GOLD.L2+GOLD.L6+GOLD.L7+GOLD.L9+GOLD.L14+GOLD.L19+GOLD.L21+GOLD.L24', scalar=True)
    if sd == '1997-12-31':
        wald_test = fit_ardl_agri.wald_test('GOLD.L0+GOLD.L2+GOLD.L7+GOLD.L9+GOLD.L21+GOLD.L24', scalar=True)
    if sd == '2007-12-31':
        wald_test = fit_ardl_agri.wald_test('GOLD.L0+GOLD.L8+GOLD.L19+GOLD.L21+GOLD.L24', scalar=True)
    if sd == '2015-12-31':
        wald_test = fit_ardl_agri.wald_test('GOLD.L0+GOLD.L9+GOLD.L14+GOLD.L21+GOLD.L21', scalar=True)
    print('Wald test for long-run impact (p-value) = ', wald_test.pvalue)
    print('\n')

    # Next, agriculture on gold...same things as before...
    #
    ardl_gold = ARDL(diff_data['GOLD'], sel_gold.ar_lags, pd.DataFrame(diff_data['iAGRICULTURE']), 24)
    fit_ardl_gold = ardl_gold.fit()
    new_gold = pd.DataFrame(ardl_gold.exog_names).iloc[np.where(fit_ardl_gold.pvalues <= 0.075)[0]].index
    if any(new_gold < np.max(sel_gold.ar_lags)+1):
        if np.max(sel_gold.ar_lags) <= 1:
            setp = np.max(sel_gold.ar_lags)
        else:
            setp = np.max(sel_gold.ar_lags)+1
        xlags = new_gold[setp:] - np.max(sel_gold.ar_lags)-1
        ylags = np.max(sel_gold.ar_lags)
    else:
        xlags = new_gold - np.max(sel_gold.ar_lags)-1
        ylags = 0
    # Fit the final model, compute Wald test for long-run impact
    ardl_gold = ARDL(diff_data['GOLD'], ylags, pd.DataFrame(diff_data['iAGRICULTURE']), xlags)
    fit_ardl_gold = ardl_gold.fit()
    print(fit_ardl_gold.summary())
    if sd == '1967-12-31':
        wald_test = fit_ardl_gold.wald_test('iAGRICULTURE.L0+iAGRICULTURE.L7', scalar=True)
    if sd == '1977-12-31':
        wald_test = fit_ardl_gold.wald_test('iAGRICULTURE.L7', scalar=True)
    if sd == '1987-12-31':
        wald_test = fit_ardl_gold.wald_test('iAGRICULTURE.L0+iAGRICULTURE.L10', scalar=True)
    if sd == '1997-12-31':
        wald_test = fit_ardl_gold.wald_test('iAGRICULTURE.L0+iAGRICULTURE.L10', scalar=True)
    if sd == '2007-12-31':
        wald_test = fit_ardl_gold.wald_test('iAGRICULTURE.L0+iAGRICULTURE.L7+iAGRICULTURE.L10+iAGRICULTURE.L11+iAGRICULTURE.L16+iAGRICULTURE.L19', scalar=True)
    if sd == '2015-12-31':
        wald_test = fit_ardl_gold.wald_test('iAGRICULTURE.L0+iAGRICULTURE.L11', scalar=True)
    #
    print('Wald test for long-run impact (p-value) = ', wald_test.pvalue)