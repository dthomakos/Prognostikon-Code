#
# Python code adding for the post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-adaptive-mean/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

## Import the packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader as pdr
import yfinance as yf

## Get some data
variable = ['CPIAUCSL']
raw_data = pdr.fred.FredReader(variable, start='2000-01-01', end='2024-12-01').read()
rets = raw_data.pct_change(periods=1).dropna() # diff(periods=1).dropna() # for TB3MS and FEDFUNDS
nobs = rets.shape[0]

## Initialize parameters and storage
roll = 36
burn_in = roll + 2

for gamma in [0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]:
    #
    store_fc = pd.DataFrame(data=None, index=rets.index, columns=['FCB-roll-'+str(roll), 'FC-roll-'+str(roll), 'FCB-rec', 'FB-rec'])
    store_ac = pd.DataFrame(data=None, index=rets.index, columns=['Actual'])
    store_fe = pd.DataFrame(data=None, index=rets.index, columns=['FEB-roll-'+str(roll), 'FE-roll-'+str(roll), 'FEB-rec', 'FE-rec'])
    store_sgn = pd.DataFrame(data=None, index=rets.index, columns=['FEB-roll-'+str(roll), 'FE-roll-'+str(roll), 'FEB-rec', 'FE-rec'])


    ## The computations are very direct
    for i in np.arange(roll, nobs, 1):

        # Section the data
        xrec = rets.iloc[:i]
        xrol = xrec[-roll:]
        actual = rets.iloc[i].values[0]

        # Compute the benchmark forecasts
        mu_rol = xrol.mean().values[0]
        mu_rec = xrec.mean().values[0]
        feb_rol = actual - mu_rol
        feb_rec = actual - mu_rec

        if i > burn_in:
            Ip_rol = (store_fe.iloc[:i,1] > 0).astype(float).iloc[-roll:]
            Pp_rol = (store_fe.iloc[:i,1] > 0).iloc[-roll:].mean()
            x_rol = (Ip_rol - Pp_rol).values.reshape(-1,1)
            y_rol = (xrol - mu_rol).values
            cxy_rol = np.mean(y_rol*x_rol)
            vx_rol  = np.mean(x_rol ** 2)
            b_rol = 2*np.sign(cxy_rol)*cxy_rol/vx_rol
            if cxy_rol > 0:
                factor = gamma ** np.log(i)
            elif cxy_rol < 0:
                factor = (1/gamma) ** np.log(i)
            xf_rol = mu_rol + factor*np.sign(b_rol)*b_rol*x_rol[-1]
            #
            Ip_rec = (store_fe.iloc[:i,3] > 0).astype(float)
            Pp_rec = (store_fe.iloc[:i,3] > 0).mean()
            x_rec = (Ip_rec - Pp_rec).values.reshape(-1, 1)
            y_rec = (xrec - mu_rec).values
            cxy_rec = np.mean(y_rec*x_rec)
            vx_rec  = np.mean(x_rec ** 2)
            b_rec = 2*np.sign(cxy_rec)*cxy_rec/vx_rec
            if cxy_rec > 0:
                factor = gamma ** np.log(i)
            elif cxy_rec < 0:
                factor = (1/gamma) ** np.log(i)
            xf_rec = mu_rec + factor*np.sign(b_rec)*b_rec*x_rec[-1]
            #
            fe_rol = actual - xf_rol
            fe_rec = actual - xf_rec
        else:
            xf_rol = mu_rol
            xf_rec = mu_rec
            fe_rol = feb_rol
            fe_rec = feb_rec

        # Store the forecasts, forecast errors and signs
        store_fc.iloc[i] = np.hstack([mu_rol, xf_rol, mu_rec, xf_rec])
        store_ac.iloc[i] = actual
        store_fe.iloc[i] = np.hstack([feb_rol, fe_rol, feb_rec, fe_rec])
        store_sgn.iloc[i] = (np.sign(actual) == np.hstack([np.sign(mu_rol), np.sign(xf_rol), np.sign(mu_rec), np.sign(xf_rec)]))

    print('Scaling is = ',factor)
    mse = (store_fe.dropna() ** 2).mean()
    rmse = mse/mse.iloc[2]
    mae = (store_fe.dropna().abs()).mean()
    rmae = mae/mae.iloc[2]
    ssr = store_sgn.dropna().mean()
    all = pd.concat([rmse, rmae, ssr], axis=1)
    all.columns = ['relMSE', 'relMAE', 'SSR']
    print(all)
