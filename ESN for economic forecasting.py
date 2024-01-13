#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2024/01/13/echo-state-networks-economic-forecasting-and-minimum-predictive-loss/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#
# Code for the training of the ESN adapted from:
#
# """
# A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data
# in "plain" scientific Python.
# from https://mantas.info/code/simple_esn/
# (c) 2012-2020 Mantas Lukoševičius
# Distributed under MIT license https://opensource.org/licenses/MIT
# """
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eig
from numpy.linalg import solve
import pandas as pd
import pandas_datareader as pdr
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg

# Function to generate an ESN reservoir
def esn_reservoir(inSize, resSize=1000, seed=10):
    np.random.seed(seed)
    Win = (np.random.rand(resSize,1+inSize) - 0.5) * 1
    W = np.random.rand(resSize,resSize) - 0.5
    rhoW = max(abs(eig(W)[0]))
    W *= 1.25 / rhoW
    return Win, W

# Function to create the state matrix and train the output
def esn_train(inData, outData, resSize=1000, seed=10, alpha=0.3, initNobs=100, regcoef=1e-8):
    inSize = inData.shape[0]
    inNobs = inData.shape[1]
    Win, W = esn_reservoir(inSize, resSize, seed)
    X = np.zeros((1+inSize+resSize,inNobs-initNobs))
    Yt = outData[:,initNobs:(inNobs+1)]
    x = np.zeros((resSize,1))
    for t in range(inNobs):
        u = inData[:,t].reshape(-1, 1)
        x = (1-alpha)*x + alpha*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
        if t >= initNobs:
            X[:,t-initNobs] = np.vstack((1,u,x))[:,0]
    Wout = solve( np.dot(X, X.T) + regcoef*np.eye(1+inSize+resSize), np.dot(X, Yt.T) ).T
    return Wout, X, Win, W, alpha, x

# Function to predict given a trained network
def esn_predict(Wout, X, Win, W, alpha, x, testData):
    outSize = Wout.shape[0]
    testNobs = testData.shape[1]
    Y = np.zeros((outSize,testNobs))
    for t in range(testNobs):
        u = testData[:,t].reshape(-1, 1)
        x = (1-alpha)*x + alpha*np.tanh( np.dot( Win, np.vstack((1,u)) ) + np.dot( W, x ) )
        y = np.dot( Wout, np.vstack((1,u,x)) )
        Y[:,t] = y
    return Y

# Function to compute in-sample fit given a trained network
def esn_fit(inData, outData, resSize, seed, alpha, initNobs):
    Wout, X, Win, W, alpha, x = esn_train(inData, outData, resSize, seed, alpha, initNobs)
    fit = esn_predict(Wout, X, Win, W, alpha, x, inData)
    return fit, Wout

# Function to compute the in-sample AIC or BIC for resolution size selection
def esn_model_selection(x, y, set_resSize, set_seed, set_alpha, set_initNobs):
    # Get range of resolutions values
    min_resSize = set_resSize.min()
    max_resSize = set_resSize.max()
    # initialize storage for the AIC and the BIC
    aic = np.zeros([set_resSize.shape[0], 2])
    bic = np.zeros([set_resSize.shape[0], 2])
    # run the loop
    for res in set_resSize:
        # Fit
        fit, Wout = esn_fit(x, y, res, set_seed, set_alpha, set_initNobs)
        # ...you must remove initial observations
        yplot = y[:,set_initNobs:].T
        fit = fit[:,set_initNobs:].T

        # Compute the MSE
        mse = ((yplot - fit)**2).mean()

        # and the BIC criterion
        kappa = len(Wout.T)
        n = len(fit)
        bic[res-min_resSize, 0] = res
        bic[res-min_resSize, 1] = n*np.log(mse) + kappa*np.log(n)
        #
        aic[res-min_resSize, 0] = res
        aic[res-min_resSize, 1] = n*np.log(mse) + 2*kappa

    # Find optimal resolution size and return
    bic_resSize = int(bic[bic[:,1].argmin(), 0])
    aic_resSize = int(aic[aic[:,1].argmin(), 0])
    #
    return aic_resSize, bic_resSize

# Function to generate via recursion the Chebyshev polynomial basis for a time trend
def Chebyshev_basis(deg, nobs):
    trend = np.arange(1, nobs+1, 1)/nobs
    basis = np.array(np.zeros([nobs, deg+1]))
    basis[:, 0] = np.ones(nobs)
    if deg == 0:
        return basis
    else:
        basis[:, 1] = trend
        for i in np.arange(2, deg+1, 1):
            basis[:, i] = 2*trend*basis[:,i-1] - basis[:,i-2]
        return basis

# Select pairs for analysis, download and prepare the data
pair1 = ['RSAFS', 'CPIAUCSL']
pair1_names = ['Retail Trade', 'Inflation']
pair2 = ['MCUMFN', 'RSAFS']
pair2_names = ['Capacity in Mfg', 'Retail Trade']
pair3 = ['MCUMFN', 'IPMAN']
pair3_names = ['Capacity in Mfg', 'Ind. Production Mfg']
set_pair = pair1
set_pair_names = pair1_names
#
data = pdr.fred.FredReader(set_pair, start='1990-01-01', end='2023-12-31').read()
data.columns = set_pair_names
data = data.apply(np.log).diff(periods=1)

# Set the starting date
set_start_date = '2000-01-01'
data = data.dropna().loc[set_start_date:]

# Set dependent and exmplanatory variables
set_depvar = set_pair_names[0]
set_expvar = set_pair_names[1]

# Initial definition for the number of observations
nobs = data.shape[0]

# Add the Chebyshev basis
set_deg = 0
basis_raw = Chebyshev_basis(set_deg, nobs+1)
basis = pd.DataFrame(basis_raw[:-1,:], index=data.index, columns=['B'+str(i) for i in range(set_deg+1)])
basis_names = basis.columns.tolist()

# Add lags, use appropriate lag or lags below!
set_lags = np.array([3])
data_lags = pd.DataFrame(data=None, index=data.index, columns=[])
for i in set_lags:
    lags_i = data.shift(periods=i)
    lags_i.columns = [set_pair_names[j]+str(i) for j in np.arange(len(set_pair_names))]
    data_lags = pd.concat([data_lags, lags_i], axis=1)
# Keep the lag names
lag_names = data_lags.columns.tolist()

# Put together, drop the NAs
data = pd.concat([data, data_lags, basis], axis=1).dropna()

# Redefine nobs and initial window
nobs = data.shape[0]
ini_wind = 60

# Rolling or recursive estimation?
set_rolling = False

# Define the network's architecture - very simple!
set_resSize = 1
set_seed = 22
set_alpha = 0.5
set_initNobs = 12
train_every = 1

# and initialize storage
store_frc = pd.DataFrame(data=None, index=data.index, columns=['Naive', 'AR(1)', 'LTF', 'ESN', 'AR(1)+ESN'])
store_err = pd.DataFrame(data=None, index=data.index, columns=['Naive', 'AR(1)', 'LTF', 'ESN', 'AR(1)+ESN'])

# Initialize weights for combined forecast
w_ar1 = 0.5
w_esn = 0.5

# Run a simple loop to get the signals and the strategy returns
for i in np.arange(0, nobs-ini_wind, 1):
    if set_rolling:
        z_i = data.iloc[i:(i+ini_wind)]
    else:
        z_i = data.iloc[:(i+ini_wind)]
    # Careful in defining the y and x features for the network
    y = z_i[set_depvar].values.reshape(-1, 1)
    # Drop the dependent and the contemporaneous explanatory
    xd = z_i.drop(columns=[set_depvar, set_expvar])
    x = xd.values
    # You must transpose then!
    y = y.T
    x = x.T
    # Training is infrequent!!
    if (i%train_every) == 0:
        Wout, X, Win, W, alpha, xstar = esn_train(x, y, set_resSize, set_seed, set_alpha, set_initNobs)
    # Get the x-features for prediction and forecast, here you have to be careful!!!
    set_flags = set_lags-1
    xf = pd.DataFrame(data=None, index=z_i.index, columns=[])
    for ell in set_flags:
        lags_ell = z_i[[set_depvar, set_expvar]].shift(periods=ell)
        xf = pd.concat([xf, lags_ell], axis=1)
    xf = np.hstack([xf.iloc[-1,:].values.reshape(1, xf.shape[1]), basis_raw[i+ini_wind,:].reshape(1, -1)])
    # and you must transpose here too!
    xf = xf.T
    esn = esn_predict(Wout, X, Win, W, alpha, xstar, xf)[0][0]
    # Predict with an AR(1) model
    ar1_fit = AutoReg(y.T, lags=1).fit()
    beta = ar1_fit.params
    ar1_res = ar1_fit.resid.reshape(1, -1)
    ar1 = ar1_fit.predict(start=len(y.T), end=len(y.T))[0]
    # Linear transfer function
    beta = solve( np.dot(x, x.T) , np.dot(x, y.T) ).T
    lf = beta@xf
    # Compute the forecast errors and the forecasts
    ya = data[set_depvar].iloc[i+ini_wind]
    ef = ya - esn
    el = ya - lf[0]
    eb0 = ya - y[:,-1][0]
    eb1 = ya - ar1
    ebc = ya - w_esn*esn - w_ar1*ar1
    #
    store_frc.iloc[i+ini_wind, :] = np.hstack([y[:,-1][0], ar1, lf[0], esn, w_esn*esn + w_ar1*ar1])
    store_err.iloc[i+ini_wind, :] = np.hstack([eb0, eb1, el, ef, ebc])
    # Update the combination weights
    if i >= 10:
        w_ar1 = np.exp(-store_err['AR(1)'].iloc[:(i+ini_wind+1)].apply(lambda x: x ** 2).mean())
        w_esn = np.exp(-store_err['ESN'].iloc[:(i+ini_wind+1)].apply(lambda x: x ** 2).mean())
        w_sum = w_ar1 + w_esn
        w_ar1 = w_ar1/w_sum
        w_esn = w_esn/w_sum

# Prepare for evaluation
store_err = store_err.dropna()
store_frc = store_frc.dropna()
store_frc.iloc[np.where(store_frc == 0.0)] = np.nan

# Compute descriptive measures of forecasting performance
mse = store_err.apply(lambda x: x**2).mean().apply(np.sqrt)
mae = store_err.apply(lambda x: x.abs()).mean()
print(mse/mse[0])
print(mae/mae[0])
#
store_act = store_frc + store_err
store_act.iloc[np.where(store_act == 0.0)] = np.nan
# and also the predictive precision and predictive loss
syhat = (store_err/store_frc).apply(lambda x: x**2).mean()
syact = (store_err/store_act).apply(lambda x: x**2).mean()
Rs = syact/syhat
Prec = 1/np.sqrt(1000*Rs + 1)
print(Prec/0.0316)
print(Prec.apply(np.log))

# Next repeat these measures on a rolling manner, using the same rolling window as ini_wind
mse_tv = (store_err ** 2).rolling(window=ini_wind).mean()
mae_tv = (store_err.abs()).rolling(window=ini_wind).mean()
#
mse_tv = (mse_tv.div(mse_tv['Naive'], axis='index')).dropna()
mae_tv = (mae_tv.div(mae_tv['Naive'], axis='index')).dropna()
#
syhat_tv = ((store_err/store_frc) ** 2).rolling(window=ini_wind).mean().dropna()
syact_tv = ((store_err/store_act) ** 2).rolling(window=ini_wind).mean().dropna()
Rs_tv = syact_tv/syhat_tv
Prec_tv = 1/np.sqrt(1000*Rs_tv + 1)
Ploss_tv = Prec_tv.apply(np.log)
# do some nice plots - I hard coded the details only for the first pair!!!
mse_tv.plot(title='Relative MSE based on a rolling window of 60 months for Advance Retail Sales and Inflation', xlabel='Date', ylabel='Relative MSE', grid='both')
mae_tv.plot()
Prec_tv.plot()
Ploss_tv.plot(title='Predictive Loss based on a rolling window of 60 months for Advance Retail Sales and Inflation', xlabel='Date', ylabel='Relative MSE', grid='both')
plt.show()

# Model-selection based on predictive loss
sel_Ploss = Ploss_tv.apply(np.argmax, axis=1)
store_err1 = store_err.loc[sel_Ploss.index]
store_frc1 = store_frc.loc[sel_Ploss.index]
store_act1 = store_act.loc[sel_Ploss.index]
sel_err = pd.DataFrame(data=None, index=sel_Ploss.index, columns=['Best'])
sel_frc = pd.DataFrame(data=None, index=sel_Ploss.index, columns=['Best'])
#
for i in np.arange(0, sel_Ploss.shape[0]-1, 1):
    mi = sel_Ploss.iloc[i]
    sel_err.iloc[i+1] = store_err1.iloc[i+1, mi]
    sel_frc.iloc[i+1] = store_frc1.iloc[i+1, mi]
# Recompute the performance measures, easy, so I do this only for the MSE and predictive loss!
sel_err = sel_err.dropna()
store_err1 = store_err1.loc[sel_err.index]
store_err1_all = pd.concat([store_err1, sel_err], axis=1)
mse_all = (store_err1_all ** 2).mean().apply(np.sqrt)
mse_all = (mse_all/mse_all[0])
print(mse_all)
#
store_frc1 = store_frc.loc[sel_err.index]
store_act1 = store_act.loc[sel_err.index]
store_frc1_all = pd.concat([store_frc1, sel_frc], axis=1)
store_act1_all = pd.concat([store_act1, store_act1['Naive']], axis=1)
store_act1_all.columns = store_frc1_all.columns
syhat1 = (store_err1_all/store_frc1_all).apply(lambda x: x**2).mean()
syact1 = (store_err1_all/store_act1_all).apply(lambda x: x**2).mean()
Rs1 = syact1/syhat1
Prec1 = 1/np.sqrt(1000*Rs1 + 1)
print(Prec1/0.0316)
print(Prec1.apply(np.log))
