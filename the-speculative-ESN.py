#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2024/01/05/the-echo-speculative-network/
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
import yfinance as yf

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

# Now for the analysis, for the post I used the same parametrizations for all assets examined!
ticker = 'TNA'
data = yf.download(ticker, period='max', interval='1d')['Adj Close'].dropna()
r = data.pct_change().dropna().loc['2023-06-02':] # you can change this of course
r.name = ticker

# Number of observations and initial window
nobs = r.shape[0]
ini_wind = 21

# Add the Chebyshev basis as well
set_deg = 4
basis = Chebyshev_basis(set_deg, nobs)

# Define the network's architecture
set_resSize = np.arange(14, 21, 1)
set_seed = 22
set_alpha = 0.5
set_initNobs = 14
train_every = 5

# Use signs only on the dependent variable?
use_signs = True

# and initialize storage
store = pd.DataFrame(data=None, index=r.index, columns=[ticker, 'Speculative ESN'])

# Run a simple loop to get the signals and the strategy returns
for i in np.arange(0, nobs-ini_wind, 1):
    z_i = r.iloc[:(i+ini_wind)] # this is recursive estimation, change to i:(i+ini_wind) for rolling!!
    basis_i = basis[:(i+ini_wind),:]
    # Careful in defining the y and x features for the network
    if use_signs:
        y = z_i.iloc[1:].apply(np.sign).values.reshape(-1, 1)
    else:
        y = z_i.iloc[1:].values.reshape(-1, 1)
    x = basis_i[:-1,:]
    # You must transpose then!
    y = y.T
    x = x.T
    # Training is infrequent!!
    if (i%train_every) == 0:
        aic_resSize, bic_resSize = esn_model_selection(x, y, set_resSize, set_seed, set_alpha, set_initNobs)
        Wout, X, Win, W, alpha, xstar = esn_train(x, y, aic_resSize, set_seed, set_alpha, set_initNobs)
    # Get the x-features for prediction and forecast
    xf = basis_i[-1,:].reshape(1, set_deg+1)
    # and you must transpose here too!
    xf = xf.T
    fit = esn_predict(Wout, X, Win, W, alpha, xstar, xf)[0][0]
    # and finally we get the strategy returns!
    z_f = np.sign(fit)
    bnh = r.iloc[i+ini_wind]
    stg = bnh*z_f
    store.iloc[i+ini_wind, :] = np.hstack([bnh, stg])

# Compute the cumulative return and plot
cret = ((store.dropna() + 1).cumprod() - 1)*100
cret.plot(grid='both', title='The speculative ESN strategy for '+ticker, xlabel='Date', ylabel='return in percent', color=['black', 'green'])
plt.savefig(ticker+'-ESN.png')
plt.show()
#
print(cret.iloc[-1])
