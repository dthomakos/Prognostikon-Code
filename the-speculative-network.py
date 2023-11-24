#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/11/25/the-speculative-network/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.special import expit
from numpy import tanh

# Get the fitted/predicted values from an MLP(K, L, M) network
#
def MLP_fit(WK, WM, WO, bK, bM, bO, L, M, activation, X, y=None):
    # Get the shapes of arrays
    nobs = X.shape[0]
    K = X.shape[1]

    # Construct the main function with the input and hidden layers
    H = X
    f_all = np.zeros([L, nobs, M])
    #
    for i in range(L):
        if i == 0:
            z_i = np.dot(H, WK).reshape(nobs, M)
            b_i = np.tile(bK, nobs).reshape(nobs, M)
        else:
            z_i = np.dot(H, WM[i-1]).reshape(nobs, M)
            b_i = np.tile(bM[i-1], nobs).reshape(nobs, M)
        if activation == 'tanh':
            H = tanh(z_i + b_i)
        elif activation == 'logistic':
            H = expit(z_i + b_i)
        f_all[i] = H

    # Get the fitted values on the output layer
    if activation == 'tanh':
        fit = tanh(np.dot(H, WO) + bO)
    elif activation == 'logistic':
        fit = expit(np.dot(H, WO) + bO)

    # and the error, if this is done in-sample
    if y is not None:
        error = y - fit
    else:
        error = None

    return f_all, fit, error, WK, WM, WO, bK, bM, bO

# Train an MLP(K, L, M) network via backpropagation
#
def MLP_backpropagation(L, M, X, y, epochs=1000, learning_rate=0.1, activation='tanh', ini_random=False):
    # Get the shapes of arrays and set-up initial weights
    nobs = X.shape[0]
    K = X.shape[1]

    # Initialize the parameters
    if ini_random: # this is NOT the preferred choice, it precludes reproducibility
        iniW = np.array(np.random.uniform(low=0, high=1, size=K*M + M*M*L)).reshape(K+L*M, M)
        inib = np.array(np.random.uniform(low=0, high=1, size=(L+1)*M)).reshape(L+1, M)
    else: # the non-random choice makes much more sense!
        iniW = np.ones([K+L*M, M])*0.5
        inib = np.ones([L+1, M])*0.5
    WK = iniW[:K,:]
    WM = iniW[K:-M,:].reshape(L-1, M, M)
    WO = iniW[-M:,0].reshape(-1, 1)
    bK = inib[0,:]
    bM = inib[1:-1,:]
    bO = inib[-1,0]

    # Initialize MSE progress
    mse_progress = np.zeros([epochs, 1])

    # Loop over the epochs to train the weights
    for itr in range(epochs):
        # Get fitted values, error and weights
        f_all, fit, error, WK, WM, WO, bK, bM, b0 = MLP_fit(WK, WM, WO, bK, bM, bO, L, M, activation, X, y)

        # Calculate error metric and store progress, ensure MSE is reduced at every step
        mse_progress[itr] = (error ** 2).mean()
        if itr > 2:
            # Break from training is the MSE is not changing too much - the threshold is hard-coded!
            if mse_progress[itr] < mse_progress[itr-1]:
                if (np.abs(mse_progress[itr] - mse_progress[itr-1]) <= 1e-6):
                    break
            else: # if not, perturb slightly the initial weights and re-train
                e = np.random.uniform(low=-1, high=1, size=1)
                WK = iniW[:K,:] + e
                WM = iniW[K:-M,:].reshape(L-1, M, M) + e
                WO = iniW[-M:,0].reshape(-1, 1) + e
                bK = inib[0,:] + e
                bM = inib[1:-1,:] + e
                bO = inib[-1,0] + e
                f_all, fit, error, WK, WM, WO, bK, bM, b0 = MLP_fit(WK, WM, WO, bK, bM, bO, L, M, activation, X, y)
                # Compute the new MSE and re-evaluate - do not overtrain!!
                mse_progress[itr] = (error ** 2).mean()
                if (i > 20) and (mse_progress[itr] < mse_progress[itr-1]):
                    if (np.abs(mse_progress[itr] - mse_progress[itr-1]) <= 1e-6):
                        break

        # Backpropagation requires a loop as in that connecting fo the weights but reversed!!
        if activation == 'tanh':
            dWM = - error * (1 - (fit ** 2))
            WO_new = WO - learning_rate * (np.dot(f_all[-1].T, dWM) / nobs)
            bO_new = b0 - learning_rate * np.sum(dWM)
            dWi = dWM
            WM_new = np.zeros(WM.shape)
            bM_new = np.zeros(bM.shape)
            for i in np.arange(1, L+1, 1):
                if i == 1:
                    dWi =  - np.dot(dWi, WO.T) * (1 - (f_all[-i-1] ** 2))
                    WM_new[-i] = WM[-i] - learning_rate * (np.dot(f_all[-i-1].T, dWi) / nobs)
                    bM_new[-i] = bM[-i] - learning_rate * np.sum(dWi)
                elif (i > 1) and (i < L):
                    dWi =  - np.dot(dWi, WM[-i].T) * (1 - (f_all[-i-1] ** 2))
                    WM_new[-i] = WM[-i] - learning_rate * (np.dot(f_all[-i-1].T, dWi) / nobs)
                    bM_new[-i] = bM[-i] - learning_rate * np.sum(dWi)
                elif i == L:
                    dWi =  - np.dot(dWi, WK.T)
                    actK = np.dot(X, WK) + np.tile(bK, nobs).reshape(nobs, M)
                    WK_new = WK - learning_rate * (np.dot(actK.T, dWi) / nobs).T
                    bK_new = bK - learning_rate * np.sum(dWi)
        elif activation == 'logistic':
            dWM = - error * fit * (1 - fit)
            WO_new = WO - learning_rate * (np.dot(f_all[-1].T, dWM) / nobs)
            bO_new = b0 - learning_rate * np.sum(dWM)
            dWi = dWM
            WM_new = np.zeros(WM.shape)
            bM_new = np.zeros(bM.shape)
            for i in np.arange(1, L+1, 1):
                if i == 1:
                    dWi =  - np.dot(dWi, WO.T) * f_all[-i-1] * (1 - f_all[-i-1])
                    WM_new[-i] = WM[-i] - learning_rate * (np.dot(f_all[-i-1].T, dWi) / nobs)
                    bM_new[-i] = bM[-i] - learning_rate * np.sum(dWi)
                elif (i > 1) and (i < L):
                    dWi =  - np.dot(dWi, WM[-i].T) * f_all[-i-1] * (1 - f_all[-i-1])
                    WM_new[-i] = WM[-i] - learning_rate * (np.dot(f_all[-i-1].T, dWi) / nobs)
                    bM_new[-i] = bM[-i] - learning_rate * np.sum(dWi)
                elif i == L:
                    dWi =  - np.dot(dWi, WK.T)
                    actK = np.dot(X, WK) + np.tile(bK, nobs).reshape(nobs, M)
                    WK_new = WK - learning_rate * (np.dot(actK.T, dWi) / nobs).T
                    bK_new = bK - learning_rate * np.sum(dWi)

        # Done, re-initialize the weights and repeat training
        WO = WO_new
        WM = WM_new
        WK = WK_new
        bO = bO_new
        bM = bM_new
        bK = bK_new

    # Exit the training, and return everything
    return mse_progress, f_all, fit, error, WK, WM, WO, bK, bM, bO

# Download some data, construct the variables for the network: --> denotes the ones used for the post!!
# GLD monthly full sample (L=3, M=1, train=6) initial varies
# OIH monthly full sample (L=2, M=1, train=120) initial varies
# TNA monthly full sample (L=2, M=1, train=12) initial varies
# --> TNA daily from 2021-06-01 (L=2, M=1, train=126) initial 126, learning_rate=0.001
# --> TZA daily from 2021-06-01 (L=3, M=1, train=126) initial 126, learning_rate=0.0015
# --> SPY daily from 2021-06-01 (L=2, M=1, train=21) initial 126, learning_rate=0.001
# SPY daily from 2019-06-01 (L=2, M=5, train=126) initial 126
# --> GLD daily from 2021-06-01 (L=2, M=2, train=7) initial 126, learning_rate=0.0015
# DBA & DBC daily from 2021-06-01 (L=2, M=3, train=21) initial 126, learning_rate=0.001
# USO daily from 2021-06-01 (L=3, M=3, train=21) initial 126, learning_rate=0.001
# USO daily from 2019-06-01 (L=4, M=3, train=6) initial 126
# --> OIH daily from 2021-06-01 (L=2, M=3, train=126) initial 126, learning_rate=0.0005
# --> TLT daily from 2021-06-01 (L=3, M=2, train=126) initial 126, learning_rate=0.0025
# TLT daily from 2019-06-01 (L=3, M=4, train=126) initial 126
ticker = 'TLT'
data = yf.download(ticker, period='max', interval='1d')['Adj Close'].dropna()
r = data.pct_change().dropna().loc['2021-06-01':] # you can change this of course
r.name = ticker

# Compute the necessary variables
dr = r.diff()
z1p = ((dr > 0) & (r > 0)).astype(float)
z2p = ((dr <= 0) & (r > 0)).astype(float)
z1n = ((dr <= 0) & (r <= 0)).astype(float)
z2n = ((dr > 0) & (r <= 0)).astype(float)
# Convert the zeroes to -ones, this is important!!
z1p.iloc[z1p == 0] = -1.0
z2p.iloc[z2p == 0] = -1.0
z1n.iloc[z1n == 0] = -1.0
z2n.iloc[z2n == 0] = -1.0
# Put together
z_all = pd.concat([r, dr, z1p, z2p, z1n, z2n], axis=1).dropna()
z_all.columns = [ticker, 'D-'+ticker, 'Z1+', 'Z2+', 'Z1-', 'Z2-']

# Number of observations and initial window
nobs = z_all.shape[0]
ini_wind = 126

# Define the network's architecture
L = 3
M = 2
train_every = 126
set_learning_rate = 0.0025

# and initialize storage
store = pd.DataFrame(data=None, index=z_all.index, columns=[ticker, 'Speculative Network'])

# Run a simple loop to get the signals and the strategy returns
for i in np.arange(0, nobs-ini_wind, 1):
    z_i = z_all.iloc[:(i+ini_wind),:] # this is recursive estimation, change to i:(i+ini_wind) for rolling!!
    # Careful in defining the y and x features for the network
    y = z_i.iloc[1:,0].apply(np.sign).values.reshape(-1, 1)
    x = z_i.iloc[:-1,1:].values
    # Training is infrequent!!
    if (i%train_every) == 0:
        mse_progress, f_all, fit, error, WK, WM, WO, bK, bM, bO = \
        MLP_backpropagation(L, M, x, y, epochs=1500, learning_rate=set_learning_rate, activation='tanh')
    # Get the x-features for prediction and forecast
    xf = z_i.iloc[-1, 1:].values.reshape(1, 5)
    f_all, fit, error, WK, WM, WO, bK, bM, bO = MLP_fit(WK, WM, WO, bK, bM, bO, L, M, 'tanh', xf, y=None)
    # The sign for the prediction is computed relative to the mean of the dependent variable
    if fit < y.mean():
        z_f = -1
    else:
        z_f = +1
    # and finally we get the strategy returns!
    bnh = z_all.iloc[i+ini_wind, 0]
    stg = bnh*z_f
    store.iloc[i+ini_wind, :] = np.hstack([bnh, stg])

# Compute the cumulative return and plot
cret = ((store + 1).cumprod() - 1)*100
cret.plot(grid='both', title='The speculative network strategy for '+ticker, xlabel='Date', ylabel='return in percent')
#plt.savefig(ticker+'.png')
plt.show()
#
print(cret.iloc[-1])
