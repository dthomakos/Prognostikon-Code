#
# Python code replicating results on this post:
#
# https://prognostikon.wordpress.com/2023/12/31/trend-extraction-with-echo-state-networks/
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

# Get the US real GDP data
#
data = pdr.get_data_fred(['GDPC1', 'USREC'], start='1940-01-01', end='2023-07-31').dropna()
y = data['GDPC1'].apply(np.log).values
nobs = y.shape[0]
set_initNobs = 80
xdates = data.index[set_initNobs:]
rdates = data['USREC'].iloc[set_initNobs:]

# Get the Chebyshev basis
cheb_degree = 2
x = Chebyshev_basis(cheb_degree, nobs)

# Get the parameters of the ESN
min_resSize = 3
max_resSize = 60
set_resSize = np.arange(min_resSize, max_resSize+1, 1)
set_seed = 10
set_alpha = 0.5

# Transpose once for the fitting
y = y.reshape(1, -1)
x = x.T

# Fit the ESN in a loop over the resolution size
#
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

# Find optimal resolution size, re-estimate and plot
bic_resSize = int(bic[bic[:,1].argmin(), 0])
aic_resSize = int(aic[aic[:,1].argmin(), 0])
osc_resSize = int(np.median([bic_resSize, aic_resSize]))

# Fit for all three criteria
fit_bic, Wout_bic = esn_fit(x, y, bic_resSize, set_seed, set_alpha, set_initNobs)
fit_aic, Wout_aic = esn_fit(x, y, aic_resSize, set_seed, set_alpha, set_initNobs)
fit_osc, Wout_osc = esn_fit(x, y, osc_resSize, set_seed, set_alpha, set_initNobs)

# transpose again for plotting, note that you must remove initial observations
yplot = y[:,set_initNobs:].T
fit_bic = fit_bic[:,set_initNobs:].T
fit_aic = fit_aic[:,set_initNobs:].T
fit_osc = fit_osc[:,set_initNobs:].T

# Now for the plotting
plt.plot(xdates, yplot, label='GDPC1')
plt.plot(xdates, fit_aic, label='ESN fit')
plt.title('US real GDP and ESN-based trend (potential output)')
plt.xlabel('Date')
plt.ylabel('log of GDPC1')
plt.grid()
plt.legend()
#
plt.show()

# and a last plot of the business cycle!!
y_data = pd.DataFrame((yplot-fit_aic)*100, index=data.index[set_initNobs:], columns=['US Real Business Cycle'])
r_data = rdates
r_data.iloc[r_data == 1] = -100
fig, ax = plt.subplots()
y_data.plot.line(ax=ax, figsize=(8, 5), color="blue", title='US real GDP, deviations from ESN-based trend (potential output)')
r_data.plot.area(ax=ax, figsize=(8, 5), alpha=0.5, color="gray", ylabel='growth rate in %')
r_data.iloc[r_data == -100] = 100
r_data.plot.area(ax=ax, figsize=(8, 5), alpha=0.5, color="gray")
plt.ylim(y_data.min().values, y_data.max().values)
plt.show()


