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

# Let us now simulate some data
#
# Select stationary or non-stationary (integrated) noise
is_stationary = True

# Initial observations, to be discarded in the ESN estimation
set_initNobs = 100
# Proper observatoins
nobs = 300+set_initNobs

# Create the time component
deg_time = 2
if is_stationary:
    params_time = (1, 3, 0.5, -0.5, 1.5)
    time = np.arange(1, nobs+1, 1)/nobs
else:
    params_time = (1, 0.015, 0.0005, -0.0005, 0.00025)
    time = np.arange(1, nobs+1, 1)
trend = np.zeros([nobs, deg_time+1])
for i in range(deg_time+1):
    trend[:, i] = params_time[i]*(time ** i)

# Create the cycle component
deg_cycle = 2
params_cycle0 = ([0.2, 0.1], [1.2, -0.8])
params_cycle1 = ([1.2, 5.2], [5.2, 1.2])
cycle = np.zeros([nobs, deg_cycle])
for i in range(deg_cycle):
    cycle[:, i] = params_cycle0[i][0]*np.sin(params_cycle1[i][0]*np.pi*time) + params_cycle0[i][1]*np.cos(params_cycle1[i][1]*np.pi*time)

# Add noise
sigma = 0.5
noise = np.random.normal(loc=0, scale=sigma, size=(nobs, ))

# Create the model and the dependent variable
if is_stationary:
    model = trend.sum(axis=1) + cycle.sum(axis=1)
    y = model + noise
    data_type = 'trend stationary'
else:
    model = trend.sum(axis=1)
    y = model + noise.cumsum()
    data_type = 'non-stationary, I(1)'

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
    modelplot = model[set_initNobs:]

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
modelplot = model[set_initNobs:]

# Now for the plotting
fig, axs = plt.subplots(1, 2, sharex=True)
fig.suptitle('ESN trend+cycle extraction: when does it work?!')
axs[0].plot(yplot, label='Data that are '+data_type)
axs[0].plot(fit_aic, label='AIC')
axs[0].set_title('Simulated data and ESN fit based on the AIC')
axs[0].grid()
axs[0].legend()
#
axs[1].plot(modelplot, label='Model')
axs[1].plot(fit_bic, label='BIC')
axs[1].plot(fit_aic, label='AIC')
axs[1].plot(fit_osc, label='OSC')
axs[1].set_title('True model vs. ESN fit')
axs[1].legend()
axs[1].grid()
#
plt.show()
