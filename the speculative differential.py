#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-speculative-differential/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# Import the packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Getting data
ticker = 'TNA'
freq = '1wk'
data = yf.download(ticker, period='max', interval=freq)['Adj Close'].dropna()
rets = data.pct_change().dropna()
start_from = '2011-01-01'
rets = rets.loc[start_from:]
nobs = rets.shape[0]

# Initialize adaptive errors
pe_adt = 0.5
pr_adt = 0.5
pe_std = 0.5
pr_std = 0.5

# Adaptive gammas initializations
gamma_e = 0.1
gamma_r = 0.1

# Window range
roll_range = np.arange(3, 27, 1)

# Saving of total excess returns
total_er = pd.DataFrame(data=None, index=['me', 'mr', 'pe-avg', 'pr-avg', 'pe-avg-B', 'pr-avg-B', 'pe-adt', 'pr-adt', 'pe-adt-B', 'pr-adt-B', 'pe-std', 'pr-std', 'pe-std-B', 'pr-std-B'], columns=roll_range)

# Saving top performing model
top_perf = pd.DataFrame(data=np.zeros([14, 1]), index=['me', 'mr', 'pe-avg', 'pr-avg', 'pe-avg-B', 'pr-avg-B', 'pe-adt', 'pr-adt', 'pe-adt-B', 'pr-adt-B', 'pe-std', 'pr-std', 'pe-std-B', 'pr-std-B'], columns=['X'])

# The training and evaluation loop
for set_roll in roll_range:
    # Initialize storage
    probs = pd.DataFrame(data=np.zeros([nobs, 8]), index=rets.index, columns=['pe-adt', 'pr-adt', 'pe-avg', 'pr-avg', 'pe-std', 'pr-std', 'alpha-e', 'alpha-r'])
    store = pd.DataFrame(data=np.zeros([nobs, 15]), index=rets.index, columns=[ticker, 'me', 'mr', 'pe-avg', 'pr-avg', 'pe-avg-B', 'pr-avg-B', 'pe-adt', 'pr-adt', 'pe-adt-B', 'pr-adt-B', 'pe-std', 'pr-std', 'pe-std-B', 'pr-std-B'])
    store_std_errors = pd.DataFrame(data=np.zeros([nobs, 4]), index=rets.index, columns=['ee_avg', 'er_avg', 'ee_std', 'er_std'])

    # Get things in a nice expanding loop, that includes the rolling computations
    for i in np.arange(set_roll, nobs, 1):
        # Split the data
        xe = rets.iloc[:i]
        xr = rets.iloc[(i-set_roll):i]
        # And get the future return
        f = rets.iloc[i]

        # Compute the new probabilities
        ye = (xe <= 0)
        yr = (xr <= 0)
        pe_avg = ye.mean()
        pr_avg = yr.mean()

        # Compute the updated alpha
        ee_adt = ye.iloc[-1] - pe_adt
        er_adt = yr.iloc[-1] - pr_adt
        #
        alpha_e = (ee_adt ** 2)/(pe_avg + (ee_adt ** 2))
        alpha_r = (er_adt ** 2)/(pr_avg + (er_adt ** 2))
        if np.isnan(alpha_r.to_numpy()):
            alpha_r = 1

        # Scale them
        alpha_e2 = (alpha_e ** 2)/(2*(alpha_e ** 2) - 2*alpha_e + 1)
        alpha_r2 = (alpha_r ** 2)/(2*(alpha_r ** 2) - 2*alpha_r + 1)

        # Compute the updated probabilities
        pe_adt = alpha_e2*pe_avg + (1-alpha_e2)*(ee_adt ** 2)
        pr_adt = alpha_r2*pr_avg + (1-alpha_r2)*(er_adt ** 2)

        # and the standard adaptive learning ones
        ee_std = ye.iloc[-1] - pe_std
        ee_avg = ye.iloc[-1] - pe_avg
        er_std = yr.iloc[-1] - pr_std
        er_avg = yr.iloc[-1] - pr_avg
        # store them
        store_std_errors.iloc[i-1] = np.hstack([ee_avg, er_avg, ee_std, er_std])
        # After some observations estimate gamma
        if i > 10:
            zi = store_std_errors.iloc[:i]
            A = (zi['ee_std']*(zi['ee_std'].shift(periods=1))).dropna().mean()
            B = (zi['ee_avg']*(zi['ee_std'].shift(periods=1))).dropna().mean()
            C = (zi['ee_std'].iloc[:-1] ** 2).mean()
            if (C == 0):
                C = 1.0
            gamma_e = (A - B)/C
            A = (zi['er_std']*(zi['er_std'].shift(periods=1))).dropna().mean()
            B = (zi['er_avg']*(zi['er_std'].shift(periods=1))).dropna().mean()
            C = (zi['er_std'].iloc[:-1] ** 2).mean()
            if (C == 0):
                C = 1.0
            gamma_r = (A - B)/C
        #
        pe_std = pe_avg + gamma_e*ee_std
        pr_std = pr_avg + gamma_r*er_std

        # Store the probabilities
        probs.iloc[i,:] = np.hstack([pe_adt, pr_adt, pe_avg, pr_avg, pe_std, pr_std, alpha_e2, alpha_r2])

        if np.isnan(pr_adt.to_numpy()):
            raise ValueError

        # Trade
        mu_e = xe.mean()
        mu_r = xr.mean()
        tme = f*np.sign(mu_e)
        tmr = f*np.sign(mu_r)
        te0 = f*np.sign(1-2*pe_avg)
        tr0 = f*np.sign(1-2*pr_avg)
        te1 = f*np.sign(mu_e*(1-2*pe_avg))
        tr1 = f*np.sign(mu_r*(1-2*pr_avg))
        te2 = f*np.sign(1-2*pe_adt)
        tr2 = f*np.sign(1-2*pr_adt)
        te3 = f*np.sign(mu_e*(1-2*pe_adt))
        tr3 = f*np.sign(mu_r*(1-2*pr_adt))
        #
        te4 = f*np.sign(1-2*pe_std)
        tr4 = f*np.sign(1-2*pr_std)
        te5 = f*np.sign(mu_e*(1-2*pe_std))
        tr5 = f*np.sign(mu_r*(1-2*pr_std))
        # Store the trading returns
        store.iloc[i] = np.hstack([f, tme, tmr, te0, tr0, te1, tr1, te2, tr2, te3, tr3, te4, tr4, te5, tr5])

    # Compute average probabilities
    avg_probs = probs.dropna().mean()

    # Some plotting, you can do the rest!
    store = store.iloc[set_roll:]
    tr = ((store + 1).cumprod()-1)*100
    # tr.plot()
    # plt.show()
    er = tr.iloc[-1,1:] - tr.iloc[-1,0]
    total_er[set_roll] = er
    top_perf.loc[er.index[er.argmax()]] = top_perf.loc[er.index[er.argmax()]] + 1
    #
    # Uncomment below to see detailed results
    # if any(er > 0):
    #     print('Rolling window = '+str(set_roll))
    #     print('Top-model is: '+er.index[er.argmax()]+' with total ER = '+str(round(er.iloc[er.argmax()],3)))
    #     top_perf.loc[er.index[er.argmax()]] = top_perf.loc[er.index[er.argmax()]] + 1
    #     print('Average probabilities = '+str(np.round(avg_probs.to_numpy().T,3)))
    #     print(er.sort_values(ascending=False))
    #     print('-------------')

# Compute total percentage of predictions leading to positive excess returns
psuccess = (total_er[total_er > 0] > 0).sum().sum()/(total_er.shape[0]*total_er.shape[1])

# and percentage of the top performing models
topsuccess = top_perf/top_perf.sum()

# and finally the average top performance across rolling windows
maxperf = total_er[total_er > 0].apply(lambda x: x.max())

# Print nicely
print('Ticker analyzed is: '+ticker)
print('Total success='+str(round(psuccess*100, 3))+'%')
print('Top 3 models = '+str(round(topsuccess.sort_values(by='X', ascending=False)[:3]*100, 3)))
print('Minimum total excess return = '+str(round(maxperf.min(),3))+'%')
print('Average total excess return = '+str(round(maxperf.mean(),3))+'%')
print('Average total excess return = '+str(round(maxperf.median(),3))+'%')
print('Maximum total excess return = '+str(round(maxperf.max(),3))+'%')
