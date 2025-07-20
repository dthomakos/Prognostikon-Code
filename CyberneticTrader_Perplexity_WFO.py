#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-cybernetic-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
# -------------------------------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from scipy.signal import wiener
import yfinance as yf
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------------------
# Function & Class definition(s)
# -------------------------------------------------------------------------------------

def force_flat_series(arr, index=None, name=None):
    """Ensure input is a 1D pandas Series with specified index and name."""
    if isinstance(arr, pd.DataFrame):
        arr = arr.iloc[:,0]
    elif isinstance(arr, np.ndarray):
        arr = arr.flatten()
        if index is None:
            index = np.arange(len(arr))
        arr = pd.Series(arr, index=index, name=name)
    elif isinstance(arr, pd.Series):
        arr = arr.squeeze()
        if index is not None:
            arr = arr.reindex(index)
        if name is not None:
            arr.name = name
        return arr
    else:
        arr = pd.Series(arr, index=index, name=name)
    return arr

class CyberneticTradingSystem:
    def __init__(
        self,
        price_series,
        feature_df,
        initial_capital=100_000,
        trading_fee=0.001,
        entropy_window=20,
        entropy_bins=10,
        return_lag=1,
        rolling_window=10,
        lookback=30,
        num_features=3,
        mi_bins=5,
        wiener_noise=0.1,
        feedback_kp=1.0,
        learn_rate=0.01,
        learn_epochs=1,
        trade_threshold=0.0
    ):
        # Core data
        self.prices = force_flat_series(price_series)  # 1D Series

        feats_clean = {}
        for col in feature_df.columns:
            feats_clean[col] = force_flat_series(feature_df[col], index=feature_df.index, name=col)
        self.features = pd.DataFrame(feats_clean)

        # Parameters to store for reuse
        self.initial_capital = initial_capital
        self.trading_fee = trading_fee
        self.entropy_window = entropy_window
        self.entropy_bins = entropy_bins
        self.return_lag = return_lag
        self.rolling_window = rolling_window
        self.lookback = lookback
        self.num_features = num_features
        self.mi_bins = mi_bins
        self.wiener_noise = wiener_noise
        self.feedback_kp = feedback_kp
        self.learn_rate = learn_rate
        self.learn_epochs = learn_epochs
        self.trade_threshold = trade_threshold

        self.history = []
        self.backtest_result = None

    def compute_entropy(self, series, window=None, bins=None):
        # Use self defaults unless explicitly supplied
        window = window or self.entropy_window
        bins = bins or self.entropy_bins
        s1d = force_flat_series(series)
        arr = s1d.values
        ent = np.full((len(arr),), np.nan)
        for i in range(window, len(arr)):
            windowed = arr[i-window:i]
            if np.all(np.isnan(windowed)):
                continue
            hist, _ = np.histogram(windowed[~np.isnan(windowed)], bins=bins, density=True)
            hist = hist[hist > 0]
            if len(hist)==0: ent[i]=np.nan
            else: ent[i] = -np.sum(hist * np.log2(hist))
        return pd.Series(ent, index=s1d.index, name="entropy")

    def compute_mutual_information(self, X, y, bins=None):
        y = force_flat_series(y, index=X.index)
        df = X.copy()
        df['__y__'] = y
        df = df.dropna()
        X2 = df.drop(columns='__y__')
        y2 = df['__y__']
        mi = mutual_info_regression(X2, y2, discrete_features=False)
        return pd.Series(mi, index=X2.columns).sort_values(ascending=False)

    def wiener_signal(self, signal, noise_var=None):
        s1d = force_flat_series(signal)
        noise_var = self.wiener_noise if noise_var is None else noise_var
        filtered = wiener(s1d.values, mysize=5, noise=noise_var)
        return pd.Series(filtered, index=s1d.index, name=s1d.name)

    def feedback_controller(self, desired, actual, k_p=None):
        k_p = self.feedback_kp if k_p is None else k_p
        error = desired - actual
        return k_p * error

    def adaptive_weights(self, X, y, eta=None, epochs=None):
        X = np.array(X)
        y = np.array(y).flatten()
        eta = self.learn_rate if eta is None else eta
        epochs = self.learn_epochs if epochs is None else epochs
        w = np.zeros(X.shape[1])
        for _ in range(epochs):
            for xi, target in zip(X, y):
                pred = np.dot(xi, w)
                error = target - pred
                w += eta * error * xi
        return w

    def backtest(self, lookback=None, threshold=None):
        lookback = lookback if lookback is not None else self.lookback
        threshold = threshold if threshold is not None else self.trade_threshold

        df = self.features.copy()
        df['price'] = self.prices
        df = df.dropna()
        portfolio = []
        positions = []
        cash = self.initial_capital
        position = 0
        self.history.clear()

        for t in range(lookback, len(df)):
            idx = df.index[t]
            window_feat = df.iloc[t-lookback:t, :-1]
            window_target = df['price'].iloc[t-lookback:t].pct_change().dropna()
            # Alignment fix
            if window_feat.shape[0] != window_target.shape[0]:
                minlen = min(window_feat.shape[0], window_target.shape[0])
                window_feat = window_feat.iloc[-minlen:]
                window_target = window_target.iloc[-minlen:]

            mi = self.compute_mutual_information(window_feat, window_target)
            top_features = mi.index[:self.num_features]
            signals_list = []
            for col in top_features:
                filtered = self.wiener_signal(window_feat[col])
                signals_list.append(filtered.values[-1])
            signals = np.array(signals_list)

            weights = self.adaptive_weights(window_feat[top_features], window_target)
            pred_move = np.dot(signals, weights)
            trade_signal = self.feedback_controller(threshold, pred_move)
            next_price = df['price'].iloc[t]
            last_price = df['price'].iloc[t-1]
            trade_size = np.sign(trade_signal)
            trade_cost = abs(trade_size - position) * self.trading_fee * next_price

            cash -= trade_cost
            pnl = position * (next_price - last_price)
            cash += pnl
            position = trade_size
            portfolio.append(cash + position * next_price)
            positions.append(position)
            self.history.append(dict(
                time=idx, position=position, cash=cash,
                signal=pred_move, trade_signal=trade_signal,
                top_features=list(top_features),
                weights=weights.copy(),
                pnl=pnl, portfolio=portfolio[-1])
            )
        self.backtest_result = pd.DataFrame(self.history).set_index('time')
        return self.backtest_result

    def performance_report(self):
        equity = self.backtest_result['portfolio']
        returns = equity.pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-9) * np.sqrt(252)
        max_dd = (equity / equity.cummax() - 1).min()
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        return {
            'Total Return': total_return,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Final Equity': equity.iloc[-1]
        }



class CyberneticWFO:
    def __init__(self, trader_class, price_series, feature_df,
                 train_period=252*2, oos_period=126, step=126, # About 2y train, 6m OOS, step=6m
                 **trader_kwargs):
        """
        trader_class: The trading system class (passed as a class, not an instance).
        price_series, feature_df: The full price/feature data.
        train_period: Number of days in training (in-sample) per WFO segment.
        oos_period: Number of days in out-of-sample (walk-forward) test per segment.
        step: Forward jump after each OOS segment (may overlap between WFO cycles).
        trader_kwargs: All additional trading system hyperparameters.
        """
        self.trader_class = trader_class
        self.price_series = price_series
        self.feature_df  = feature_df
        self.train_period = train_period
        self.oos_period = oos_period
        self.step = step
        self.trader_kwargs = trader_kwargs

        self.segments = []       # (train_idx, oos_idx) pairs
        self.results = []        # Each OOS result, as DataFrame
        self.param_history = []  # Store selected parameters/metrics per segment

    def split_walk_forward(self):
        """
        Prepare rolling WFO intervals.
        """
        idx = self.feature_df.index  # This is the synchronized index already
        n = len(idx)
        splits = []
        t0 = 0
        while True:
            train_start = t0
            train_end = train_start + self.train_period
            oos_start = train_end
            oos_end = min(oos_start + self.oos_period, n)
            # Enough room for a full OOS after train?
            if oos_end > n or train_end > n:
                break
            splits.append((train_start, train_end, oos_start, oos_end))
            t0 += self.step
        return splits

    def run(self, verbose=True):
        self.segments = self.split_walk_forward()
        prev_equity = None
        prev_cash = None
        for i, (t0, t1, t2, t3) in enumerate(self.segments):
            if verbose:
                print(f"WFO Segment {i+1}/{len(self.segments)}: "
                      f"IN [{t0}:{t1}], OOS [{t2}:{t3}]")

            # Fit on training set (can use it to tune params in future)
            train_price = self.price_series.iloc[t0:t1]
            train_feat  = self.feature_df.iloc[t0:t1]
            test_price  = self.price_series.iloc[t2:t3]
            test_feat   = self.feature_df.iloc[t2:t3]

            trader = self.trader_class(
                price_series=pd.concat([train_price, test_price]),
                feature_df=pd.concat([train_feat, test_feat]),
                **self.trader_kwargs
            )

            # Fit model on "train" then backtest only on "OOS"
            res = trader.backtest(lookback=self.trader_kwargs.get("lookback", 30))
            # Only take OOS portion for equity curve/performance
            res_oos = res.loc[test_feat.index].copy()
            # For 'rolling equity', hand over last capital/equity from segment to next (optional)
            if prev_equity is not None:
                delta = res_oos['portfolio'] - res_oos['portfolio'].iloc[0] + prev_equity
                res_oos['portfolio'] = delta
            prev_equity = res_oos['portfolio'].iloc[-1]
            # May also update 'cash' for next period if you want strict continuity

            self.results.append(res_oos)
            self.param_history.append({"segment":i,"performance":trader.performance_report()})

        # Concatenate all OOS periods
        full_result = pd.concat(self.results, axis=0)
        return full_result

    def aggregate_performance(self):
        """Aggregate OOS metrics."""
        perf_df = pd.DataFrame(self.param_history)
        perf = pd.DataFrame(list(perf_df["performance"])).mean().to_dict()
        return perf

# -------------------------------------------------------------------------------------
# A simple real world example
# -------------------------------------------------------------------------------------

symbol = 'SPY'
data = yf.download(symbol, start="2022-01-01", end="2025-06-30", progress=False)
close_prices = force_flat_series(data['Close'])

set_rwind = 5
returns = close_prices.pct_change()
rmean = close_prices.rolling(set_rwind).mean()
rstd = close_prices.rolling(set_rwind).std()
feature_df = pd.DataFrame({'returns': returns, 'rmean': rmean, 'rstd': rstd})

# Instantiate class first with chosen params
cyber_trader = CyberneticTradingSystem(
    price_series=close_prices,
    feature_df=feature_df,
    initial_capital=200_000,
    trading_fee=0.0005,
    entropy_window=20,
    entropy_bins=10,
    rolling_window=10,
    lookback=30,
    num_features=3,
    mi_bins=5,
    wiener_noise=0.1,
    feedback_kp=2.0,
    learn_rate=0.0005,
    learn_epochs=3,
    trade_threshold=0.02
)

# Compute entropy as a feature (if desired)
cyber_trader.features["entropy"] = cyber_trader.compute_entropy(close_prices)

# Drop any NaNs before running
cyber_trader.features = cyber_trader.features.dropna()
cyber_trader.prices = cyber_trader.prices.loc[cyber_trader.features.index]

# Backtest and report
result = cyber_trader.backtest()
print(cyber_trader.performance_report())

result['portfolio'].plot()
plt.title("Cybernetic Trading Equity Curve")
plt.show()

# -------------------------------------------------------------------------------------
# A simple WFO example
# -------------------------------------------------------------------------------------

wfo = CyberneticWFO(
    trader_class=CyberneticTradingSystem,
    price_series=close_prices,
    feature_df=feature_df,
    train_period=252,   # 2 years in-sample
    oos_period=63,        # 1 quarter OOS
    step=63,              # Advance by 1 quarter per WFO run
    # any trading system hyperparameters here...
    lookback=21,
    num_features=3,
    feedback_kp=2.0,
    learn_rate=0.005,
)

wfo_result = wfo.run(verbose=True)
print("\n=== Aggregate OOS performance (WFO) ===")
print(wfo.aggregate_performance())
wfo_result['portfolio'].plot(title="Cybernetic Trading System Walk-Forward OOS Equity")
