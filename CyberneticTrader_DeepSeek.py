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
from scipy.stats import entropy, zscore
from scipy.fft import rfft, rfftfreq, irfft
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

class CyberneticTrader:
    def __init__(self, data, initial_capital=100000):
        """
        Initialize the cybernetic trading system

        Parameters:
        data : pandas.DataFrame
            Market data with columns: ['open', 'high', 'low', 'close', 'volume']
        initial_capital : float
            Starting capital for trading
        """
        self.data = data.copy()
        self.capital = initial_capital
        self.position = 0
        self.portfolio_value = [initial_capital]
        self.weights = np.array([0.4, 0.3, 0.1, 0.2])  # [trend, volatility, information, entropy]
        self.learning_rate = 0.01
        self.noise_variance = None
        self.initialize_system()

    def initialize_system(self):
        """Preprocess data and initialize system parameters"""
        # Calculate returns
        self.data['returns'] = self.data['close'].pct_change()

        # Calculate volatility
        self.data['volatility'] = self.data['returns'].rolling(20).std()

        # Estimate noise variance
        self.estimate_noise_variance()

        # Apply Wiener filter to closing prices
        self.data['filtered_close'] = self.wiener_filter(self.data['close'].values)

        # Calculate information metrics
        self.data['information'] = self.data['returns'].rolling(50).apply(
            lambda x: self.compute_market_information(x.dropna())
        )

    def estimate_noise_variance(self, window=20):
        """Estimate noise variance using residuals from linear trend"""
        returns = self.data['returns'].dropna().values
        X = np.arange(len(returns)).reshape(-1, 1)
        y = returns.reshape(-1, 1)

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        self.noise_variance = np.var(residuals)

    def compute_market_information(self, returns, bins=20):
        """
        Quantify market information content using Wiener-Shannon entropy principles

        Parameters:
        returns : array-like
            Series of price returns
        bins : int
            Number of bins for probability distribution

        Returns:
        information : float
            Normalized information metric (0-1 scale)
        """
        hist, _ = np.histogram(returns, bins=bins)
        prob = hist / np.sum(hist)
        market_entropy = entropy(prob)

        # Normalized information metric
        max_entropy = np.log(bins)
        information = 1 - (market_entropy / max_entropy) if max_entropy > 0 else 0

        return information

    def wiener_filter(self, signal):
        """
        Implement Wiener's optimal filtering for extracting true signal from noise

        Parameters:
        signal : array-like
            Input time series (e.g., closing prices)

        Returns:
        filtered : array-like
            Filtered time series
        """
        if self.noise_variance is None or self.noise_variance <= 0:
            return signal

        F = rfft(signal)
        frequencies = rfftfreq(len(signal))

        # Power spectral density (signal)
        S_signal = np.abs(F)**2 / len(signal)

        # Avoid division by zero
        S_signal = np.clip(S_signal, 1e-10, None)

        # Frequency response (Wiener filter)
        H = S_signal / (S_signal + self.noise_variance)

        # Apply filter
        filtered = irfft(F * H, n=len(signal))
        return filtered

    def detect_feedback_regime(self, window=30):
        """
        Identify homeostatic (mean-reverting) vs anti-homeostatic (trending) regimes

        Returns:
        regime : str
            Current market regime classification
        feedback_strength : float
            Strength of the feedback mechanism
        """
        if len(self.data) < window + 2:
            return "undefined", 0.0

        returns = self.data['returns'].iloc[-window:].dropna()

        if len(returns) < 2:
            return "undefined", 0.0

        autocorr = returns.autocorr(lag=1)
        volatility = returns.std()

        # Classify regime
        if abs(autocorr) < 0.2:
            regime = "homeostatic"
        elif autocorr > 0.3:
            regime = "anti-homeostatic (trending)"
        elif autocorr < -0.3:
            regime = "anti-homeostatic (mean-reverting)"
        else:
            regime = "transitional"

        return regime, autocorr

    def prevent_overreaction(self, signal, threshold=2.5):
        """
        Implement Wiener's 'purpose tremor' concept to prevent overreaction

        Parameters:
        signal : array-like
            Input signal series
        threshold : float
            Z-score threshold for clipping

        Returns:
        stabilized : array-like
            Stabilized signal with extreme values clipped
        """
        z_scores = zscore(signal)
        stabilized = np.where(np.abs(z_scores) > threshold, np.nan, signal)

        # Linear interpolation to fill clipped values
        stabilized = pd.Series(stabilized).interpolate().values
        return stabilized

    def detect_anomalies(self, window=50, threshold=3):
        """
        Detect information-theoretic anomalies (Maxwell Demon-like behaviors)

        Returns:
        anomaly_flag : bool
            Whether an anomaly is detected at current timestep
        """
        if len(self.data) < window:
            return False

        information = self.data['information'].dropna()
        if len(information) < window:
            return False

        current_info = information.iloc[-1]
        z_score = (current_info - np.mean(information)) / np.std(information)
        return abs(z_score) > threshold

    def extract_features(self, i):
        """Extract cybernetic features for decision-making"""
        if i < 50:  # Need sufficient data
            return np.zeros(4)

        # Feature 1: Trend strength (filtered price slope)
        prices = self.data['filtered_close'].iloc[i-20:i]
        X = np.arange(len(prices)).reshape(-1, 1)
        model = LinearRegression().fit(X, prices)
        trend_strength = model.coef_[0]

        # Feature 2: Volatility
        volatility = self.data['volatility'].iloc[i]

        # Feature 3: Market information
        information = self.data['information'].iloc[i]

        # Feature 4: Entropy-based stability
        returns = self.data['returns'].iloc[i-20:i].dropna()
        hist, _ = np.histogram(returns, bins=10)
        prob = hist / np.sum(hist)
        market_entropy = entropy(prob)

        # Normalize features
        features = np.array([trend_strength, volatility, information, market_entropy])
        return features

    def decide(self, features, current_price):
        """
        Make trading decision based on cybernetic features

        Parameters:
        features : array-like
            Cybernetic feature vector
        current_price : float
            Current market price

        Returns:
        action : str
            Trading decision ('BUY', 'SELL', 'HOLD')
        """
        # Linear combination of features
        score = np.dot(features, self.weights)

        # Action thresholding
        if score > 0.1 and self.capital > 0:
            return "BUY", 0.1  # Use 10% of capital
        elif score < -0.1 and self.position > 0:
            return "SELL", 0.1  # Sell 10% of position
        else:
            return "HOLD", 0

    def update_weights(self, reward, features):
        """
        Update decision weights based on performance feedback

        Parameters:
        reward : float
            Performance reward (positive or negative)
        features : array-like
            Feature vector at decision time
        """
        # Gradient-based weight adjustment
        gradient = reward * features
        self.weights += self.learning_rate * gradient

        # Normalize weights to maintain stability
        self.weights /= np.sum(np.abs(self.weights)) + 1e-8

    def backtest(self, start_idx=50, transaction_cost=0.0005):
        """
        Backtest the cybernetic trading strategy

        Parameters:
        start_idx : int
            Index to start backtesting (need sufficient warm-up period)
        transaction_cost : float
            Percentage transaction cost per trade

        Returns:
        portfolio_history : list
            Historical portfolio values
        trade_log : list
            Record of all trades executed
        """
        portfolio_history = []
        trade_log = []

        for i in range(start_idx, len(self.data)):
            current_price = self.data['close'].iloc[i]
            features = self.extract_features(i)

            # Get trading decision
            action, amount = self.decide(features, current_price)

            # Execute trade
            if action == "BUY" and self.capital > 0:
                trade_amount = min(self.capital * amount, self.capital)
                shares = trade_amount / current_price
                cost = trade_amount * transaction_cost
                self.position += shares
                self.capital -= (trade_amount + cost)
                trade_log.append({
                    'timestamp': self.data.index[i],
                    'action': 'BUY',
                    'shares': shares,
                    'price': current_price,
                    'cost': cost
                })

            elif action == "SELL" and self.position > 0:
                sell_shares = min(self.position * amount, self.position)
                sale_value = sell_shares * current_price
                cost = sale_value * transaction_cost
                self.position -= sell_shares
                self.capital += (sale_value - cost)
                trade_log.append({
                    'timestamp': self.data.index[i],
                    'action': 'SELL',
                    'shares': sell_shares,
                    'price': current_price,
                    'cost': cost
                })

            # Calculate portfolio value
            position_value = self.position * current_price
            portfolio_value = self.capital + position_value
            self.portfolio_value.append(portfolio_value)
            portfolio_history.append(portfolio_value)

            # Calculate reward (percentage change in portfolio value)
            if len(self.portfolio_value) > 1:
                prev_value = self.portfolio_value[-2]
                current_value = self.portfolio_value[-1]
                reward = (current_value - prev_value) / prev_value
                self.update_weights(reward, features)

        # Convert trade log to DataFrame
        trade_log_df = pd.DataFrame(trade_log)

        return portfolio_history, trade_log_df

    def simulate(self, live_data_handler=None, steps=100):
        """
        Run live trading simulation

        Parameters:
        live_data_handler : function
            Function that returns latest market data
        steps : int
            Number of steps to simulate
        """
        if live_data_handler is None:
            print("No live data handler provided. Using historical data for simulation.")
            return self.backtest()

        portfolio_history = []

        for _ in range(steps):
            # Get live data
            new_data = live_data_handler()
            self.data = pd.concat([self.data, new_data])
            self.data = self.data.iloc[-1000:]  # Keep recent data

            # Update system
            self.initialize_system()

            # Make decision
            i = len(self.data) - 1
            current_price = self.data['close'].iloc[i]
            features = self.extract_features(i)
            action, amount = self.decide(features, current_price)

            # Execute trade (in simulation)
            # In a real system, this would connect to brokerage API
            print(f"Action: {action} at price: {current_price:.2f}")

            # Update portfolio (simulated)
            position_value = self.position * current_price
            portfolio_value = self.capital + position_value
            self.portfolio_value.append(portfolio_value)
            portfolio_history.append(portfolio_value)

            # Calculate reward and update weights
            if len(self.portfolio_value) > 1:
                prev_value = self.portfolio_value[-2]
                current_value = self.portfolio_value[-1]
                reward = (current_value - prev_value) / prev_value
                self.update_weights(reward, features)

        return portfolio_history

    def generate_signals(self):
        """Generate trading signals for the entire dataset"""
        signals = []
        for i in range(len(self.data)):
            if i < 50:  # Warm-up period
                signals.append(0)
                continue

            features = self.extract_features(i)
            current_price = self.data['close'].iloc[i]
            action, _ = self.decide(features, current_price)

            # Convert action to signal
            if action == "BUY":
                signals.append(1)
            elif action == "SELL":
                signals.append(-1)
            else:
                signals.append(0)

        self.data['signal'] = signals
        return self.data

    def evaluate_performance(self):
        """Evaluate strategy performance metrics"""
        if 'signal' not in self.data.columns:
            self.generate_signals()

        # Calculate strategy returns
        self.data['strategy_returns'] = self.data['signal'].shift(1) * self.data['returns']

        # Calculate cumulative returns
        self.data['cumulative_market'] = (1 + self.data['returns']).cumprod()
        self.data['cumulative_strategy'] = (1 + self.data['strategy_returns']).cumprod()

        # Performance metrics
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown = self.calculate_max_drawdown()

        return {
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_return': self.data['cumulative_strategy'].iloc[-1]
        }

    def calculate_sharpe_ratio(self, risk_free_rate=0.0):
        """Calculate Sharpe ratio for strategy returns"""
        excess_returns = self.data['strategy_returns'].dropna() - risk_free_rate
        return excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative_returns = self.data['cumulative_strategy'].dropna()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

# Load market data
import yfinance as yf

# Download historical data
data = yf.download('TNA', start='2022-01-01', end='2025-06-30')
data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
data.columns = ['open', 'high', 'low', 'close', 'volume']

# Initialize cybernetic trader
trader = CyberneticTrader(data, initial_capital=100000)

# Run backtest
portfolio_history, trade_log = trader.backtest()

# Evaluate performance
performance = trader.evaluate_performance()
print(f"Strategy Performance:")
print(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {performance['max_drawdown']:.2%}")
print(f"Final Return: {performance['final_return']:.2f}x")

# Generate trading signals
signals = trader.generate_signals()

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 8))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['close'], label='Price')
buy_signals = signals[signals['signal'] == 1]
sell_signals = signals[signals['signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='g', label='Buy')
plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='r', label='Sell')
plt.title('Cybernetic Trading Signals')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(portfolio_history, label='Portfolio Value')
plt.title('Portfolio Performance')
plt.xlabel('Time')
plt.ylabel('Value ($)')
plt.tight_layout()
plt.show()