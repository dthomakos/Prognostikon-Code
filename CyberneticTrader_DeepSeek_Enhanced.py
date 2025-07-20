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
from scipy.stats import entropy, zscore, kurtosis
from scipy.fft import rfft, rfftfreq, irfft
from scipy.special import eval_laguerre
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from pyinform.transferentropy import transfer_entropy
import networkx as nx
import heapq

class CyberneticTrader:
    def __init__(self, data, initial_capital=100000,
                 # System parameters
                 weights=None, meta_weights=None, learning_rate=0.01,
                 # Feature calculation parameters
                 trend_window=20, entropy_bins=10, information_bins=20,
                 entropy_calc_bins=20, noise_est_window=20,
                 # Filter parameters
                 volterra_order=2, volterra_memory=10,
                 # Agent parameters
                 perceptor_memory_size=100, effector_threshold=0.1,
                 homeostat_k=0.5, homeostat_alpha=0.01, homeostat_beta=0.1,
                 # Trading parameters
                 transaction_cost=0.0005, start_idx=50,
                 # Thermodynamic parameters
                 te_window=20, te_k=2,
                 # Circuit breaker parameters
                 entropy_threshold=0.9, drawdown_threshold=0.2, kurtosis_threshold=5,
                 # Feedback parameters
                 overreaction_threshold=2.5, anomaly_window=50, anomaly_threshold=3,
                 # Morphological parameters
                 morph_window=50, morph_threshold=0.001,
                 # Feedback regime parameters
                 feedback_window=30, autocorr_threshold=0.2):
        """
        Initialize the enhanced cybernetic trading system with configurable parameters
        """
        self.data = data.copy()
        self.capital = initial_capital
        self.position = 0
        self.portfolio_value = [initial_capital]

        # System parameters
        self.weights = weights if weights is not None else np.array([0.4, 0.3, 0.1, 0.2])
        self.meta_weights = meta_weights if meta_weights is not None else np.array([0.01, 0.01])
        self.learning_rate = learning_rate
        self.noise_variance = None
        self.market_entropy_history = []
        self.performance_history = []

        # Feature calculation parameters
        self.trend_window = trend_window
        self.entropy_bins = entropy_bins
        self.information_bins = information_bins
        self.entropy_calc_bins = entropy_calc_bins
        self.noise_est_window = noise_est_window

        # Filter parameters
        self.volterra_order = volterra_order
        self.volterra_memory = volterra_memory

        # Trading parameters
        self.transaction_cost = transaction_cost
        self.start_idx = start_idx

        # Thermodynamic parameters
        self.te_window = te_window
        self.te_k = te_k

        # Circuit breaker parameters
        self.entropy_threshold = entropy_threshold
        self.drawdown_threshold = drawdown_threshold
        self.kurtosis_threshold = kurtosis_threshold

        # Feedback parameters
        self.overreaction_threshold = overreaction_threshold
        self.anomaly_window = anomaly_window
        self.anomaly_threshold = anomaly_threshold

        # Morphological parameters
        self.morph_window = morph_window
        self.morph_threshold = morph_threshold

        # Feedback regime parameters
        self.feedback_window = feedback_window
        self.autocorr_threshold = autocorr_threshold

        # Create agent ecosystem with configurable parameters
        self.agents = {
            'perceptor': PerceptorAgent(memory_size=perceptor_memory_size),
            'effector': EffectorAgent(decision_threshold=effector_threshold),
            'homeostat': HomeostatAgent(k=homeostat_k, alpha=homeostat_alpha, beta=homeostat_beta)
        }

        self.initialize_system()

    def initialize_system(self):
        """Preprocess data and initialize system parameters"""
        # Calculate returns
        self.data['returns'] = self.data['close'].pct_change()

        # Calculate volatility
        self.data['volatility'] = self.data['returns'].rolling(20).std()

        # Estimate noise variance
        self.estimate_noise_variance(window=self.noise_est_window)

        # Apply Wiener filter to closing prices
        self.data['filtered_close'] = self.wiener_filter(self.data['close'].values)

        # Apply Volterra filter
        self.data['volterra_close'] = self.volterra_filter(
            self.data['close'].values,
            order=self.volterra_order,
            memory=self.volterra_memory
        )

        # Calculate information metrics
        self.data['information'] = self.data['returns'].rolling(50).apply(
            lambda x: self.compute_market_information(x.dropna(), bins=self.information_bins)
        )

        # Calculate market entropy
        self.data['entropy'] = self.data['returns'].rolling(50).apply(
            lambda x: self.compute_entropy(x.dropna(), bins=self.entropy_calc_bins)
        )

    def estimate_noise_variance(self, window=20):
        """Estimate noise variance using residuals from linear trend"""
        returns = self.data['returns'].dropna().values
        if len(returns) < window:
            self.noise_variance = 0.01
            return

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
        """
        if len(returns) < 10:
            return 0.5  # Neutral information value

        hist, _ = np.histogram(returns, bins=bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]  # Remove zero probabilities

        market_entropy = entropy(prob)

        # Normalized information metric
        max_entropy = np.log(bins)
        information = 1 - (market_entropy / max_entropy) if max_entropy > 0 else 0

        return information

    def compute_entropy(self, returns, bins=20):
        """Compute raw entropy value for thermodynamic constraints"""
        if len(returns) < 5:
            return 1.0

        hist, _ = np.histogram(returns, bins=bins)
        prob = hist / np.sum(hist)
        prob = prob[prob > 0]
        return entropy(prob)

    def wiener_filter(self, signal):
        """
        Implement Wiener's optimal filtering for extracting true signal from noise
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

    def volterra_filter(self, signal, order=2, memory=10):
        """
        Nonlinear Volterra filter implementation based on Wiener's work
        """
        n = len(signal)
        filtered = np.zeros(n)

        # First-order (linear) component
        for i in range(memory, n):
            filtered[i] = np.sum(signal[i-memory:i])

        # Second-order nonlinear component
        if order >= 2:
            for i in range(2*memory, n):
                for j in range(memory):
                    for k in range(j, memory):
                        filtered[i] += signal[i-j] * signal[i-k]

        # Normalize
        filtered = (filtered - np.mean(filtered)) / np.std(filtered)
        return filtered * np.std(signal) + np.mean(signal)

    # ================== CYBERNETIC FUNCTIONS ================== #

    def second_order_learning(self, reward, features):
        """
        Meta-learning of learning parameters based on performance
        """
        # Update meta-weights based on performance gradient
        meta_gradient = reward * np.array([self.learning_rate, np.mean(features)])
        self.meta_weights += 0.001 * meta_gradient

        # Constrain to reasonable values
        self.meta_weights = np.clip(self.meta_weights, 0.001, 0.1)

        # Update learning parameters
        self.learning_rate = self.meta_weights[0]
        self.entropy_threshold = 0.5 + 0.4 * (1 / (1 + np.exp(-self.meta_weights[1])))

    def distributed_decision(self, features, current_price):
        """
        Make trading decisions using in-memory agent ecosystem
        """
        # Process features with perceptor
        perception = self.agents['perceptor'].process(features)

        # Decide with effector
        action, amount = self.agents['effector'].decide(perception, current_price)

        # Update homeostat
        self.agents['homeostat'].update(action)

        return action, amount

    def thermodynamic_constraint(self):
        """
        Implement Maxwell Demon-inspired position sizing
        """
        # Compute transfer entropy between volume and returns
        returns = self.data['returns'].dropna().iloc[-self.te_window:].values
        volume = self.data['volume'].pct_change().dropna().iloc[-self.te_window:].values

        if len(returns) < 10 or len(volume) < 10:
            return 0.1

        te = transfer_entropy(volume, returns, k=self.te_k)

        # Constrain position size
        max_position = min(1.0, te * 10)
        return max_position

    def anti_fragile_feedback(self, volatility):
        """
        Adaptive gain control to prevent overreaction
        """
        # Dynamic adjustment based on volatility
        k = 0.5 * np.exp(-0.5 * volatility * 100)
        b = 0.1 * (1 + 2 * volatility)
        return k / (1 + b)

    def morphological_computation(self):
        """
        Analyze market structure using topological methods
        """
        returns = self.data['returns'].dropna().iloc[-self.morph_window:].values
        if len(returns) < 10:
            return 0.5

        # Create recurrence network
        G = nx.Graph()
        for i in range(len(returns)):
            for j in range(i+1, len(returns)):
                if abs(returns[i] - returns[j]) < self.morph_threshold:
                    G.add_edge(i, j)

        # Compute topological persistence
        if len(G.nodes) > 0:
            persistence = nx.algorithms.components.number_connected_components(G) / len(G.nodes)
            return persistence
        return 0.5

    def circuit_breaker(self):
        """
        Cybernetic circuit breaker for extreme market conditions
        """
        if len(self.portfolio_value) < 10:
            return False

        # Calculate current drawdown
        current_value = self.portfolio_value[-1]
        peak_value = max(self.portfolio_value)
        drawdown = (peak_value - current_value) / peak_value

        # Check entropy and kurtosis
        current_entropy = self.data['entropy'].iloc[-1] if not pd.isna(self.data['entropy'].iloc[-1]) else 0.5
        kurt = self.data['returns'].tail(50).kurtosis()
        kurt = kurt if not pd.isna(kurt) else 0

        # Check thresholds
        if (current_entropy > self.entropy_threshold and
            drawdown > self.drawdown_threshold and
            kurt > self.kurtosis_threshold):
            return True
        return False

    def detect_feedback_regime(self):
        """
        Identify market feedback regimes
        """
        if len(self.data) < self.feedback_window + 2:
            return "undefined", 0.0

        returns = self.data['returns'].iloc[-self.feedback_window:].dropna()

        if len(returns) < 2:
            return "undefined", 0.0

        autocorr = returns.autocorr(lag=1)
        volatility = returns.std()

        # Classify regime
        if abs(autocorr) < self.autocorr_threshold:
            regime = "homeostatic"
        elif autocorr > 0.3:
            regime = "anti-homeostatic (trending)"
        elif autocorr < -0.3:
            regime = "anti-homeostatic (mean-reverting)"
        else:
            regime = "transitional"

        return regime, autocorr

    def prevent_overreaction(self, signal):
        """
        Implement 'purpose tremor' concept to prevent overreaction
        """
        z_scores = zscore(signal)
        stabilized = np.where(np.abs(z_scores) > self.overreaction_threshold, np.nan, signal)
        stabilized = pd.Series(stabilized).interpolate().values
        return stabilized

    def detect_anomalies(self):
        """
        Detect information-theoretic anomalies
        """
        if len(self.data) < self.anomaly_window:
            return False

        information = self.data['information'].dropna()
        if len(information) < self.anomaly_window:
            return False

        current_info = information.iloc[-1]
        z_score = (current_info - np.mean(information)) / np.std(information)
        return abs(z_score) > self.anomaly_threshold

    def extract_features(self, i):
        """Extract cybernetic features for decision-making"""
        if i < 50:  # Need sufficient data
            return np.zeros(4)

        # Feature 1: Trend strength
        prices = self.data['filtered_close'].iloc[i-self.trend_window:i]
        X = np.arange(len(prices)).reshape(-1, 1)
        model = LinearRegression().fit(X, prices)
        trend_strength = model.coef_[0]

        # Feature 2: Volatility
        volatility = self.data['volatility'].iloc[i]

        # Feature 3: Market information
        information = self.data['information'].iloc[i] if not pd.isna(self.data['information'].iloc[i]) else 0.5

        # Feature 4: Entropy-based stability
        returns = self.data['returns'].iloc[i-self.trend_window:i].dropna()
        if len(returns) > 5:
            hist, _ = np.histogram(returns, bins=self.entropy_bins)
            prob = hist / np.sum(hist)
            market_entropy = entropy(prob)
        else:
            market_entropy = 1.0

        # Normalize features
        features = np.array([trend_strength, volatility, information, market_entropy])
        return features

    def update_weights(self, reward, features):
        """
        Update decision weights based on performance feedback
        """
        # Gradient-based weight adjustment
        gradient = reward * features
        self.weights += self.learning_rate * gradient

        # Normalize weights to maintain stability
        self.weights /= np.sum(np.abs(self.weights)) + 1e-8

    def backtest(self):
        """
        Backtest the enhanced cybernetic trading strategy
        """
        portfolio_history = []
        trade_log = []

        for i in range(self.start_idx, len(self.data)):
            # Check circuit breaker
            if self.circuit_breaker():
                trade_log.append({
                    'timestamp': self.data.index[i],
                    'action': 'HALT',
                    'shares': 0,
                    'price': self.data['close'].iloc[i],
                    'cost': 0,
                    'reason': 'Market instability'
                })
                portfolio_history.append(self.portfolio_value[-1])
                continue

            current_price = self.data['close'].iloc[i]
            features = self.extract_features(i)

            # Get trading decision
            action, amount = self.distributed_decision(features, current_price)

            # Apply thermodynamic constraint
            max_amount = self.thermodynamic_constraint()
            amount = min(amount, max_amount)

            # Execute trade
            trade_result = self.execute_trade(
                action,
                amount,
                current_price,
                self.transaction_cost,
                timestamp=self.data.index[i]
            )
            if trade_result:
                trade_log.append(trade_result)

            # Update portfolio value
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
                self.second_order_learning(reward, features)

        # Convert trade log to DataFrame
        trade_log_df = pd.DataFrame(trade_log)

        return portfolio_history, trade_log_df

    def execute_trade(self, action, amount, current_price, transaction_cost, timestamp):
        """Execute a trade with proper accounting"""
        trade_record = None

        if action == "BUY" and self.capital > 0:
            trade_amount = min(self.capital * amount, self.capital)
            shares = trade_amount / current_price
            cost = trade_amount * transaction_cost
            self.position += shares
            self.capital -= (trade_amount + cost)
            trade_record = {
                'timestamp': timestamp,
                'action': 'BUY',
                'shares': shares,
                'price': current_price,
                'cost': cost
            }

        elif action == "SELL" and self.position > 0:
            sell_shares = min(self.position * amount, self.position)
            sale_value = sell_shares * current_price
            cost = sale_value * transaction_cost
            self.position -= sell_shares
            self.capital += (sale_value - cost)
            trade_record = {
                'timestamp': timestamp,
                'action': 'SELL',
                'shares': sell_shares,
                'price': current_price,
                'cost': cost
            }

        return trade_record

    def simulate(self, live_data_handler=None, steps=100):
        """
        Run live trading simulation
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
            action, amount = self.distributed_decision(features, current_price)

            # Execute trade (in simulation)
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
                self.second_order_learning(reward, features)

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
            action, _ = self.distributed_decision(features, current_price)

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
        return excess_returns.mean() / (excess_returns.std() + 1e-8)

    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative_returns = self.data['cumulative_strategy'].dropna()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()

# ================== AGENT CLASSES ================== #
class PerceptorAgent:
    """Specialized in market perception and feature extraction"""
    def __init__(self, memory_size=100):
        self.feature_memory = []
        self.memory_size = memory_size

    def process(self, features):
        """Process features with memory and attention"""
        # Store features
        self.feature_memory.append(features)
        if len(self.feature_memory) > self.memory_size:
            self.feature_memory.pop(0)

        # Apply temporal attention
        weights = np.exp(0.1 * np.arange(len(self.feature_memory)))
        weights /= np.sum(weights)

        # Weighted historical features
        weighted_features = np.zeros_like(features)
        for i, f in enumerate(self.feature_memory):
            weighted_features += weights[i] * f

        return weighted_features

class EffectorAgent:
    """Specialized in executing trading decisions"""
    def __init__(self, decision_threshold=0.1, buy_amount=0.1, sell_amount=0.1):
        self.decision_threshold = decision_threshold
        self.volatility_factor = 1.0
        self.buy_amount = buy_amount
        self.sell_amount = sell_amount

    def decide(self, perception, current_price):
        """Make trading decision based on perception"""
        score = np.mean(perception)
        self.volatility_factor = max(0.5, min(2.0, 1.0 / current_price))
        threshold = self.decision_threshold * self.volatility_factor

        if score > threshold:
            return "BUY", self.buy_amount
        elif score < -threshold:
            return "SELL", self.sell_amount
        else:
            return "HOLD", 0

class HomeostatAgent:
    """Maintains system stability and adapts parameters"""
    def __init__(self, k=0.5, target_k=0.5, alpha=0.01, beta=0.1,
                 hold_adjust=-0.01, trade_adjust=0.05):
        self.k = k
        self.target_k = target_k
        self.alpha = alpha
        self.beta = beta
        self.hold_adjust = hold_adjust
        self.trade_adjust = trade_adjust

    def update(self, action):
        """Update parameters based on market feedback"""
        if action != "HOLD":
            self.target_k = max(0.1, min(0.9, self.target_k + self.trade_adjust))
        else:
            self.target_k = max(0.1, min(0.9, self.target_k + self.hold_adjust))

        # Homeostatic adjustment
        self.k = self.k - self.alpha * (self.k - self.target_k) + self.beta * np.random.randn() * 0.1

# Example usage
if __name__ == "__main__":
    import yfinance as yf
    import matplotlib.pyplot as plt

    # Download historical data
    data = yf.download('SPY', start='2010-01-01', end='2020-12-31')
    data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    data.columns = ['open', 'high', 'low', 'close', 'volume']

    # Initialize trader with custom parameters
    trader = CyberneticTrader(
        data,
        initial_capital=100000,
        # System parameters
        weights=np.array([0.35, 0.25, 0.2, 0.2]),
        learning_rate=0.005,
        # Feature parameters
        trend_window=7,
        entropy_bins=12,
        # Trading parameters
        transaction_cost=0.0000,
        start_idx=40,
        # Circuit breaker
        entropy_threshold=0.85,
        drawdown_threshold=0.15,
        # Agents
        effector_threshold=0.08,
        perceptor_memory_size=80
    )

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