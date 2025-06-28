#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-inferential-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
# -------------------------------------------------------------------------------------

import numpy as np
import yfinance as yf
from collections import defaultdict
import matplotlib.pyplot as plt

class CoxProbabilisticTrader_V2:
    """
    Unified Cox-style probabilistic trading system with:
    - Momentum, Sharpe, and Utility binary features
    - Bayesian belief updating without normal distributions
    - Ensemble backtesting (Monte Carlo)
    - Entropy-based risk management
    """

    def __init__(self, lookback_window=30, entropy_threshold=0.2, prior_belief=0.5,
                 gamma=0.75, momentum_threshold=0.0, sharpe_threshold=1.0,
                 utility_threshold=0.0, trade_threshold=0.7):
        # Trading parameters
        self.lookback = lookback_window
        self.entropy_threshold = entropy_threshold
        self.prior = prior_belief
        self.gamma = gamma
        self.momentum_threshold = momentum_threshold
        self.sharpe_threshold = sharpe_threshold
        self.utility_threshold = utility_threshold
        self.trade_threshold = trade_threshold

        # State tracking
        self.probability_map = defaultdict(lambda: prior_belief)
        self.entropy_history = []
        self.portfolio_log = []
        self.market_states = []
        self.returns = None

    def compute_entropy(self, p):
        """Compute Shannon entropy for a binary probability"""
        if p == 0 or p == 1:
            return 0
        q = 1 - p
        return - (p * np.log2(p) + q * np.log2(q))

    def update_belief(self, evidence, market_condition):
        """Update belief using Bayes' theorem with binary evidence"""
        prior = self.probability_map[market_condition]
        posterior = (prior * evidence) / (prior * evidence + (1 - prior) * (1 - evidence))
        self.probability_map[market_condition] = posterior

    def process_market_data(self, prices):
        """Compute features from price window"""
        returns = np.diff(prices) / prices[:-1]
        window = returns[-self.lookback:]

        # Calculate features
        momentum = np.prod(1 + window) - 1
        sharpe = np.mean(window) / np.std(window) if np.std(window) > 0 else 0
        utility = np.mean(window) - self.gamma * np.std(window)

        # Create binary features
        features = {
            'momentum': 1 if momentum > self.momentum_threshold else 0,
            'sharpe': 1 if sharpe > self.sharpe_threshold else 0,
            'utility': 1 if utility > self.utility_threshold else 0
        }

        # Create market state
        market_state = self._state_from_features(features)
        self.market_states.append(market_state)
        return market_state, features

    def _state_from_features(self, features):
        """Create market state from binary features"""
        return f"{features['momentum']}{features['sharpe']}{features['utility']}"

    def _calculate_evidence(self, features):
        """Calculate evidence as fraction of bullish features"""
        bullish_count = sum(features.values())
        return bullish_count / len(features)

    def trading_signal(self, current_market):
        """Generate trading signal based on market probability and entropy"""
        p = self.probability_map[current_market]
        market_entropy = self.compute_entropy(p)
        self.entropy_history.append(market_entropy)

        if market_entropy < self.entropy_threshold:
            return "BUY" if p > self.trade_threshold else "SELL"
        return "HOLD"

    def risk_assessment(self):
        """Assess risk using entropy-price covariance"""
        if len(self.portfolio_log) < 2 or len(self.entropy_history) < 2:
            return 0.5

        price_changes = np.diff(self.portfolio_log) / self.portfolio_log[:-1]
        ent = np.array(self.entropy_history[1:])
        pc = price_changes

        min_len = min(len(ent), len(pc))
        if min_len < 2:
            return 0.5

        covariance = np.cov(ent[:min_len], pc[:min_len])[0, 1]
        return np.clip(np.tanh(covariance * 10) - 1, -1, 1)

    def _execute_trade(self, signal, current_position):
        """Execute trade based on signal and risk assessment"""
        risk_factor = self.risk_assessment()
        if signal == 'BUY':
            return risk_factor
        elif signal == 'SELL':
            return -risk_factor
        else:
            return current_position

    def reset_logs(self):
        """Reset internal logs and state"""
        self.entropy_history = []
        self.portfolio_log = []
        self.market_states = []
        self.probability_map = defaultdict(lambda: self.prior)

    def backtest(self, historical_prices, n_ensembles=1, seed=42):
        """Run ensemble backtest with Monte Carlo simulation"""
        historical_prices = np.asarray(historical_prices).flatten()
        if len(historical_prices) <= self.lookback + 1:
            raise ValueError("Insufficient data for backtest")

        np.random.seed(seed)
        ensemble_results = []

        for _ in range(n_ensembles):
            perturbed_prices = self._generate_ensemble_member(historical_prices) if n_ensembles > 1 else historical_prices
            result = self._single_backtest(perturbed_prices)
            ensemble_results.append(result)

        return self._analyze_ensemble(ensemble_results)

    def _generate_ensemble_member(self, prices):
        """Generate synthetic price series for Monte Carlo simulation"""
        prices = np.asarray(prices).flatten()
        if len(prices) < 2:
            return prices.copy()

        returns = np.diff(prices) / prices[:-1]
        vol = np.std(returns)
        synthetic_returns = np.random.normal(0, vol, len(prices)-1)
        synthetic_prices = np.cumprod(np.insert(1 + synthetic_returns, 0, prices[0]))
        return synthetic_prices

    def _single_backtest(self, historical_prices):
        """Run single backtest instance"""
        self.reset_logs()
        portfolio_value = 1.0
        position = 0.0

        for i in range(self.lookback, len(historical_prices)):
            window = historical_prices[i - self.lookback:i]
            market_state, features = self.process_market_data(window)
            new_evidence = self._calculate_evidence(features)
            self.update_belief(new_evidence, market_state)
            signal = self.trading_signal(market_state)
            position = self._execute_trade(signal, position)

            if historical_prices[i-1] != 0:
                daily_return = (historical_prices[i] - historical_prices[i-1]) / historical_prices[i-1]
            else:
                daily_return = 0

            portfolio_value *= (1 + position * daily_return)
            self.portfolio_log.append(portfolio_value)

        return self._analyze_performance()

    def _analyze_performance(self):
        """Analyze single backtest performance"""
        if len(self.portfolio_log) < 2:
            return {
                'cumulative_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'portfolio_log': self.portfolio_log.copy()
            }

        returns = np.diff(self.portfolio_log) / self.portfolio_log[:-1]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0
        roll_max = np.maximum.accumulate(self.portfolio_log)
        drawdowns = (roll_max - self.portfolio_log) / roll_max
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

        return {
            'cumulative_return': self.portfolio_log[-1] - 1,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'portfolio_log': self.portfolio_log.copy()
        }

    def _analyze_ensemble(self, results):
        """Aggregate results from ensemble backtests"""
        returns = [r['cumulative_return'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        max_drawdowns = [r['max_drawdown'] for r in results]

        success_prob = np.mean(np.array(returns) > 0)
        hist, _ = np.histogram(returns, bins=20)
        prob_dist = hist / np.sum(hist) if np.sum(hist) > 0 else np.ones_like(hist) / len(hist)
        return_entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))

        return {
            'success_probability': success_prob,
            'expected_return': np.mean(returns),
            'mean_sharpe': np.mean(sharpe_ratios),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'return_entropy': return_entropy,
            'sample_portfolio_log': results[0]['portfolio_log'] if results else []
        }

# Real Data Example
if __name__ == "__main__":
    # Initialize trader with parameters
    trader = CoxProbabilisticTrader_V2(
        lookback_window=7,
        entropy_threshold=0.1,
        prior_belief=0.5,
        gamma=0.75,
        momentum_threshold=-0.03,
        sharpe_threshold=0.5,
        utility_threshold=0,
        trade_threshold=0.5
    )

    # Download historical data for S&P 500 ETF (SPY)
    data = yf.download("SPY", start="2022-01-01", end="2025-06-01")
    prices = data['Close'].values

    # Run backtest
    results = trader.backtest(prices, n_ensembles=1)

    # Display results
    print(f"Success Probability: {results['success_probability']:.2%}")
    print(f"Expected Return: {results['expected_return']:.2%}")
    print(f"Mean Sharpe Ratio: {results['mean_sharpe']:.2f}")
    print(f"Mean Max Drawdown: {results['mean_max_drawdown']:.2%}")
    print(f"Return Entropy: {results['return_entropy']:.4f}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(results['sample_portfolio_log'], label="Cox Portfolio Value")
    plt.title("Cox Probabilistic Trading System Backtest")
    plt.xlabel("Trading Days")
    plt.ylabel("Portfolio Value (Normalized)")
    plt.legend()
    plt.grid(True)
    plt.show()