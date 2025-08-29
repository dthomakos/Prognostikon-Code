#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-gpt/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
# -------------------------------------------------------------------------------------

#!/usr/bin/env python3
"""
Enhanced Gaussian Process Trading System with Chebyshev Polynomial Basis Functions

This module implements a sophisticated trading system based on Gaussian Process regression
using Chebyshev polynomials as basis functions. The system predicts future returns based
on lagged returns and generates trading signals.

Key improvements:
- Configurable ticker and date selection
- Multi-frequency data support (daily, weekly, monthly)
- Proper annualization factors
- Benchmark performance metrics
- Enhanced look-ahead bias protection
- Corrected feature/target alignment

Author: Dimitrios Thomakos with prompts to Claude (Anthropic)
Date: August 2025
"""

# Import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky, solve_triangular, LinAlgError
from scipy.special import eval_chebyt
from scipy.optimize import minimize
import yfinance as yf
from typing import Tuple, Optional, Dict, List
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class TradingMetrics:
    """Container for trading performance metrics."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    volatility: float
    calmar_ratio: float
    sortino_ratio: float

class BasisFunction(ABC):
    """Abstract base class for basis functions."""

    @abstractmethod
    def compute(self, x: np.ndarray) -> np.ndarray:
        """Compute basis function values."""
        pass

    @abstractmethod
    def get_num_features(self) -> int:
        """Return number of basis functions."""
        pass

class ChebyshevBasis(BasisFunction):
    """Chebyshev polynomial basis functions of the first kind."""

    def __init__(self, max_degree: int = 5, normalize: bool = True):
        self.max_degree = max_degree
        self.normalize = normalize
        self.fitted_min = None
        self.fitted_max = None

    def _normalize_input(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        """Normalize input to [-1, 1] range."""
        if not self.normalize:
            return x

        if fit or (self.fitted_min is None or self.fitted_max is None):
            self.fitted_min = np.min(x)
            self.fitted_max = np.max(x)

        if self.fitted_max == self.fitted_min:
            return np.zeros_like(x)

        return 2 * (x - self.fitted_min) / (self.fitted_max - self.fitted_min) - 1

    def compute(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        """Compute Chebyshev polynomial basis functions."""
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples, n_lags = x.shape
        n_features = self.get_num_features()

        if hasattr(self, '_expected_n_lags') and n_lags != self._expected_n_lags:
            raise ValueError(f"Expected {self._expected_n_lags} lags, got {n_lags}")

        basis_matrix = np.ones((n_samples, n_features))
        x_normalized = np.zeros_like(x)

        for lag in range(n_lags):
            x_col = x[:, lag].flatten()
            x_normalized[:, lag] = self._normalize_input(x_col, fit=fit)

        feature_idx = 0
        feature_idx += 1  # Skip constant term (already initialized)

        # Linear terms
        for lag in range(n_lags):
            if feature_idx < n_features:
                basis_matrix[:, feature_idx] = x_normalized[:, lag]
                feature_idx += 1

        # Higher order terms
        for degree in range(2, self.max_degree + 1):
            for lag in range(n_lags):
                if feature_idx < n_features:
                    x_lag = x_normalized[:, lag]
                    basis_matrix[:, feature_idx] = eval_chebyt(degree, x_lag)
                    feature_idx += 1

        # Interaction terms
        for lag1 in range(n_lags):
            for lag2 in range(lag1 + 1, n_lags):
                if feature_idx < n_features:
                    basis_matrix[:, feature_idx] = x_normalized[:, lag1] * x_normalized[:, lag2]
                    feature_idx += 1

        return basis_matrix

    def get_num_features(self) -> int:
        if not hasattr(self, '_num_features'):
            return 50
        return self._num_features

    def set_num_features(self, n_lags: int) -> None:
        """Set number of features based on number of lags."""
        features = 1  # constant
        features += n_lags  # linear terms
        features += (self.max_degree - 1) * n_lags  # higher order terms
        features += n_lags * (n_lags - 1) // 2  # interaction terms
        self._num_features = features

class GaussianProcessRegressor:
    """Gaussian Process Regressor with flexible kernel and basis functions."""

    def __init__(self,
                 basis_function: BasisFunction,
                 noise_variance: float = 1e-4,
                 prior_variance: float = 1.0,
                 jitter: float = 1e-6):
        self.basis_function = basis_function
        self.noise_variance = noise_variance
        self.prior_variance = prior_variance
        self.jitter = jitter

        # Fitted parameters
        self.X_train = None
        self.y_train = None
        self.Phi = None
        self.L = None
        self.alpha = None

        # Hyperparameter bounds
        self.bounds = {
            'noise_variance': (1e-6, 1.0),
            'prior_variance': (1e-3, 100.0)
        }

    def _compute_kernel_matrix(self, Phi: np.ndarray) -> np.ndarray:
        """Compute kernel matrix K = Φ Σ_p Φ^T + σ²_n I."""
        n_samples, n_features = Phi.shape
        K = self.prior_variance * (Phi @ Phi.T) + self.noise_variance * np.eye(n_samples)
        K += self.jitter * np.eye(n_samples)
        return K

    def _safe_cholesky(self, K: np.ndarray) -> np.ndarray:
        """Compute Cholesky decomposition with fallback strategies."""
        try:
            return cholesky(K, lower=True)
        except LinAlgError:
            logger.warning("Cholesky decomposition failed, adding more jitter")
            K_jittered = K + 1e-3 * np.eye(K.shape[0])
            try:
                return cholesky(K_jittered, lower=True)
            except LinAlgError:
                logger.warning("Using eigenvalue regularization")
                eigenvals, eigenvecs = np.linalg.eigh(K)
                eigenvals = np.maximum(eigenvals, 1e-6)
                K_reg = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
                return cholesky(K_reg, lower=True)

    def fit(self, X: np.ndarray, y: np.ndarray, optimize_hyperparams: bool = True) -> 'GaussianProcessRegressor':
        """Fit Gaussian Process to training data."""
        X = np.asarray(X)
        y = np.asarray(y).flatten()

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_samples_X, n_features_X = X.shape
        n_samples_y = len(y)

        if n_samples_X != n_samples_y:
            raise ValueError(f"X and y have incompatible shapes: X={X.shape}, y={y.shape}")

        self.X_train = X.copy()
        self.y_train = y.copy()

        if hasattr(self.basis_function, 'set_num_features'):
            self.basis_function.set_num_features(n_features_X)

        self.Phi = self.basis_function.compute(X, fit=True)

        if optimize_hyperparams:
            self._optimize_hyperparameters()

        K = self._compute_kernel_matrix(self.Phi)
        self.L = self._safe_cholesky(K)
        self.alpha = solve_triangular(self.L, y, lower=True)

        return self

    def _log_marginal_likelihood(self, hyperparams: np.ndarray) -> float:
        """Compute log marginal likelihood for hyperparameter optimization."""
        noise_var = np.exp(hyperparams[0])
        prior_var = np.exp(hyperparams[1])

        old_noise_var = self.noise_variance
        old_prior_var = self.prior_variance

        self.noise_variance = noise_var
        self.prior_variance = prior_var

        try:
            K = self._compute_kernel_matrix(self.Phi)
            L = self._safe_cholesky(K)
            alpha = solve_triangular(L, self.y_train, lower=True)
            log_likelihood = -0.5 * np.sum(alpha**2) - np.sum(np.log(np.diag(L))) - 0.5 * len(self.y_train) * np.log(2 * np.pi)
            return -log_likelihood

        except Exception as e:
            logger.warning(f"Error in likelihood computation: {e}")
            return 1e10

        finally:
            self.noise_variance = old_noise_var
            self.prior_variance = old_prior_var

    def _optimize_hyperparameters(self) -> None:
        """Optimize hyperparameters using L-BFGS-B."""
        initial_params = np.array([np.log(self.noise_variance), np.log(self.prior_variance)])

        bounds = [
            (np.log(self.bounds['noise_variance'][0]), np.log(self.bounds['noise_variance'][1])),
            (np.log(self.bounds['prior_variance'][0]), np.log(self.bounds['prior_variance'][1]))
        ]

        try:
            result = minimize(
                self._log_marginal_likelihood,
                initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500}
            )

            if result.success:
                self.noise_variance = np.exp(result.x[0])
                self.prior_variance = np.exp(result.x[1])
                logger.info(f"Optimized hyperparameters: noise_var={self.noise_variance:.6f}, prior_var={self.prior_variance:.6f}")
            else:
                logger.warning("Hyperparameter optimization failed, using initial values")

        except Exception as e:
            logger.warning(f"Hyperparameter optimization error: {e}")

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions on new data."""
        if self.L is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_test_samples = X.shape[0]
        Phi_test = self.basis_function.compute(X)

        # Compute cross-covariance
        K_star = self.prior_variance * (Phi_test @ self.Phi.T)

        # Predictions
        v = solve_triangular(self.L, K_star.T, lower=True)
        predictions = v.T @ self.alpha
        predictions = predictions.flatten()

        if not return_std:
            return predictions, None

        # Predictive variance
        K_star_star = self.prior_variance * np.sum(Phi_test**2, axis=1)
        var = K_star_star - np.sum(v**2, axis=0) + self.noise_variance
        std = np.sqrt(np.maximum(var, 1e-10))

        return predictions, std

class TradingStrategy:
    """Base class for trading strategies."""

    def __init__(self, transaction_cost: float = 0.001):
        self.transaction_cost = transaction_cost
        self.position = 0

    def generate_signal(self, prediction: float, uncertainty: Optional[float] = None) -> int:
        """Generate trading signal based on prediction."""
        if prediction > 0:
            return 1
        elif prediction < 0:
            return -1
        else:
            return 0

class GPTradingSystem:
    """Complete Gaussian Process Trading System."""

    def __init__(self,
                 n_lags: int = 10,
                 max_degree: int = 3,
                 lookback_window: int = 252,
                 min_periods: int = 60,
                 transaction_cost: float = 0.001,
                 frequency: str = 'D'):
        """
        Initialize GP Trading System.

        Parameters:
        -----------
        frequency : str
            Data frequency: 'D' (daily), 'W' (weekly), 'M' (monthly)
        """
        self.n_lags = n_lags
        self.max_degree = max_degree
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        self.transaction_cost = transaction_cost
        self.frequency = frequency.upper()

        # Set annualization factor based on frequency
        if self.frequency == 'D':
            self.annualization_factor = 252  # Trading days
        elif self.frequency == 'W':
            self.annualization_factor = 52   # Weeks
        elif self.frequency == 'M':
            self.annualization_factor = 12   # Months
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        # Initialize components
        self.basis_function = ChebyshevBasis(max_degree=max_degree, normalize=True)
        self.gp = GaussianProcessRegressor(
            basis_function=self.basis_function,
            noise_variance=1e-4,
            prior_variance=1.0
        )
        self.strategy = TradingStrategy(transaction_cost=transaction_cost)

        # Results storage
        self.predictions = []
        self.signals = []
        self.returns = []
        self.positions = []
        self.equity_curve = []

    def prepare_features(self, returns: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare lagged return features and targets - CORRECTED FOR PROPER ALIGNMENT.

        CRITICAL TIMING STRUCTURE:
        At time t, we observe:
        - Features: [r_{t-n}, r_{t-n+1}, ..., r_{t-1}]  (past returns)
        - Target: r_t (current return to predict)

        This ensures no look-ahead bias.
        """
        n = len(returns)

        if n <= self.n_lags:
            raise ValueError(f"Need at least {self.n_lags + 1} observations, got {n}")

        returns_array = returns.values
        n_samples = n - self.n_lags

        features = np.zeros((n_samples, self.n_lags))
        targets = np.zeros(n_samples)

        for i in range(n_samples):
            # At time t = i + n_lags
            current_time = i + self.n_lags

            # Features: r_{t-n_lags}, ..., r_{t-1}
            # This is returns_array[current_time - n_lags : current_time]
            features[i, :] = returns_array[current_time - self.n_lags:current_time].flatten()

            # Target: r_t = returns_array[current_time]
            targets[i] = returns_array[current_time]

        return features, targets

    def backtest(self, prices: pd.Series, optimize_hyperparams: bool = True) -> Dict:
        """
        Run complete backtesting procedure with proper timing alignment.

        CORRECTED STRATEGY RETURN CALCULATION:
        - At time t, we make prediction using features available at time t
        - We take position based on this prediction
        - At time t+1, we realize the return and calculate P&L
        """
        logger.info("Starting backtest...")

        # Resample data based on frequency
        if self.frequency == 'W':
            prices = prices.resample('W').last().dropna()
        elif self.frequency == 'M':
            prices = prices.resample('M').last().dropna()

        returns = prices.pct_change().dropna()
        features, targets = self.prepare_features(returns)

        # Initialize tracking
        predictions = np.full(len(features), np.nan)
        signals = np.full(len(features), 0)
        positions = np.full(len(features), 0)
        strategy_returns = np.full(len(features), np.nan)

        # Rolling window backtesting
        for i in range(len(features)):
            if i < self.min_periods:
                continue

            # Training window - use only past data
            start_idx = max(0, i - self.lookback_window)
            end_idx = i

            if end_idx <= start_idx:
                continue

            X_train = features[start_idx:end_idx]
            y_train = targets[start_idx:end_idx]

            if len(X_train) < self.min_periods:
                continue

            try:
                # Fit model on historical data
                self.gp.fit(X_train, y_train, optimize_hyperparams=optimize_hyperparams)

                # Make prediction for current period
                X_test = features[i:i+1]
                pred, uncertainty = self.gp.predict(X_test, return_std=True)

                predictions[i] = pred[0]
                signal = self.strategy.generate_signal(pred[0], uncertainty[0] if uncertainty is not None else None)
                signals[i] = signal

                # Position management
                positions[i] = signal

                # CORRECTED: Strategy return calculation
                # The return we realize is from the position we took in the PREVIOUS period
                if i > 0:
                    previous_position = positions[i-1]
                    realized_return = targets[i-1]  # This is the return from period i-1 to i

                    # Transaction cost for position change
                    position_change = abs(positions[i] - previous_position)
                    transaction_cost_impact = position_change * self.transaction_cost

                    # Strategy return = position * realized_return - transaction_costs
                    strategy_return = previous_position * realized_return - transaction_cost_impact
                    strategy_returns[i-1] = strategy_return

            except Exception as e:
                logger.warning(f"Error at step {i}: {e}")
                continue

            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(features)} samples")

        # Store results
        self.predictions = predictions
        self.signals = signals
        self.positions = positions
        self.returns = strategy_returns

        # Calculate equity curve
        valid_returns = pd.Series(strategy_returns).fillna(0)
        self.equity_curve = (1 + valid_returns).cumprod()

        # Calculate performance metrics
        benchmark_returns = targets  # Buy-and-hold benchmark
        strategy_metrics = self._calculate_metrics(strategy_returns, "Strategy")
        benchmark_metrics = self._calculate_metrics(benchmark_returns, "Benchmark")

        # Create results dictionary
        results = {
            'predictions': predictions,
            'signals': signals,
            'positions': positions,
            'strategy_returns': strategy_returns,
            'benchmark_returns': benchmark_returns,
            'equity_curve': self.equity_curve,
            'strategy_metrics': strategy_metrics,
            'benchmark_metrics': benchmark_metrics,
            'returns_series': returns,
            'features': features,
            'targets': targets,
            'prices': prices
        }

        logger.info("Backtest completed!")
        return results

    def _calculate_metrics(self, returns: np.ndarray, name: str) -> TradingMetrics:
        """Calculate comprehensive trading performance metrics."""
        # Remove NaN values
        if isinstance(returns, np.ndarray):
            valid_returns = returns[~np.isnan(returns)]
        else:
            valid_returns = np.array(returns)
            valid_returns = valid_returns[~np.isnan(valid_returns)]

        if len(valid_returns) == 0:
            return TradingMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Calculate period count for annualization
        n_periods = len(valid_returns)
        years = n_periods / self.annualization_factor

        # Basic metrics
        total_return = np.prod(1 + valid_returns) - 1
        annualized_return = (1 + total_return)**(1/years) - 1 if years > 0 else 0
        volatility = np.std(valid_returns) * np.sqrt(self.annualization_factor)

        # Sharpe ratio
        if np.std(valid_returns) > 0:
            sharpe_ratio = (np.mean(valid_returns) * self.annualization_factor) / volatility
        else:
            sharpe_ratio = 0

        # Sortino ratio (using downside deviation)
        downside_returns = valid_returns[valid_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(self.annualization_factor)
            sortino_ratio = (np.mean(valid_returns) * self.annualization_factor) / downside_deviation
        else:
            sortino_ratio = sharpe_ratio  # No downside, same as Sharpe

        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + valid_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)

        # Trade analysis
        trades = valid_returns[valid_returns != 0]
        if len(trades) > 0:
            win_rate = np.sum(trades > 0) / len(trades)
            winning_trades = trades[trades > 0]
            losing_trades = trades[trades < 0]

            if len(losing_trades) > 0 and np.sum(np.abs(losing_trades)) > 0:
                profit_factor = np.sum(winning_trades) / np.sum(np.abs(losing_trades))
            else:
                profit_factor = np.inf if len(winning_trades) > 0 else 0

            avg_trade_return = np.mean(trades)
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_return = 0

        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0

        return TradingMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            avg_trade_return=avg_trade_return,
            volatility=volatility,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )

    def plot_results(self, results: Dict, figsize: Tuple[int, int] = (16, 12)) -> None:
        """Create comprehensive visualization of backtesting results."""
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        # Plot 1: Equity curves comparison
        ax1 = axes[0, 0]

        # Strategy equity curve
        strategy_returns = pd.Series(results['strategy_returns']).fillna(0)
        strategy_equity = (1 + strategy_returns).cumprod()

        # Benchmark equity curve
        benchmark_returns = pd.Series(results['benchmark_returns']).fillna(0)
        benchmark_equity = (1 + benchmark_returns).cumprod()

        ax1.plot(strategy_equity.values, label='Strategy', linewidth=2)
        ax1.plot(benchmark_equity.values, label='Buy & Hold', linewidth=2, alpha=0.7)
        ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Predictions vs Actual
        ax2 = axes[0, 1]
        predictions = results['predictions']
        targets = results['targets']

        valid_mask = ~np.isnan(predictions)
        if np.sum(valid_mask) > 0:
            ax2.scatter(targets[valid_mask], predictions[valid_mask], alpha=0.6, s=20)
            ax2.plot([np.min(targets), np.max(targets)], [np.min(targets), np.max(targets)], 'r--', alpha=0.8)
            ax2.set_xlabel('Actual Returns')
            ax2.set_ylabel('Predicted Returns')
            ax2.set_title('Predictions vs Actual Returns', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

        # Plot 3: Rolling Sharpe Ratio
        ax3 = axes[0, 2]
        window = min(50, len(strategy_returns) // 4)
        if window > 10:
            strategy_rolling_sharpe = strategy_returns.rolling(window).mean() / strategy_returns.rolling(window).std() * np.sqrt(self.annualization_factor)
            benchmark_rolling_sharpe = benchmark_returns.rolling(window).mean() / benchmark_returns.rolling(window).std() * np.sqrt(self.annualization_factor)

            ax3.plot(strategy_rolling_sharpe.values, label='Strategy', linewidth=2)
            ax3.plot(benchmark_rolling_sharpe.values, label='Benchmark', linewidth=2, alpha=0.7)
            ax3.set_title(f'Rolling Sharpe Ratio ({window} periods)', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # Plot 4: Positions and Returns
        ax4 = axes[1, 0]
        positions = results['positions']

        ax4_twin = ax4.twinx()
        ax4.bar(range(len(positions)), positions, alpha=0.6, label='Positions', color='orange')
        ax4_twin.plot(results['targets'], color='blue', alpha=0.7, label='Returns', linewidth=1)

        ax4.set_ylabel('Position', color='orange')
        ax4_twin.set_ylabel('Returns', color='blue')
        ax4.set_title('Trading Positions and Returns', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Time')

        # Plot 5: Performance Metrics Comparison
        ax5 = axes[1, 1]
        strategy_metrics = results['strategy_metrics']
        benchmark_metrics = results['benchmark_metrics']

        metrics_names = ['Sharpe', 'Sortino', 'Calmar']
        strategy_values = [strategy_metrics.sharpe_ratio, strategy_metrics.sortino_ratio, strategy_metrics.calmar_ratio]
        benchmark_values = [benchmark_metrics.sharpe_ratio, benchmark_metrics.sortino_ratio, benchmark_metrics.calmar_ratio]

        x = np.arange(len(metrics_names))
        width = 0.35

        ax5.bar(x - width/2, strategy_values, width, label='Strategy', alpha=0.7)
        ax5.bar(x + width/2, benchmark_values, width, label='Benchmark', alpha=0.7)
        ax5.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels(metrics_names)
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # Plot 6: Drawdown Comparison
        ax6 = axes[1, 2]

        # Calculate drawdowns
        strategy_cumulative = (1 + strategy_returns.fillna(0)).cumprod()
        strategy_peak = strategy_cumulative.expanding().max()
        strategy_drawdown = (strategy_cumulative - strategy_peak) / strategy_peak * 100

        benchmark_cumulative = (1 + benchmark_returns.fillna(0)).cumprod()
        benchmark_peak = benchmark_cumulative.expanding().max()
        benchmark_drawdown = (benchmark_cumulative - benchmark_peak) / benchmark_peak * 100

        ax6.fill_between(range(len(strategy_drawdown)), strategy_drawdown.values, 0,
                        alpha=0.6, color='red', label='Strategy')
        ax6.fill_between(range(len(benchmark_drawdown)), benchmark_drawdown.values, 0,
                        alpha=0.4, color='blue', label='Benchmark')
        ax6.set_title('Drawdown Comparison (%)', fontsize=14, fontweight='bold')
        ax6.set_ylabel('Drawdown (%)')
        ax6.set_xlabel('Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print performance comparison
        self._print_performance_comparison(strategy_metrics, benchmark_metrics)

    def _print_performance_comparison(self, strategy_metrics: TradingMetrics, benchmark_metrics: TradingMetrics) -> None:
        """Print detailed performance comparison."""
        print("\n" + "="*80)
        print("GAUSSIAN PROCESS TRADING SYSTEM - PERFORMANCE COMPARISON")
        print("="*80)
        print(f"{'Metric':<25} {'Strategy':<15} {'Benchmark':<15} {'Difference':<15}")
        print("-"*80)

        metrics = [
            ('Total Return', strategy_metrics.total_return, benchmark_metrics.total_return),
            ('Annualized Return', strategy_metrics.annualized_return, benchmark_metrics.annualized_return),
            ('Sharpe Ratio', strategy_metrics.sharpe_ratio, benchmark_metrics.sharpe_ratio),
            ('Sortino Ratio', strategy_metrics.sortino_ratio, benchmark_metrics.sortino_ratio),
            ('Max Drawdown', strategy_metrics.max_drawdown, benchmark_metrics.max_drawdown),
            ('Volatility', strategy_metrics.volatility, benchmark_metrics.volatility),
            ('Calmar Ratio', strategy_metrics.calmar_ratio, benchmark_metrics.calmar_ratio),
            ('Win Rate', strategy_metrics.win_rate, benchmark_metrics.win_rate),
            ('Profit Factor', strategy_metrics.profit_factor, benchmark_metrics.profit_factor),
        ]

        for name, strat_val, bench_val in metrics:
            if 'Return' in name or 'Drawdown' in name or 'Rate' in name or 'Volatility' in name:
                strat_str = f"{strat_val:.2%}"
                bench_str = f"{bench_val:.2%}"
                diff_str = f"{strat_val - bench_val:.2%}"
            else:
                strat_str = f"{strat_val:.3f}"
                bench_str = f"{bench_val:.3f}"
                diff_str = f"{strat_val - bench_val:.3f}"

            print(f"{name:<25} {strat_str:<15} {bench_str:<15} {diff_str:<15}")

        print("-"*80)
        print(f"{'Num Trades (Strategy)':<25} {strategy_metrics.num_trades:<15}")
        print(f"{'Data Frequency':<25} {self.frequency:<15}")
        print(f"{'Annualization Factor':<25} {self.annualization_factor:<15}")
        print("="*80)

def get_user_inputs():
    """Get user inputs for ticker, dates, and frequency."""
    print("\n" + "="*60)
    print("GAUSSIAN PROCESS TRADING SYSTEM CONFIGURATION")
    print("="*60)

    # Get ticker
    ticker = input("Enter ticker symbol (e.g., SPY, QQQ, AAPL): ").strip().upper()
    if not ticker:
        ticker = "SPY"
        print(f"Using default ticker: {ticker}")

    # Get start date
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for default: ").strip()
    if not start_date:
        start_date = "2020-01-01"
        print(f"Using default start date: {start_date}")

    # Get end date
    end_date = input("Enter end date (YYYY-MM-DD) or press Enter for default: ").strip()
    if not end_date:
        end_date = "2024-12-31"
        print(f"Using default end date: {end_date}")

    # Get frequency
    print("\nAvailable frequencies:")
    print("D - Daily")
    print("W - Weekly")
    print("M - Monthly")
    frequency = input("Enter frequency (D/W/M) or press Enter for daily: ").strip().upper()
    if frequency not in ['D', 'W', 'M']:
        frequency = 'D'
        print(f"Using default frequency: Daily")

    # Get rolling window
    rolling_window = input("Enter length of rolling window or press Enter for default: ").strip()
    if not rolling_window:
        # Initialize trading system with frequency-aware parameters
        if frequency == 'D':
            default_lookback = 63    # ~3 months daily
            default_min_periods = 21  # ~1 month daily
        elif frequency == 'W':
            default_lookback = 26    # ~6 months weekly
            default_min_periods = 8  # ~2 months weekly
        else:  # Monthly
            default_lookback = 12    # ~1 year monthly
            default_min_periods = 6  # ~6 months monthly
    else:
        default_lookback = int(rolling_window)
        default_min_periods = 10

    # Get the number of lags and max degree of basis functions
    number_of_lags = input("Enter number of lags (required) >= 1: ").strip()
    number_of_lags = int(number_of_lags)
    max_degree = input("Enter maximum degree of polynamial basis (required) >= 1: ").strip()
    max_degree = int(max_degree)

    return ticker, start_date, end_date, frequency, default_lookback, default_min_periods, number_of_lags, max_degree

def main():
    """
    Main function to demonstrate the Enhanced GP Trading System.

    KEY IMPROVEMENTS:
    1. User-configurable ticker, dates, and frequency
    2. Proper multi-frequency data handling
    3. Correct annualization factors
    4. Benchmark performance metrics
    5. Enhanced look-ahead bias protection
    6. Corrected feature/target alignment
    """
    np.random.seed(42)

    print("Enhanced Gaussian Process Trading System")
    print("LOOK-AHEAD BIAS PROTECTION: ENABLED")
    print("MULTI-FREQUENCY SUPPORT: ENABLED")

    # Get user configuration
    ticker, start_date, end_date, frequency, default_lookback, default_min_periods, number_of_lags, max_degree = get_user_inputs()

    print(f"\nConfiguration Summary:")
    print(f"  Ticker: {ticker}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Frequency: {frequency}")

    # Download data
    print(f"\nDownloading {ticker} data from Yahoo Finance...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"No data found for {ticker}")
            return

        prices = data['Close'].dropna()
        print(f"Downloaded {len(prices)} price points from {prices.index[0].date()} to {prices.index[-1].date()}")

    except Exception as e:
        print(f"Error downloading data: {e}")
        return

    # Initialize the trading system
    trading_system = GPTradingSystem(
        n_lags=number_of_lags,
        max_degree=max_degree,
        lookback_window=default_lookback,
        min_periods=default_min_periods,
        transaction_cost=0.0005,  # 0.000x bps
        frequency=frequency
    )

    print(f"\nTrading System Configuration:")
    print(f"  - Lagged Returns: {trading_system.n_lags}")
    print(f"  - Max Polynomial Degree: {trading_system.max_degree}")
    print(f"  - Lookback Window: {trading_system.lookback_window} periods")
    print(f"  - Minimum Periods: {trading_system.min_periods} periods")
    print(f"  - Transaction Cost: {trading_system.transaction_cost*10000:.1f} bps")
    print(f"  - Frequency: {frequency}")
    print(f"  - Annualization Factor: {trading_system.annualization_factor}")

    # Data integrity check
    if frequency == 'W':
        resampled_prices = prices.resample('W').last().dropna()
    elif frequency == 'M':
        resampled_prices = prices.resample('M').last().dropna()
    else:
        resampled_prices = prices

    returns = resampled_prices.pct_change().dropna()

    print(f"\nData Integrity Check:")
    print(f"  - Original price observations: {len(prices)}")
    print(f"  - Resampled price observations: {len(resampled_prices)}")
    print(f"  - Return observations: {len(returns)}")
    print(f"  - First return date: {returns.index[0].date()}")
    print(f"  - Last return date: {returns.index[-1].date()}")

    # Check if we have enough data
    min_required = trading_system.n_lags + trading_system.min_periods + 10
    if len(returns) < min_required:
        print(f"\nInsufficient data: need at least {min_required} observations, got {len(returns)}")
        print("Try using a longer date range or lower frequency.")
        return

    # Run backtest
    print(f"\nRunning backtest with enhanced look-ahead bias protection...")
    start_time = pd.Timestamp.now()

    try:
        results = trading_system.backtest(prices, optimize_hyperparams=True)

        end_time = pd.Timestamp.now()
        print(f"Backtest completed in {(end_time - start_time).total_seconds():.1f} seconds")

        # Verify results
        strategy_returns = results['strategy_returns']
        valid_strategy_returns = strategy_returns[~np.isnan(strategy_returns)]

        print(f"\nBacktest Results Summary:")
        print(f"  - Strategy return observations: {len(valid_strategy_returns)}")
        print(f"  - Prediction observations: {np.sum(~np.isnan(results['predictions']))}")
        print(f"  - Trading periods: {np.sum(results['positions'] != 0)}")

        # Display results
        trading_system.plot_results(results)

        # Additional analysis
        print("\n" + "="*60)
        print("ADDITIONAL ANALYSIS")
        print("="*60)

        # Prediction accuracy
        predictions = results['predictions']
        targets = results['targets']
        valid_mask = ~np.isnan(predictions)

        if np.sum(valid_mask) > 0:
            pred_valid = predictions[valid_mask]
            targets_valid = targets[valid_mask]

            direction_accuracy = np.mean(np.sign(pred_valid) == np.sign(targets_valid))
            correlation = np.corrcoef(pred_valid, targets_valid)[0, 1]
            mse = np.mean((pred_valid - targets_valid)**2)
            mae = np.mean(np.abs(pred_valid - targets_valid))

            print(f"Direction Accuracy:    {direction_accuracy:>10.2%}")
            print(f"Prediction Correlation:{correlation:>10.3f}")
            print(f"Mean Squared Error:    {mse:>10.6f}")
            print(f"Mean Absolute Error:   {mae:>10.6f}")

        # Risk analysis
        if len(valid_strategy_returns) > 0:
            var_95 = np.percentile(valid_strategy_returns, 5)
            var_99 = np.percentile(valid_strategy_returns, 1)

            print(f"\nRisk Metrics:")
            print(f"Value at Risk (95%):   {var_95:>10.4f}")
            print(f"Value at Risk (99%):   {var_99:>10.4f}")

        # Trading frequency analysis
        positions = results['positions']
        position_changes = np.sum(np.abs(np.diff(positions)) > 0)

        if frequency == 'D':
            freq_label = "trades/year"
            trading_frequency = position_changes / len(positions) * 252
        elif frequency == 'W':
            freq_label = "trades/year"
            trading_frequency = position_changes / len(positions) * 52
        else:
            freq_label = "trades/year"
            trading_frequency = position_changes / len(positions) * 12

        print(f"Trading Frequency:     {trading_frequency:>10.1f} {freq_label}")

        return results

    except Exception as e:
        print(f"Error during backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

# Advanced features
class AdvancedGPTradingSystem(GPTradingSystem):
    """Advanced version with regime detection and dynamic position sizing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.confidence_threshold = 0.4
        self.max_position_size = 1.0

    def dynamic_position_sizing(self, prediction: float, uncertainty: float) -> float:
        """Calculate position size based on prediction confidence."""
        if uncertainty == 0:
            confidence = 1.0
        else:
            confidence = 1.0 / (1.0 + uncertainty)

        if confidence < self.confidence_threshold:
            return 0.0

        base_position = np.sign(prediction)
        position_magnitude = min(confidence * abs(prediction) * 10, self.max_position_size)

        return base_position * position_magnitude

# Example usage and testing
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    try:
        results = main()

        if results is not None:
            print("\n" + "="*60)
            print("Enhanced GP Trading System completed successfully!")
            print("Results available in 'results' variable for further analysis.")
            print("="*60)

    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()