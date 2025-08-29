#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-gpt/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
# -------------------------------------------------------------------------------------

"""
Author: Dimitrios Thomakos with prompts to Claude (Anthropic)
Date: August 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky, solve_triangular, LinAlgError
from scipy.special import eval_chebyt
from scipy.optimize import minimize
import yfinance as yf
from typing import Tuple, Optional, Dict, List, Union
import warnings
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from itertools import combinations
import cvxpy as cp
from sklearn.covariance import LedoitWolf

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _estimate_correlations(self):
        """Estimate cross-asset correlation structure from GP residuals."""
        if self.X_train is None or self.y_train is None:
            return

        # Get residuals from each asset GP
        residuals = np.zeros_like(self.y_train)
        for i in range(self.n_assets):
            predictions, _ = self.asset_gps[i].predict(self.X_train)
            predictions = np.asarray(predictions).flatten()
            residuals[:, i] = self.y_train[:, i] - predictions

        # Handle case where residuals have no variance
        if residuals.shape[0] < 2:
            self.correlation_matrix = np.eye(self.n_assets)
            return

        # Check for zero variance columns
        residual_stds = np.std(residuals, axis=0)
        if np.any(residual_stds < 1e-10):
            self.correlation_matrix = np.eye(self.n_assets)
            return

        try:
            # Estimate correlation matrix with shrinkage
            lw = LedoitWolf()
            cov_matrix = lw.fit(residuals).covariance_

            # Convert to correlation matrix
            std_devs = np.sqrt(np.diag(cov_matrix))
            std_devs = np.maximum(std_devs, 1e-10)  # Prevent division by zero
            self.correlation_matrix = cov_matrix / np.outer(std_devs, std_devs)

        except Exception as e:
            logger.warning(f"Error estimating correlations: {e}")
            self.correlation_matrix = np.eye(self.n_assets)

        # Ensure positive definite
        try:
            eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-6)
            self.correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
        except Exception:
            self.correlation_matrix = np.eye(self.n_assets)

def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make multi-asset predictions.

        Returns:
        --------
        predictions : np.ndarray
            Shape (n_samples, n_assets)
        uncertainties : np.ndarray, optional
            Shape (n_samples, n_assets, n_assets) - full covariance
        """
        if any(gp.L is None for gp in self.asset_gps.values()):
            raise ValueError("Models must be fitted before making predictions")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_test_samples = X.shape[0]

        # Get individual predictions and uncertainties
        predictions = np.zeros((n_test_samples, self.n_assets))
        individual_vars = np.zeros((n_test_samples, self.n_assets))

        for i in range(self.n_assets):
            pred, std = self.asset_gps[i].predict(X, return_std=True)
            predictions[:, i] = pred
            if std is not None:
                individual_vars[:, i] = std**2

        if not return_std:
            return predictions, None

        # Construct full covariance matrix incorporating correlations
        covariance_matrices = np.zeros((n_test_samples, self.n_assets, self.n_assets))

        for t in range(n_test_samples):
            # Diagonal: individual variances
            std_t = np.sqrt(individual_vars[t, :])
            cov_t = np.outer(std_t, std_t) * self.correlation_matrix
            covariance_matrices[t] = cov_t

        return predictions, covariance_matrices#!/usr/bin/env python3
"""
Multi-Asset Gaussian Process Trading System

This module extends the single-asset GP trading system to handle portfolios of multiple assets.
It implements cross-asset correlation modeling, portfolio optimization, and multi-dimensional
Gaussian Process regression for simultaneous prediction across assets.

Key Features:
- Multi-asset GP modeling with cross-correlations
- Portfolio-level optimization and risk management
- Dynamic asset allocation based on GP predictions
- Cross-asset feature engineering
- Comprehensive multi-asset backtesting framework

Author: Dimitrios Thomakos with prompts to Claude (Anthropic)
Date: August 2025
"""


# Import base classes - these need to be defined locally since we can't import from GPT_v3
# We'll redefine the essential base classes here

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
            K_jittered = K + 1e-3 * np.eye(K.shape[0])
            try:
                return cholesky(K_jittered, lower=True)
            except LinAlgError:
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

        K = self._compute_kernel_matrix(self.Phi)
        self.L = self._safe_cholesky(K)
        self.alpha = solve_triangular(self.L, y, lower=True)

        return self

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions on new data."""
        if self.L is None:
            raise ValueError("Model must be fitted before making predictions")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiAssetMetrics:
    """Container for multi-asset portfolio performance metrics."""
    # Individual asset metrics
    individual_metrics: Dict[str, TradingMetrics]

    # Portfolio-level metrics
    portfolio_metrics: TradingMetrics

    # Cross-asset statistics
    correlation_matrix: np.ndarray
    asset_weights: np.ndarray
    diversification_ratio: float
    portfolio_turnover: float

    # Risk decomposition
    marginal_contributions: Dict[str, float]
    component_contributions: Dict[str, float]

class CrossAssetBasisFunction(BasisFunction):
    """Extended basis functions incorporating cross-asset interactions."""

    def __init__(self,
                 n_assets: int,
                 max_degree: int = 3,
                 include_cross_terms: bool = True,
                 normalize: bool = True):
        self.n_assets = n_assets
        self.max_degree = max_degree
        self.include_cross_terms = include_cross_terms
        self.normalize = normalize

        # Fitted normalization parameters for each asset
        self.fitted_min = {}
        self.fitted_max = {}

        # Initialize with default values
        self.n_lags = 5  # Default value, will be set properly later
        self._num_features = 100  # Default value

        # Cache for feature names
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build descriptive names for all features."""
        self.feature_names = ['const']

        # Linear terms for each asset and lag
        for asset in range(self.n_assets):
            for lag in range(self.n_lags):
                self.feature_names.append(f'asset_{asset}_lag_{lag+1}')

        # Higher-order terms
        for degree in range(2, self.max_degree + 1):
            for asset in range(self.n_assets):
                for lag in range(self.n_lags):
                    self.feature_names.append(f'asset_{asset}_lag_{lag+1}_deg_{degree}')

        # Cross-asset interaction terms
        if self.include_cross_terms:
            for asset1 in range(self.n_assets):
                for asset2 in range(asset1 + 1, self.n_assets):
                    for lag1 in range(self.n_lags):
                        for lag2 in range(self.n_lags):
                            self.feature_names.append(f'cross_asset_{asset1}_{asset2}_lag_{lag1+1}_{lag2+1}')

    def _normalize_input(self, x: np.ndarray, asset_idx: int, fit: bool = False) -> np.ndarray:
        """Normalize input for specific asset to [-1, 1] range."""
        if not self.normalize:
            return x

        key = f'asset_{asset_idx}'

        if fit or key not in self.fitted_min:
            self.fitted_min[key] = np.min(x)
            self.fitted_max[key] = np.max(x)

        min_val = self.fitted_min[key]
        max_val = self.fitted_max[key]

        if max_val == min_val:
            return np.zeros_like(x)

        return 2 * (x - min_val) / (max_val - min_val) - 1

    def compute(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Compute cross-asset basis functions.

        Parameters:
        -----------
        x : np.ndarray
            Input array of shape (n_samples, n_assets * n_lags)
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        n_samples, total_features = x.shape

        # Verify input shape
        if total_features % self.n_assets != 0:
            raise ValueError(f"Input features {total_features} not divisible by n_assets {self.n_assets}")

        n_lags_per_asset = total_features // self.n_assets
        self.n_lags = n_lags_per_asset

        # Rebuild feature names if needed
        if len(self.feature_names) <= 1:
            self._build_feature_names()

        # Reshape input: (n_samples, n_assets, n_lags)
        x_reshaped = x.reshape(n_samples, self.n_assets, n_lags_per_asset)

        n_basis_features = self.get_num_features()
        basis_matrix = np.ones((n_samples, n_basis_features))

        # Normalize each asset's features
        x_normalized = np.zeros_like(x_reshaped)
        for asset in range(self.n_assets):
            for lag in range(n_lags_per_asset):
                x_asset_lag = x_reshaped[:, asset, lag]
                x_normalized[:, asset, lag] = self._normalize_input(x_asset_lag, asset, fit=fit)

        feature_idx = 1  # Skip constant term

        # Linear terms for each asset
        for asset in range(self.n_assets):
            for lag in range(n_lags_per_asset):
                if feature_idx < n_basis_features:
                    basis_matrix[:, feature_idx] = x_normalized[:, asset, lag]
                    feature_idx += 1

        # Higher-order terms
        for degree in range(2, self.max_degree + 1):
            for asset in range(self.n_assets):
                for lag in range(n_lags_per_asset):
                    if feature_idx < n_basis_features:
                        x_asset_lag = x_normalized[:, asset, lag]
                        basis_matrix[:, feature_idx] = eval_chebyt(degree, x_asset_lag)
                        feature_idx += 1

        # Cross-asset interaction terms
        if self.include_cross_terms:
            for asset1 in range(self.n_assets):
                for asset2 in range(asset1 + 1, self.n_assets):
                    for lag1 in range(n_lags_per_asset):
                        for lag2 in range(n_lags_per_asset):
                            if feature_idx < n_basis_features:
                                interaction = (x_normalized[:, asset1, lag1] *
                                             x_normalized[:, asset2, lag2])
                                basis_matrix[:, feature_idx] = interaction
                                feature_idx += 1

        return basis_matrix

    def get_num_features(self) -> int:
        """Calculate total number of basis features."""
        return self._num_features

    def set_num_features(self, n_lags_per_asset: int):
        """Set number of features based on lags per asset."""
        self.n_lags = n_lags_per_asset

        features = 1  # constant term

        # Linear terms: n_assets * n_lags_per_asset
        features += self.n_assets * n_lags_per_asset

        # Higher-order terms: (max_degree - 1) * n_assets * n_lags_per_asset
        features += (self.max_degree - 1) * self.n_assets * n_lags_per_asset

        # Cross-asset interaction terms
        if self.include_cross_terms:
            n_cross_pairs = self.n_assets * (self.n_assets - 1) // 2
            features += n_cross_pairs * n_lags_per_asset * n_lags_per_asset

        self._num_features = features

        # Rebuild feature names with correct dimensions
        self._build_feature_names()

class MultiAssetGaussianProcess:
    """Multi-output Gaussian Process for simultaneous asset prediction."""

    def __init__(self,
                 n_assets: int,
                 basis_function: BasisFunction,
                 noise_variance: float = 1e-4,
                 prior_variance: float = 1.0,
                 correlation_prior: float = 0.1,
                 jitter: float = 1e-6):
        self.n_assets = n_assets
        self.basis_function = basis_function
        self.noise_variance = noise_variance
        self.prior_variance = prior_variance
        self.correlation_prior = correlation_prior
        self.jitter = jitter

        # Individual GPs for each asset
        self.asset_gps = {}
        for i in range(n_assets):
            self.asset_gps[i] = GaussianProcessRegressor(
                basis_function=basis_function,
                noise_variance=noise_variance,
                prior_variance=prior_variance,
                jitter=jitter
            )

        # Cross-asset correlation model
        self.correlation_matrix = np.eye(n_assets)

        # Fitted parameters
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray, optimize_hyperparams: bool = True) -> 'MultiAssetGaussianProcess':
        """
        Fit multi-asset GP model.

        Parameters:
        -----------
        X : np.ndarray
            Features of shape (n_samples, n_features)
        y : np.ndarray
            Targets of shape (n_samples, n_assets)
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples_X, _ = X.shape
        n_samples_y, n_assets_y = y.shape

        if n_samples_X != n_samples_y:
            raise ValueError(f"X and y have incompatible shapes: X={X.shape}, y={y.shape}")

        if n_assets_y != self.n_assets:
            raise ValueError(f"Expected {self.n_assets} assets, got {n_assets_y}")

        self.X_train = X.copy()
        self.y_train = y.copy()

        # Fit individual asset GPs
        for i in range(self.n_assets):
            y_asset = y[:, i]
            self.asset_gps[i].fit(X, y_asset, optimize_hyperparams=optimize_hyperparams)

        # Estimate cross-asset correlations from residuals
        self._estimate_correlations()

        return self

    def _estimate_correlations(self):
        """Estimate cross-asset correlation structure from GP residuals."""
        if self.X_train is None or self.y_train is None:
            return

        # Get residuals from each asset GP
        residuals = np.zeros_like(self.y_train)
        for i in range(self.n_assets):
            predictions, _ = self.asset_gps[i].predict(self.X_train)
            residuals[:, i] = self.y_train[:, i] - predictions

        # Estimate correlation matrix with shrinkage
        lw = LedoitWolf()
        self.correlation_matrix = lw.fit(residuals).covariance_

        # Convert to correlation matrix
        std_devs = np.sqrt(np.diag(self.correlation_matrix))
        self.correlation_matrix = (self.correlation_matrix /
                                 np.outer(std_devs, std_devs))

        # Ensure positive definite
        eigenvals, eigenvecs = np.linalg.eigh(self.correlation_matrix)
        eigenvals = np.maximum(eigenvals, 1e-6)
        self.correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

    def predict(self, X: np.ndarray, return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make multi-asset predictions.

        Returns:
        --------
        predictions : np.ndarray
            Shape (n_samples, n_assets)
        uncertainties : np.ndarray, optional
            Shape (n_samples, n_assets, n_assets) - full covariance
        """
        if any(gp.L is None for gp in self.asset_gps.values()):
            raise ValueError("Models must be fitted before making predictions")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        n_test_samples = X.shape[0]

        # Get individual predictions and uncertainties
        predictions = np.zeros((n_test_samples, self.n_assets))
        individual_vars = np.zeros((n_test_samples, self.n_assets))

        for i in range(self.n_assets):
            pred, std = self.asset_gps[i].predict(X, return_std=True)
            predictions[:, i] = pred
            if std is not None:
                individual_vars[:, i] = std**2

        if not return_std:
            return predictions, None

        # Construct full covariance matrix incorporating correlations
        covariance_matrices = np.zeros((n_test_samples, self.n_assets, self.n_assets))

        for t in range(n_test_samples):
            # Diagonal: individual variances
            std_t = np.sqrt(individual_vars[t, :])
            cov_t = np.outer(std_t, std_t) * self.correlation_matrix
            covariance_matrices[t] = cov_t

        return predictions, covariance_matrices

class PortfolioOptimizer:
    """Portfolio optimization using GP predictions and uncertainties."""

    def __init__(self,
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.4,
                 min_weight: float = -0.4,
                 transaction_cost: float = 0.001,
                 target_volatility: Optional[float] = None):
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.transaction_cost = transaction_cost
        self.target_volatility = target_volatility

    def optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize portfolio weights using mean-variance framework.

        Parameters:
        -----------
        expected_returns : np.ndarray
            Expected returns for each asset
        covariance_matrix : np.ndarray
            Asset return covariance matrix
        current_weights : np.ndarray, optional
            Current portfolio weights for transaction cost calculation

        Returns:
        --------
        optimal_weights : np.ndarray
            Optimized portfolio weights
        """
        expected_returns = np.asarray(expected_returns).flatten()
        covariance_matrix = np.asarray(covariance_matrix)

        n_assets = len(expected_returns)

        # Ensure covariance matrix is positive definite
        eigenvals, eigenvecs = np.linalg.eigh(covariance_matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        covariance_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

        try:
            # Create optimization variables
            w = cp.Variable(n_assets)

            # Portfolio return and risk
            portfolio_return = w.T @ expected_returns
            portfolio_risk = cp.quad_form(w, covariance_matrix)

            # Objective: maximize utility (return - risk_aversion * risk)
            objective = portfolio_return - 0.5 * self.risk_aversion * portfolio_risk

            # Add transaction costs if current weights provided
            if current_weights is not None:
                current_weights = np.asarray(current_weights).flatten()
                transaction_costs = cp.sum(cp.abs(w - current_weights)) * self.transaction_cost
                objective -= transaction_costs

            # Constraints
            constraints = [
                cp.sum(w) == 1,  # Fully invested
                w >= self.min_weight,  # Lower bounds
                w <= self.max_weight   # Upper bounds
            ]

            # Volatility targeting constraint
            if self.target_volatility is not None:
                portfolio_vol = cp.sqrt(cp.quad_form(w, covariance_matrix))
                constraints.append(portfolio_vol <= self.target_volatility)

            # Solve optimization problem
            problem = cp.Problem(cp.Maximize(objective), constraints)

            problem.solve()

            if problem.status == cp.OPTIMAL:
                optimal_weights = w.value
                if optimal_weights is not None:
                    # Ensure weights sum to 1 and handle numerical errors
                    optimal_weights = np.asarray(optimal_weights).flatten()
                    optimal_weights = optimal_weights / np.sum(optimal_weights)
                    return optimal_weights
                else:
                    logger.warning("Optimization returned None weights")
                    return np.ones(n_assets) / n_assets
            else:
                logger.warning(f"Optimization failed with status: {problem.status}")
                # Fallback to equal weights
                return np.ones(n_assets) / n_assets

        except Exception as e:
            logger.warning(f"Portfolio optimization error: {e}")
            return np.ones(n_assets) / n_assets

class MultiAssetTradingSystem:
    """Complete multi-asset GP trading system."""

    def __init__(self,
                 assets: List[str],
                 n_lags: int = 5,
                 max_degree: int = 3,
                 lookback_window: int = 252,
                 min_periods: int = 60,
                 include_cross_terms: bool = True,
                 risk_aversion: float = 1.0,
                 max_weight: float = 0.4,
                 transaction_cost: float = 0.001,
                 frequency: str = 'D',
                 rebalance_frequency: int = 5):

        self.assets = assets
        self.n_assets = len(assets)
        self.n_lags = n_lags
        self.max_degree = max_degree
        self.lookback_window = lookback_window
        self.min_periods = min_periods
        self.include_cross_terms = include_cross_terms
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.transaction_cost = transaction_cost
        self.frequency = frequency.upper()
        self.rebalance_frequency = rebalance_frequency

        # Set annualization factor based on frequency
        if self.frequency == 'D':
            self.annualization_factor = 252
        elif self.frequency == 'W':
            self.annualization_factor = 52
        elif self.frequency == 'M':
            self.annualization_factor = 12
        else:
            raise ValueError(f"Unsupported frequency: {frequency}")

        # Initialize components
        self.basis_function = CrossAssetBasisFunction(
            n_assets=self.n_assets,
            max_degree=max_degree,
            include_cross_terms=include_cross_terms,
            normalize=True
        )

        self.multi_gp = MultiAssetGaussianProcess(
            n_assets=self.n_assets,
            basis_function=self.basis_function,
            noise_variance=1e-4,
            prior_variance=1.0
        )

        self.portfolio_optimizer = PortfolioOptimizer(
            risk_aversion=risk_aversion,
            max_weight=max_weight,
            min_weight=-max_weight,
            transaction_cost=transaction_cost
        )

        # Results storage
        self.predictions = []
        self.portfolio_weights = []
        self.individual_returns = []
        self.portfolio_returns = []

    def prepare_multi_asset_features(self, returns_dict: Dict[str, pd.Series]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare multi-asset features and targets with proper temporal alignment.

        Parameters:
        -----------
        returns_dict : Dict[str, pd.Series]
            Dictionary mapping asset names to return series

        Returns:
        --------
        features : np.ndarrayspy,
            Shape (n_samples, n_assets * n_lags)
        targets : np.ndarray
            Shape (n_samples, n_assets)
        """
        # Clean returns_dict:
        assets = list(returns_dict.keys())
        returns_df = pd.DataFrame(data=None, columns=assets)
        for i in assets:
            returns_df[i] = returns_dict[i]

        # Align all return series to common dates
        returns_df = returns_df.dropna()
        n_periods = len(returns_df)

        if n_periods <= self.n_lags:
            raise ValueError(f"Need at least {self.n_lags + 1} observations, got {n_periods}")

        n_samples = n_periods - self.n_lags
        n_features_total = self.n_assets * self.n_lags

        features = np.zeros((n_samples, n_features_total))
        targets = np.zeros((n_samples, self.n_assets))

        for i in range(n_samples):
            current_time = i + self.n_lags

            # Features: lagged returns for all assets
            for asset_idx, asset in enumerate(self.assets):
                start_lag_idx = current_time - self.n_lags
                end_lag_idx = current_time

                asset_features = returns_df[asset].iloc[start_lag_idx:end_lag_idx].values

                # Place in feature matrix
                start_feature_idx = asset_idx * self.n_lags
                end_feature_idx = (asset_idx + 1) * self.n_lags
                features[i, start_feature_idx:end_feature_idx] = asset_features

            # Targets: current period returns
            targets[i, :] = returns_df.iloc[current_time].values

        return features, targets

    def backtest(self, price_data: Dict[str, pd.Series], optimize_hyperparams: bool = True) -> Dict:
        """
        Run multi-asset backtesting procedure.

        Parameters:
        -----------
        price_data : Dict[str, pd.Series]
            Dictionary mapping asset names to price series

        Returns:
        --------
        results : Dict
            Comprehensive backtesting results
        """
        logger.info("Starting multi-asset backtest...")

        # Resample data based on frequency
        resampled_prices = {}
        returns_dict = {}

        for asset, prices in price_data.items():
            if self.frequency == 'W':
                prices_resampled = prices.resample('W').last().dropna()
            elif self.frequency == 'M':
                prices_resampled = prices.resample('M').last().dropna()
            else:
                prices_resampled = prices

            resampled_prices[asset] = prices_resampled
            returns_dict[asset] = prices_resampled.pct_change().dropna()

        # Prepare features and targets
        features, targets = self.prepare_multi_asset_features(returns_dict)

        # Initialize tracking arrays
        n_samples = len(features)
        predictions = np.full((n_samples, self.n_assets), np.nan)
        covariances = np.full((n_samples, self.n_assets, self.n_assets), np.nan)
        portfolio_weights = np.full((n_samples, self.n_assets), np.nan)
        portfolio_returns = np.full(n_samples, np.nan)

        # Initialize equal weights
        current_weights = np.ones(self.n_assets) / self.n_assets

        # Rolling window backtesting
        for i in range(len(features)):
            if i < self.min_periods:
                portfolio_weights[i] = current_weights
                continue

            # Training window
            start_idx = max(0, i - self.lookback_window)
            end_idx = i

            if end_idx <= start_idx:
                continue

            X_train = features[start_idx:end_idx]
            y_train = targets[start_idx:end_idx]

            if len(X_train) < self.min_periods:
                continue

            try:
                # Set basis function parameters
                n_lags_per_asset = self.n_lags
                self.basis_function.set_num_features(n_lags_per_asset)

                # Fit multi-asset GP model
                self.multi_gp.fit(X_train, y_train, optimize_hyperparams=optimize_hyperparams)

                # Make prediction for current period
                X_test = features[i:i+1]
                pred, cov = self.multi_gp.predict(X_test, return_std=True)

                predictions[i] = pred[0]
                if cov is not None:
                    covariances[i] = cov[0]

                # Portfolio optimization (rebalance only at specified frequency)
                if i % self.rebalance_frequency == 0 or i == self.min_periods:
                    expected_returns = pred[0]
                    covariance_matrix = cov[0] if cov is not None else np.eye(self.n_assets) * 1e-4

                    optimal_weights = self.portfolio_optimizer.optimize_portfolio(
                        expected_returns=expected_returns,
                        covariance_matrix=covariance_matrix,
                        current_weights=current_weights
                    )

                    current_weights = optimal_weights

                portfolio_weights[i] = current_weights

                # Calculate portfolio return
                if i > 0:
                    realized_returns = targets[i-1]
                    previous_weights = portfolio_weights[i-1]

                    if not np.any(np.isnan(previous_weights)):
                        # Portfolio return
                        portfolio_return = np.sum(previous_weights * realized_returns)

                        # Transaction costs (weight changes)
                        if i > 1:
                            weight_changes = np.sum(np.abs(portfolio_weights[i] - portfolio_weights[i-1]))
                            transaction_cost_impact = weight_changes * self.transaction_cost
                            portfolio_return -= transaction_cost_impact

                        portfolio_returns[i-1] = portfolio_return

            except Exception as e:
                logger.warning(f"Error at step {i}: {e}")
                portfolio_weights[i] = current_weights
                continue

            if i % 50 == 0:
                logger.info(f"Processed {i}/{len(features)} samples")

        # Store results
        self.predictions = predictions
        self.portfolio_weights = portfolio_weights
        self.portfolio_returns = portfolio_returns

        # Calculate individual asset benchmark returns
        individual_benchmark_returns = {}
        for asset_idx, asset in enumerate(self.assets):
            individual_benchmark_returns[asset] = targets[:, asset_idx]

        # Calculate portfolio benchmark (equal weight)
        equal_weight_returns = np.mean(targets, axis=1)

        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_results(
            predictions, portfolio_weights, portfolio_returns,
            targets, individual_benchmark_returns, equal_weight_returns,
            covariances
        )

        logger.info("Multi-asset backtest completed!")
        return results

    def _calculate_comprehensive_results(self,
                                       predictions: np.ndarray,
                                       portfolio_weights: np.ndarray,
                                       portfolio_returns: np.ndarray,
                                       targets: np.ndarray,
                                       individual_benchmark_returns: Dict[str, np.ndarray],
                                       equal_weight_returns: np.ndarray,
                                       covariances: np.ndarray) -> Dict:
        """Calculate comprehensive multi-asset results."""

        # Individual asset metrics
        individual_metrics = {}
        for asset_idx, asset in enumerate(self.assets):
            asset_returns = targets[:, asset_idx]
            individual_metrics[asset] = self._calculate_single_asset_metrics(asset_returns, f"Benchmark_{asset}")

        # Portfolio metrics
        portfolio_metrics = self._calculate_single_asset_metrics(portfolio_returns, "Portfolio")
        equal_weight_metrics = self._calculate_single_asset_metrics(equal_weight_returns, "Equal_Weight_Benchmark")

        # Cross-asset statistics
        valid_returns_mask = ~np.any(np.isnan(targets), axis=1)
        valid_targets = targets[valid_returns_mask]

        if len(valid_targets) > 1:
            correlation_matrix = np.corrcoef(valid_targets.T)
        else:
            correlation_matrix = np.eye(self.n_assets)

        # Average portfolio weights
        valid_weights_mask = ~np.any(np.isnan(portfolio_weights), axis=1)
        if np.any(valid_weights_mask):
            avg_weights = np.mean(portfolio_weights[valid_weights_mask], axis=0)
        else:
            avg_weights = np.ones(self.n_assets) / self.n_assets

        # Diversification ratio
        if len(valid_targets) > 1:
            individual_vols = np.std(valid_targets, axis=0)
            weighted_vol = np.sum(avg_weights * individual_vols)
            portfolio_vol = np.std(portfolio_returns[~np.isnan(portfolio_returns)])
            diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        else:
            diversification_ratio = 1.0

        # Portfolio turnover
        if np.any(valid_weights_mask):
            weight_changes = np.diff(portfolio_weights[valid_weights_mask], axis=0)
            portfolio_turnover = np.mean(np.sum(np.abs(weight_changes), axis=1))
        else:
            portfolio_turnover = 0.0

        # Risk decomposition (marginal and component contributions)
        marginal_contributions = {}
        component_contributions = {}

        if len(valid_targets) > 1 and np.any(valid_weights_mask):
            cov_matrix = np.cov(valid_targets.T)
            portfolio_variance = np.dot(avg_weights, np.dot(cov_matrix, avg_weights))

            for i, asset in enumerate(self.assets):
                marginal_contrib = np.dot(cov_matrix[i], avg_weights) / np.sqrt(portfolio_variance)
                component_contrib = avg_weights[i] * marginal_contrib

                marginal_contributions[asset] = marginal_contrib
                component_contributions[asset] = component_contrib

        # Create comprehensive metrics object
        multi_asset_metrics = MultiAssetMetrics(
            individual_metrics=individual_metrics,
            portfolio_metrics=portfolio_metrics,
            correlation_matrix=correlation_matrix,
            asset_weights=avg_weights,
            diversification_ratio=diversification_ratio,
            portfolio_turnover=portfolio_turnover,
            marginal_contributions=marginal_contributions,
            component_contributions=component_contributions
        )

        return {
            'predictions': predictions,
            'portfolio_weights': portfolio_weights,
            'portfolio_returns': portfolio_returns,
            'individual_returns': targets,
            'individual_benchmark_returns': individual_benchmark_returns,
            'equal_weight_returns': equal_weight_returns,
            'covariances': covariances,
            'multi_asset_metrics': multi_asset_metrics,
            'equal_weight_metrics': equal_weight_metrics,
            'assets': self.assets,
            'correlation_matrix': correlation_matrix
        }

    def _calculate_single_asset_metrics(self, returns: np.ndarray, name: str) -> TradingMetrics:
        """Calculate metrics for a single return series."""
        # Remove NaN values
        if isinstance(returns, np.ndarray):
            valid_returns = returns[~np.isnan(returns)]
        else:
            valid_returns = np.array(returns)
            valid_returns = valid_returns[~np.isnan(valid_returns)]

        if len(valid_returns) == 0:
            return TradingMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

        # Calculate metrics using same logic as original system
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

        # Sortino ratio
        downside_returns = valid_returns[valid_returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns) * np.sqrt(self.annualization_factor)
            sortino_ratio = (np.mean(valid_returns) * self.annualization_factor) / downside_deviation
        else:
            sortino_ratio = sharpe_ratio

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

    def plot_multi_asset_results(self, results: Dict, figsize: Tuple[int, int] = (20, 16)) -> None:
        """Create comprehensive multi-asset visualization."""
        fig, axes = plt.subplots(3, 4, figsize=figsize)

        # Plot 1: Portfolio vs Equal Weight Equity Curves
        ax1 = axes[0, 0]
        portfolio_returns = pd.Series(results['portfolio_returns']).fillna(0)
        equal_weight_returns = pd.Series(results['equal_weight_returns']).fillna(0)

        portfolio_equity = (1 + portfolio_returns).cumprod()
        equal_weight_equity = (1 + equal_weight_returns).cumprod()

        ax1.plot(portfolio_equity.values, label='GP Portfolio', linewidth=2)
        ax1.plot(equal_weight_equity.values, label='Equal Weight', linewidth=2, alpha=0.7)
        ax1.set_title('Portfolio Performance Comparison', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Individual Asset Performance
        ax2 = axes[0, 1]
        individual_returns = results['individual_returns']
        for i, asset in enumerate(self.assets):
            if i < individual_returns.shape[1]:
                asset_returns = pd.Series(individual_returns[:, i]).fillna(0)
                asset_equity = (1 + asset_returns).cumprod()
                ax2.plot(asset_equity.values, label=asset, alpha=0.7)

        ax2.set_title('Individual Asset Performance', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cumulative Return')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot 3: Portfolio Weights Over Time
        ax3 = axes[0, 2]
        portfolio_weights = results['portfolio_weights']

        # Create stacked area plot
        valid_weights_mask = ~np.any(np.isnan(portfolio_weights), axis=1)
        if np.any(valid_weights_mask):
            weights_clean = portfolio_weights[valid_weights_mask]
            time_index = np.arange(len(weights_clean))

            ax3.stackplot(time_index, weights_clean.T, labels=self.assets, alpha=0.7)
            ax3.set_title('Portfolio Weights Over Time', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Weight')
            ax3.set_ylim(0, 1)
            ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            ax3.grid(True, alpha=0.3)

        # Plot 4: Correlation Heatmap
        ax4 = axes[0, 3]
        correlation_matrix = results['correlation_matrix']
        im = ax4.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title('Asset Correlation Matrix', fontsize=12, fontweight='bold')
        ax4.set_xticks(range(len(self.assets)))
        ax4.set_yticks(range(len(self.assets)))
        ax4.set_xticklabels(self.assets)
        ax4.set_yticklabels(self.assets)

        # Add correlation values to heatmap
        for i in range(len(self.assets)):
            for j in range(len(self.assets)):
                ax4.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                        ha='center', va='center', color='white' if abs(correlation_matrix[i, j]) > 0.5 else 'black')

        plt.colorbar(im, ax=ax4, shrink=0.6)

        # Plot 5: Risk Decomposition
        ax5 = axes[1, 0]
        multi_metrics = results['multi_asset_metrics']
        if multi_metrics.component_contributions:
            assets_risk = list(multi_metrics.component_contributions.keys())
            contributions = list(multi_metrics.component_contributions.values())

            ax5.pie(np.abs(contributions), labels=assets_risk, autopct='%1.1f%%', startangle=90)
            ax5.set_title('Risk Contribution by Asset', fontsize=12, fontweight='bold')

        # Plot 6: Sharpe Ratio Comparison
        ax6 = axes[1, 1]
        portfolio_sharpe = multi_metrics.portfolio_metrics.sharpe_ratio
        equal_weight_sharpe = results['equal_weight_metrics'].sharpe_ratio
        individual_sharpes = [metrics.sharpe_ratio for metrics in multi_metrics.individual_metrics.values()]

        x_labels = ['Portfolio'] + self.assets + ['Equal Weight']
        sharpe_values = [portfolio_sharpe] + individual_sharpes + [equal_weight_sharpe]
        colors = ['green'] + ['blue'] * len(self.assets) + ['orange']

        bars = ax6.bar(range(len(x_labels)), sharpe_values, color=colors, alpha=0.7)
        ax6.set_title('Sharpe Ratio Comparison', fontsize=12, fontweight='bold')
        ax6.set_xticks(range(len(x_labels)))
        ax6.set_xticklabels(x_labels, rotation=45)
        ax6.set_ylabel('Sharpe Ratio')
        ax6.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, sharpe_values):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')

        # Plot 7: Prediction Accuracy by Asset
        ax7 = axes[1, 2]
        predictions = results['predictions']
        targets = results['individual_returns']

        accuracies = []
        for i, asset in enumerate(self.assets):
            if i < predictions.shape[1] and i < targets.shape[1]:
                pred_asset = predictions[:, i]
                target_asset = targets[:, i]

                valid_mask = ~(np.isnan(pred_asset) | np.isnan(target_asset))
                if np.sum(valid_mask) > 0:
                    pred_valid = pred_asset[valid_mask]
                    target_valid = target_asset[valid_mask]

                    direction_accuracy = np.mean(np.sign(pred_valid) == np.sign(target_valid))
                    accuracies.append(direction_accuracy)
                else:
                    accuracies.append(0)

        ax7.bar(range(len(self.assets)), accuracies, alpha=0.7)
        ax7.set_title('Direction Accuracy by Asset', fontsize=12, fontweight='bold')
        ax7.set_xticks(range(len(self.assets)))
        ax7.set_xticklabels(self.assets, rotation=45)
        ax7.set_ylabel('Direction Accuracy')
        ax7.set_ylim(0, 1)
        ax7.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
        ax7.grid(True, alpha=0.3)

        # Plot 8: Rolling Sharpe Ratios
        ax8 = axes[1, 3]
        window = min(50, len(portfolio_returns) // 4)
        if window > 10:
            portfolio_rolling_sharpe = (portfolio_returns.rolling(window).mean() /
                                       portfolio_returns.rolling(window).std() *
                                       np.sqrt(self.annualization_factor))
            equal_weight_rolling_sharpe = (equal_weight_returns.rolling(window).mean() /
                                          equal_weight_returns.rolling(window).std() *
                                          np.sqrt(self.annualization_factor))

            ax8.plot(portfolio_rolling_sharpe.values, label='GP Portfolio', linewidth=2)
            ax8.plot(equal_weight_rolling_sharpe.values, label='Equal Weight', linewidth=2, alpha=0.7)
            ax8.set_title(f'Rolling Sharpe Ratio ({window} periods)', fontsize=12, fontweight='bold')
            ax8.set_ylabel('Sharpe Ratio')
            ax8.legend()
            ax8.grid(True, alpha=0.3)

        # Plot 9: Drawdown Comparison
        ax9 = axes[2, 0]
        portfolio_cumulative = (1 + portfolio_returns.fillna(0)).cumprod()
        portfolio_peak = portfolio_cumulative.expanding().max()
        portfolio_drawdown = (portfolio_cumulative - portfolio_peak) / portfolio_peak * 100

        equal_weight_cumulative = (1 + equal_weight_returns.fillna(0)).cumprod()
        equal_weight_peak = equal_weight_cumulative.expanding().max()
        equal_weight_drawdown = (equal_weight_cumulative - equal_weight_peak) / equal_weight_peak * 100

        ax9.fill_between(range(len(portfolio_drawdown)), portfolio_drawdown.values, 0,
                        alpha=0.6, color='red', label='GP Portfolio')
        ax9.fill_between(range(len(equal_weight_drawdown)), equal_weight_drawdown.values, 0,
                        alpha=0.4, color='blue', label='Equal Weight')
        ax9.set_title('Drawdown Comparison (%)', fontsize=12, fontweight='bold')
        ax9.set_ylabel('Drawdown (%)')
        ax9.legend()
        ax9.grid(True, alpha=0.3)

        # Plot 10: Portfolio Turnover
        ax10 = axes[2, 1]
        if np.any(~np.isnan(portfolio_weights).all(axis=1)):
            weight_changes = np.diff(portfolio_weights[~np.isnan(portfolio_weights).any(axis=1)], axis=0)
            turnover_series = np.sum(np.abs(weight_changes), axis=1)

            ax10.plot(turnover_series, alpha=0.7)
            ax10.set_title('Portfolio Turnover Over Time', fontsize=12, fontweight='bold')
            ax10.set_ylabel('Turnover')
            ax10.grid(True, alpha=0.3)

        # Plot 11: Prediction vs Actual Scatter (Portfolio)
        ax11 = axes[2, 2]
        portfolio_predictions = np.sum(predictions * portfolio_weights, axis=1)
        portfolio_targets = portfolio_returns.fillna(0).values

        valid_mask = ~(np.isnan(portfolio_predictions) | np.isnan(portfolio_targets))
        if np.sum(valid_mask) > 0:
            pred_valid = portfolio_predictions[valid_mask]
            target_valid = portfolio_targets[valid_mask]

            ax11.scatter(target_valid, pred_valid, alpha=0.6, s=20)
            ax11.plot([target_valid.min(), target_valid.max()],
                     [target_valid.min(), target_valid.max()], 'r--', alpha=0.8)
            ax11.set_xlabel('Actual Portfolio Returns')
            ax11.set_ylabel('Predicted Portfolio Returns')
            ax11.set_title('Portfolio: Predictions vs Actual', fontsize=12, fontweight='bold')
            ax11.grid(True, alpha=0.3)

        # Plot 12: Performance Metrics Summary
        ax12 = axes[2, 3]

        # Create performance metrics table
        metrics_data = {
            'Portfolio': [
                f"{multi_metrics.portfolio_metrics.total_return:.2%}",
                f"{multi_metrics.portfolio_metrics.annualized_return:.2%}",
                f"{multi_metrics.portfolio_metrics.sharpe_ratio:.2f}",
                f"{multi_metrics.portfolio_metrics.max_drawdown:.2%}",
                f"{multi_metrics.portfolio_metrics.volatility:.2%}"
            ],
            'Equal Weight': [
                f"{results['equal_weight_metrics'].total_return:.2%}",
                f"{results['equal_weight_metrics'].annualized_return:.2%}",
                f"{results['equal_weight_metrics'].sharpe_ratio:.2f}",
                f"{results['equal_weight_metrics'].max_drawdown:.2%}",
                f"{results['equal_weight_metrics'].volatility:.2%}"
            ]
        }

        metrics_labels = ['Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Volatility']

        # Create table
        table_data = []
        for i, label in enumerate(metrics_labels):
            table_data.append([label, metrics_data['Portfolio'][i], metrics_data['Equal Weight'][i]])

        table = ax12.table(cellText=table_data,
                          colLabels=['Metric', 'GP Portfolio', 'Equal Weight'],
                          cellLoc='center',
                          loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        ax12.axis('off')
        ax12.set_title('Performance Summary', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()

        # Print detailed performance comparison
        self._print_multi_asset_performance(results)

    def _print_multi_asset_performance(self, results: Dict) -> None:
        """Print comprehensive multi-asset performance analysis."""
        multi_metrics = results['multi_asset_metrics']
        equal_weight_metrics = results['equal_weight_metrics']

        print("\n" + "="*100)
        print("MULTI-ASSET GAUSSIAN PROCESS TRADING SYSTEM - COMPREHENSIVE PERFORMANCE ANALYSIS")
        print("="*100)

        # Portfolio-level comparison
        print(f"\n{'PORTFOLIO-LEVEL PERFORMANCE':<50}")
        print("-" * 100)
        print(f"{'Metric':<30} {'GP Portfolio':<20} {'Equal Weight':<20} {'Difference':<20}")
        print("-" * 100)

        portfolio_metrics = [
            ('Total Return', multi_metrics.portfolio_metrics.total_return, equal_weight_metrics.total_return),
            ('Annualized Return', multi_metrics.portfolio_metrics.annualized_return, equal_weight_metrics.annualized_return),
            ('Sharpe Ratio', multi_metrics.portfolio_metrics.sharpe_ratio, equal_weight_metrics.sharpe_ratio),
            ('Sortino Ratio', multi_metrics.portfolio_metrics.sortino_ratio, equal_weight_metrics.sortino_ratio),
            ('Max Drawdown', multi_metrics.portfolio_metrics.max_drawdown, equal_weight_metrics.max_drawdown),
            ('Volatility', multi_metrics.portfolio_metrics.volatility, equal_weight_metrics.volatility),
            ('Calmar Ratio', multi_metrics.portfolio_metrics.calmar_ratio, equal_weight_metrics.calmar_ratio)
        ]

        for name, portfolio_val, benchmark_val in portfolio_metrics:
            if 'Return' in name or 'Drawdown' in name or 'Volatility' in name:
                port_str = f"{portfolio_val:.2%}"
                bench_str = f"{benchmark_val:.2%}"
                diff_str = f"{portfolio_val - benchmark_val:.2%}"
            else:
                port_str = f"{portfolio_val:.3f}"
                bench_str = f"{benchmark_val:.3f}"
                diff_str = f"{portfolio_val - benchmark_val:.3f}"

            print(f"{name:<30} {port_str:<20} {bench_str:<20} {diff_str:<20}")

        # Individual asset performance
        print(f"\n{'INDIVIDUAL ASSET PERFORMANCE':<50}")
        print("-" * 100)
        print(f"{'Asset':<15} {'Total Return':<15} {'Annual Return':<15} {'Sharpe Ratio':<15} {'Max Drawdown':<15} {'Volatility':<15}")
        print("-" * 100)

        for asset, metrics in multi_metrics.individual_metrics.items():
            print(f"{asset:<15} {metrics.total_return:>14.2%} {metrics.annualized_return:>14.2%} "
                  f"{metrics.sharpe_ratio:>14.2f} {metrics.max_drawdown:>14.2%} {metrics.volatility:>14.2%}")

        # Portfolio composition and risk analysis
        print(f"\n{'PORTFOLIO COMPOSITION & RISK ANALYSIS':<50}")
        print("-" * 100)
        print(f"{'Asset':<15} {'Avg Weight':<15} {'Risk Contrib':<15} {'Marginal Contrib':<20}")
        print("-" * 100)

        for i, asset in enumerate(self.assets):
            avg_weight = multi_metrics.asset_weights[i]
            risk_contrib = multi_metrics.component_contributions.get(asset, 0)
            marginal_contrib = multi_metrics.marginal_contributions.get(asset, 0)

            print(f"{asset:<15} {avg_weight:>14.2%} {risk_contrib:>14.3f} {marginal_contrib:>19.3f}")

        # System statistics
        print(f"\n{'SYSTEM STATISTICS':<50}")
        print("-" * 100)
        print(f"{'Number of Assets':<30} {self.n_assets:<20}")
        print(f"{'Diversification Ratio':<30} {multi_metrics.diversification_ratio:<20.3f}")
        print(f"{'Portfolio Turnover':<30} {multi_metrics.portfolio_turnover:<20.3f}")
        print(f"{'Rebalancing Frequency':<30} {self.rebalance_frequency:<20} periods")
        print(f"{'Data Frequency':<30} {self.frequency:<20}")
        print(f"{'Transaction Cost':<30} {self.transaction_cost*10000:<20.1f} bps")
        print(f"{'Max Position Size':<30} {self.max_weight:<20.1%}")

        # Prediction accuracy by asset
        predictions = results['predictions']
        targets = results['individual_returns']

        print(f"\n{'PREDICTION ACCURACY BY ASSET':<50}")
        print("-" * 100)
        print(f"{'Asset':<15} {'Direction Accuracy':<20} {'Correlation':<15} {'MSE':<15}")
        print("-" * 100)

        for i, asset in enumerate(self.assets):
            if i < predictions.shape[1] and i < targets.shape[1]:
                pred_asset = predictions[:, i]
                target_asset = targets[:, i]

                valid_mask = ~(np.isnan(pred_asset) | np.isnan(target_asset))
                if np.sum(valid_mask) > 0:
                    pred_valid = pred_asset[valid_mask]
                    target_valid = target_asset[valid_mask]

                    direction_accuracy = np.mean(np.sign(pred_valid) == np.sign(target_valid))
                    correlation = np.corrcoef(pred_valid, target_valid)[0, 1] if len(pred_valid) > 1 else 0
                    mse = np.mean((pred_valid - target_valid)**2)

                    print(f"{asset:<15} {direction_accuracy:>19.2%} {correlation:>14.3f} {mse:>14.6f}")

        print("="*100)

def get_multi_asset_user_inputs():
    """Get user inputs for multi-asset trading system configuration."""
    print("\n" + "="*80)
    print("MULTI-ASSET GAUSSIAN PROCESS TRADING SYSTEM CONFIGURATION")
    print("="*80)

    # Get tickers
    tickers_input = input("Enter asset tickers separated by commas (e.g., SPY,QQQ,IWM): ").strip()
    if not tickers_input:
        tickers = ["SPY", "QQQ", "IWM"]
        print(f"Using default tickers: {tickers}")
    else:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]

    # Get date range
    start_date = input("Enter start date (YYYY-MM-DD) or press Enter for default: ").strip()
    if not start_date:
        start_date = "2020-01-01"
        print(f"Using default start date: {start_date}")

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

    # Get risk aversion
    risk_aversion_input = input("Enter risk aversion parameter (higher = more conservative) or press Enter for default: ").strip()
    if not risk_aversion_input:
        risk_aversion = 1.0
        print(f"Using default risk aversion: {risk_aversion}")
    else:
        try:
            risk_aversion = float(risk_aversion_input)
        except ValueError:
            risk_aversion = 1.0
            print(f"Invalid input, using default risk aversion: {risk_aversion}")

    # Get max position size
    max_weight_input = input("Enter maximum position size per asset (e.g., 0.4 for 40%) or press Enter for default: ").strip()
    if not max_weight_input:
        max_weight = 0.4
        print(f"Using default max weight: {max_weight:.1%}")
    else:
        try:
            max_weight = float(max_weight_input)
        except ValueError:
            max_weight = 0.4
            print(f"Invalid input, using default max weight: {max_weight:.1%}")

    return tickers, start_date, end_date, frequency, risk_aversion, max_weight

def main_multi_asset():
    """
    Main function for multi-asset GP trading system demonstration.
    """
    np.random.seed(42)

    print("Multi-Asset Gaussian Process Trading System")
    print("CROSS-ASSET CORRELATIONS: ENABLED")
    print("PORTFOLIO OPTIMIZATION: ENABLED")
    print("LOOK-AHEAD BIAS PROTECTION: ENABLED")

    # Get user configuration
    tickers, start_date, end_date, frequency, risk_aversion, max_weight = get_multi_asset_user_inputs()

    print(f"\nConfiguration Summary:")
    print(f"  Assets: {', '.join(tickers)}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Frequency: {frequency}")
    print(f"  Risk Aversion: {risk_aversion}")
    print(f"  Max Weight per Asset: {max_weight:.1%}")

    # Download data for all assets
    print(f"\nDownloading price data for {len(tickers)} assets...")
    price_data = {}

    try:
        for ticker in tickers:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                print(f"No data found for {ticker}, skipping...")
                continue

            price_data[ticker] = data['Close'].dropna()
            print(f"  {ticker}: {len(price_data[ticker])} price points")

        if len(price_data) < 2:
            print("Need at least 2 assets with valid data.")
            return None

        # Update tickers list to only include assets with data
        tickers = list(price_data.keys())

    except Exception as e:
        print(f"Error downloading data: {e}")
        return None

    # Initialize multi-asset trading system
    trading_system = MultiAssetTradingSystem(
        assets=tickers,
        n_lags=3,
        max_degree=3,
        lookback_window=63 if frequency == 'D' else (26 if frequency == 'W' else 12),
        min_periods=21 if frequency == 'D' else (8 if frequency == 'W' else 6),
        include_cross_terms=True,
        risk_aversion=risk_aversion,
        max_weight=max_weight,
        transaction_cost=0.0005,
        frequency=frequency,
        rebalance_frequency=5
    )

    print(f"\nMulti-Asset Trading System Configuration:")
    print(f"  - Number of Assets: {len(tickers)}")
    print(f"  - Lagged Returns per Asset: {trading_system.n_lags}")
    print(f"  - Max Polynomial Degree: {trading_system.max_degree}")
    print(f"  - Cross-Asset Terms: {trading_system.include_cross_terms}")
    print(f"  - Lookback Window: {trading_system.lookback_window} periods")
    print(f"  - Minimum Periods: {trading_system.min_periods} periods")
    print(f"  - Risk Aversion: {trading_system.risk_aversion}")
    print(f"  - Max Weight per Asset: {trading_system.max_weight:.1%}")
    print(f"  - Transaction Cost: {trading_system.transaction_cost*10000:.1f} bps")
    print(f"  - Rebalancing Frequency: {trading_system.rebalance_frequency} periods")
    print(f"  - Frequency: {frequency}")
    print(f"  - Annualization Factor: {trading_system.annualization_factor}")

    # Data integrity check
    min_length = min(len(prices) for prices in price_data.values())
    print(f"\nData Integrity Check:")
    print(f"  - Minimum price observations across assets: {min_length}")

    for ticker, prices in price_data.items():
        returns = prices.pct_change().dropna()
        print(f"  - {ticker}: {len(prices)} prices, {len(returns)} returns")

    # Check if we have enough data
    min_required = trading_system.n_lags + trading_system.min_periods + 10
    if min_length < min_required:
        print(f"\nInsufficient data: need at least {min_required} observations, got {min_length}")
        print("Try using a longer date range or lower frequency.")
        return None

    # Run multi-asset backtest
    print(f"\nRunning multi-asset backtest with portfolio optimization...")
    start_time = pd.Timestamp.now()

    try:
        results = trading_system.backtest(price_data, optimize_hyperparams=True)

        end_time = pd.Timestamp.now()
        print(f"Multi-asset backtest completed in {(end_time - start_time).total_seconds():.1f} seconds")

        # Display results
        trading_system.plot_multi_asset_results(results)

        return results

    except Exception as e:
        print(f"Error during multi-asset backtest: {e}")
        import traceback
        traceback.print_exc()
        return None

# Advanced multi-asset features
class RiskParityPortfolioOptimizer(PortfolioOptimizer):
    """Risk parity portfolio optimization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def optimize_portfolio(self,
                          expected_returns: np.ndarray,
                          covariance_matrix: np.ndarray,
                          current_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize for equal risk contribution (risk parity).
        """
        n_assets = len(expected_returns)

        def risk_parity_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(covariance_matrix, weights)))

            # Marginal risk contributions
            marginal_contrib = np.dot(covariance_matrix, weights) / portfolio_vol

            # Risk contributions
            risk_contrib = weights * marginal_contrib

            # Target: equal risk contribution (1/n each)
            target_contrib = np.ones(n_assets) / n_assets

            # Minimize squared deviations from equal risk contribution
            return np.sum((risk_contrib - target_contrib)**2)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Fully invested
        ]

        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]

        # Initial guess
        x0 = np.ones(n_assets) / n_assets

        try:
            result = minimize(
                risk_parity_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            if result.success:
                return result.x
            else:
                return np.ones(n_assets) / n_assets

        except Exception:
            return np.ones(n_assets) / n_assets

class AdaptiveMultiAssetSystem(MultiAssetTradingSystem):
    """Multi-asset system with adaptive parameters based on market regime."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.volatility_lookback = 20
        self.correlation_threshold = 0.7

    def detect_market_regime(self, returns_data: Dict[str, pd.Series]) -> str:
        """Detect current market regime based on volatility and correlations."""

        # Calculate recent volatility
        recent_vols = {}
        for asset, returns in returns_data.items():
            if len(returns) >= self.volatility_lookback:
                recent_vol = returns.tail(self.volatility_lookback).std()
                long_vol = returns.std()
                recent_vols[asset] = recent_vol / long_vol

        avg_vol_ratio = np.mean(list(recent_vols.values()))

        # Calculate recent correlations
        recent_returns = pd.DataFrame({
            asset: returns.tail(self.volatility_lookback)
            for asset, returns in returns_data.items()
        }).dropna()

        if len(recent_returns) > 5:
            recent_corr_matrix = recent_returns.corr()
            # Average off-diagonal correlation
            mask = ~np.eye(recent_corr_matrix.shape[0], dtype=bool)
            avg_correlation = recent_corr_matrix.values[mask].mean()
        else:
            avg_correlation = 0.5

        # Classify regime
        if avg_vol_ratio > 1.5 and avg_correlation > self.correlation_threshold:
            return "crisis"
        elif avg_vol_ratio > 1.2:
            return "volatile"
        elif avg_correlation > self.correlation_threshold:
            return "high_correlation"
        else:
            return "normal"

    def adapt_parameters(self, regime: str):
        """Adapt system parameters based on detected regime."""

        if regime == "crisis":
            # More conservative in crisis
            self.risk_aversion = self.risk_aversion * 2.0
            self.max_weight = min(self.max_weight, 0.3)
            self.rebalance_frequency = max(self.rebalance_frequency, 10)

        elif regime == "volatile":
            # Slightly more conservative
            self.risk_aversion = self.risk_aversion * 1.5
            self.rebalance_frequency = max(self.rebalance_frequency, 7)

        elif regime == "high_correlation":
            # Reduce position limits when correlations are high
            self.max_weight = min(self.max_weight, 0.35)

        # Normal regime uses base parameters

# Utility functions for multi-asset analysis
def calculate_portfolio_attribution(weights: np.ndarray,
                                   returns: np.ndarray,
                                   asset_names: List[str]) -> Dict[str, float]:
    """Calculate portfolio performance attribution by asset."""

    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    if returns.ndim == 1:
        returns = returns.reshape(1, -1)

    # Ensure compatible shapes
    min_periods = min(len(weights), len(returns))
    weights = weights[:min_periods]
    returns = returns[:min_periods]

    # Calculate contributions
    contributions = weights * returns
    total_contributions = np.sum(contributions, axis=0)

    attribution = {}
    for i, asset in enumerate(asset_names):
        if i < len(total_contributions):
            attribution[asset] = total_contributions[i]

    return attribution

def analyze_correlation_stability(returns_data: Dict[str, pd.Series],
                                window: int = 60) -> pd.DataFrame:
    """Analyze correlation stability over time."""

    returns_df = pd.DataFrame(returns_data).dropna()
    assets = list(returns_data.keys())

    # Rolling correlations
    rolling_corrs = {}

    for i, asset1 in enumerate(assets):
        for j, asset2 in enumerate(assets[i+1:], i+1):
            pair = f"{asset1}_{asset2}"
            rolling_corr = returns_df[asset1].rolling(window).corr(returns_df[asset2])
            rolling_corrs[pair] = rolling_corr

    return pd.DataFrame(rolling_corrs)

def stress_test_portfolio(weights: np.ndarray,
                         returns_data: Dict[str, pd.Series],
                         stress_scenarios: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Stress test portfolio under various scenarios."""

    stress_results = {}

    for scenario_name, scenario_returns in stress_scenarios.items():
        scenario_return = 0

        for i, (asset, weight) in enumerate(zip(returns_data.keys(), weights)):
            if asset in scenario_returns:
                scenario_return += weight * scenario_returns[asset]

        stress_results[scenario_name] = scenario_return

    return stress_results

# Example usage and testing
if __name__ == "__main__":
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    try:
        results = main_multi_asset()

        if results is not None:
            print("\n" + "="*80)
            print("Multi-Asset GP Trading System completed successfully!")
            print("Results available in 'results' variable for further analysis.")

            # Additional analysis examples
            multi_metrics = results['multi_asset_metrics']

            print(f"\nQuick Summary:")
            print(f"  - Portfolio Sharpe Ratio: {multi_metrics.portfolio_metrics.sharpe_ratio:.3f}")
            print(f"  - Portfolio Annual Return: {multi_metrics.portfolio_metrics.annualized_return:.2%}")
            print(f"  - Portfolio Max Drawdown: {multi_metrics.portfolio_metrics.max_drawdown:.2%}")
            print(f"  - Diversification Ratio: {multi_metrics.diversification_ratio:.3f}")
            print(f"  - Portfolio Turnover: {multi_metrics.portfolio_turnover:.3f}")

            print("="*80)

    except KeyboardInterrupt:
        print("\nMulti-asset backtest interrupted by user.")
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()