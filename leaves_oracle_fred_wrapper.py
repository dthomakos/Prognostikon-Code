"""
LEAVES-IN-THE-WIND ORACLE: Complete Wrapper with FRED Integration
=================================================================

A comprehensive forecasting framework combining:
- Standard Leaves Oracle (v1.0) - Unit root design (Σφ=1)
- Hybrid Leaves Oracle (v2.0) - Mean reversion extension (Σφ=1-λ)
- FRED data integration via pandas_datareader
- Rolling window backtesting with benchmarks
- Publication-ready analysis and reporting

Author: Dimitrios Thomakos with prompts to Claude Sonnet 4.5 in Perplexity
Version: 3.0
Date: December 2025, updated February 2026
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from scipy import stats
import warnings
from datetime import datetime

# FRED Data Access
try:
    import pandas_datareader as pdr
    from pandas_datareader.data import DataReader
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    warnings.warn("pandas_datareader not available. Install with: pip install pandas-datareader")


# =============================================================================
# DATA CLASSES AND CONTAINERS
# =============================================================================

@dataclass
class ForecastResult:
    """Container for forecast outputs with full diagnostic information"""
    point_forecast: float
    lower_bound: float
    upper_bound: float
    velocity: float
    acceleration: float
    coherence: float
    regime: str
    lambda_t: float = 0.0
    model_name: str = ""


@dataclass
class BacktestMetrics:
    """Container for backtest evaluation metrics"""
    model_name: str
    mae: float
    rmse: float
    mape: float
    coverage: float
    sharpness: float
    hit_rate: float  # Direction accuracy
    avg_lambda: float = 0.0
    avg_coherence: float = 0.0
    n_forecasts: int = 0

    def to_dict(self) -> dict:
        return {
            'Model': self.model_name,
            'MAE': self.mae,
            'RMSE': self.rmse,
            'MAPE': self.mape,
            'Coverage': self.coverage,
            'Sharpness': self.sharpness,
            'Hit_Rate': self.hit_rate,
            'Avg_Lambda': self.avg_lambda,
            'Avg_Coherence': self.avg_coherence,
            'N_Forecasts': self.n_forecasts
        }


@dataclass
class FREDSeriesInfo:
    """Information about a FRED series"""
    series_id: str
    name: str
    frequency: str
    units: str
    description: str = ""


# =============================================================================
# FRED DATA MANAGER
# =============================================================================

class FREDDataManager:
    """
    Manager for accessing and preprocessing FRED economic data
    """

    # Common FRED series
    COMMON_SERIES = {
        'CPI': FREDSeriesInfo('CPIAUCSL', 'CPI All Items', 'Monthly', 'Index 1982-84=100'),
        'INFLATION': FREDSeriesInfo('CPIAUCSL', 'Inflation Rate (YoY)', 'Monthly', 'Percent'),
        'UNEMPLOYMENT': FREDSeriesInfo('UNRATE', 'Unemployment Rate', 'Monthly', 'Percent'),
        'GDP': FREDSeriesInfo('GDPC1', 'Real Gross Domestic Product', 'Quarterly', 'Billions $'),
        'FEDFUNDS': FREDSeriesInfo('FEDFUNDS', 'Federal Funds Rate', 'Monthly', 'Percent'),
        'SP500': FREDSeriesInfo('SP500', 'S&P 500 Index', 'Daily', 'Index'),
        'T10Y2Y': FREDSeriesInfo('T10Y2Y', '10Y-2Y Treasury Spread', 'Daily', 'Percent'),
        'INDPRO': FREDSeriesInfo('INDPRO', 'Industrial Production', 'Monthly', 'Index 2017=100'),
    }

    def __init__(self, start_date: str = '1990-01-01', end_date: Optional[str] = None):
        """
        Initialize FRED data manager

        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str, optional
            End date (defaults to today)
        """
        if not FRED_AVAILABLE:
            raise ImportError("pandas_datareader required. Install with: pip install pandas-datareader")

        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self._cache = {}

    def fetch_series(self, series_id: str, transform: Optional[str] = None, yoy_periods: int = 12) -> pd.Series:
        """
        Fetch a FRED series with optional transformation

        Parameters:
        -----------
        series_id : str
            FRED series identifier
        transform : str, optional
            Transformation: 'pct_change', 'yoy', 'diff', 'log', None
        """
        cache_key = f"{series_id}_{transform}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # Fetch raw data
        data = pdr.DataReader(series_id, 'fred', self.start_date, self.end_date)
        series = data[series_id].dropna()

        # Apply transformation
        if transform == 'pct_change':
            series = series.pct_change() * 100
        elif transform == 'yoy':
            # Year-over-year percent change
            series = series.pct_change(periods=yoy_periods) * 100
        elif transform == 'diff':
            series = series.diff()
        elif transform == 'log':
            series = np.log(series)

        series = series.dropna()
        self._cache[cache_key] = series

        return series

    def fetch_inflation_rate(self) -> pd.Series:
        """Fetch annual inflation rate (YoY CPI change)"""
        return self.fetch_series('CPIAUCSL', transform='yoy')

    def fetch_unemployment_rate(self) -> pd.Series:
        """Fetch unemployment rate"""
        return self.fetch_series('UNRATE', transform=None)

    def fetch_multiple(self, series_dict: Dict[str, Tuple[str, Optional[str]]]) -> pd.DataFrame:
        """
        Fetch multiple series

        Parameters:
        -----------
        series_dict : dict
            {name: (series_id, transform)}
        """
        result = {}
        for name, (series_id, transform) in series_dict.items():
            result[name] = self.fetch_series(series_id, transform)

        df = pd.DataFrame(result)
        return df.dropna()


# =============================================================================
# BASE FORECASTER CLASS
# =============================================================================

class BaseForecaster(ABC):
    """Abstract base class for all forecasting models"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def forecast(self, y: np.ndarray, horizon: int = 1) -> ForecastResult:
        """Generate forecast with uncertainty bounds"""
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


# =============================================================================
# STANDARD LEAVES ORACLE (v1.0)
# =============================================================================

class LeavesInWindOracle(BaseForecaster):
    """
    Standard Leaves-in-the-Wind Oracle (Unit Root version)

    Properties:
    - Σφ_j = 1 (unit coefficient sum)
    - Forecast increment = κ_t·v̄_t + ā_t
    - Suitable for I(1), trending, regime-switching data
    """

    def __init__(self, 
                 velocity_window: int = 5,
                 acceleration_window: int = 3,
                 alpha: float = 0.05,
                 uncertainty_multiplier: float = 1.5,
                 name: str = "LeavesOracle"):
        super().__init__(name)
        self.w = velocity_window
        self.w_prime = acceleration_window
        self.alpha = alpha
        self.uncertainty_multiplier = uncertainty_multiplier
        self.z_critical = stats.norm.ppf(1 - alpha/2)

    def _compute_velocity(self, y: np.ndarray) -> np.ndarray:
        return np.diff(y)

    def _compute_acceleration(self, v: np.ndarray) -> np.ndarray:
        return np.diff(v)

    def _compute_directional_momentum(self, v: np.ndarray, window: int) -> float:
        if len(v) < window:
            window = len(v)
        if window == 0:
            return 0.0
        recent_v = v[-window:]
        signs = np.sign(recent_v)
        return np.mean(signs)

    def _compute_coherence(self, momentum: float) -> float:
        return np.abs(momentum)

    def _identify_regime(self, v_current: float, a_current: float) -> str:
        if np.sign(v_current) == np.sign(a_current) and a_current != 0:
            return 'momentum'
        else:
            return 'reversal'

    def _compute_velocity_dispersion(self, v: np.ndarray, window: int) -> float:
        if len(v) < window:
            window = len(v)
        if window <= 1:
            return np.std(v) if len(v) > 0 else 1.0
        recent_v = v[-window:]
        return np.std(recent_v, ddof=1) if len(recent_v) > 1 else np.std(v)

    def forecast(self, y: np.ndarray, horizon: int = 1) -> ForecastResult:
        if len(y) < 3:
            y_hat = y[-1] if len(y) > 0 else 0.0
            sigma = np.std(y) if len(y) > 1 else 1.0
            interval_width = self.z_critical * sigma * np.sqrt(horizon)

            return ForecastResult(
                point_forecast=y_hat,
                lower_bound=y_hat - interval_width,
                upper_bound=y_hat + interval_width,
                velocity=0.0,
                acceleration=0.0,
                coherence=0.0,
                regime='insufficient_data',
                lambda_t=0.0,
                model_name=self.name
            )

        v = self._compute_velocity(y)
        a = self._compute_acceleration(v)

        v_bar = np.mean(v[-self.w:]) if len(v) >= self.w else np.mean(v)
        a_bar = np.mean(a[-self.w_prime:]) if len(a) >= self.w_prime else (np.mean(a) if len(a) > 0 else 0.0)

        momentum = self._compute_directional_momentum(v, self.w)
        kappa = self._compute_coherence(momentum)

        v_current = v[-1] if len(v) > 0 else 0.0
        a_current = a[-1] if len(a) > 0 else 0.0
        regime = self._identify_regime(v_current, a_current)

        h = horizon
        y_current = y[-1]

        if regime == 'momentum':
            velocity_component = h * kappa * v_bar
            acceleration_component = (h * (h + 1) / 2) * a_bar
            y_hat = y_current + velocity_component + acceleration_component
        else:
            y_hat = y_current + h * 0.5 * kappa * v_bar

        sigma_v = self._compute_velocity_dispersion(v, self.w)
        model_uncertainty = (1.0 + 0.5 * (1 - kappa))
        interval_width = (self.z_critical * sigma_v * np.sqrt(horizon) * 
                         self.uncertainty_multiplier * model_uncertainty)

        return ForecastResult(
            point_forecast=y_hat,
            lower_bound=y_hat - interval_width,
            upper_bound=y_hat + interval_width,
            velocity=v_bar,
            acceleration=a_bar,
            coherence=kappa,
            regime=regime,
            lambda_t=0.0,
            model_name=self.name
        )


# =============================================================================
# HYBRID LEAVES ORACLE (v2.0)
# =============================================================================

class HybridLeavesOracle(LeavesInWindOracle):
    """
    Hybrid Leaves Oracle with Mean Reversion

    Forecast: ŷ_{t+h} = (1-λ)·[kinematic_forecast] + λ·μ̂

    Properties:
    - Σφ_j = 1-λ (stationary for λ>0)
    - Balances local momentum with long-run mean
    - Suitable for stationary, mean-reverting series
    """

    def __init__(self,
                 velocity_window: int = 5,
                 acceleration_window: int = 3,
                 alpha: float = 0.05,
                 uncertainty_multiplier: float = 1.5,
                 lambda_param: float = 0.2,
                 lambda_strategy: str = 'fixed',
                 mean_window: Optional[int] = None,
                 name: str = "HybridLeaves"):
        super().__init__(velocity_window, acceleration_window, alpha, 
                        uncertainty_multiplier, name)
        self.lambda_param = lambda_param
        self.lambda_strategy = lambda_strategy
        self.mean_window = mean_window  # Rolling mean window (None = full sample)
        self.mean_estimate = None

    def _estimate_mean(self, y: np.ndarray) -> float:
        """Estimate long-run mean"""
        if self.mean_window is not None and len(y) > self.mean_window:
            return np.mean(y[-self.mean_window:])
        return np.mean(y)

    def _compute_acf_decay(self, y: np.ndarray, max_lag: int = 5) -> float:
        """Compute ACF decay rate as indicator of mean reversion strength"""
        if len(y) < max_lag + 5:
            return 0.3

        y_demeaned = y - np.mean(y)
        c0 = np.dot(y_demeaned, y_demeaned) / len(y)

        if c0 < 1e-10:
            return 0.3

        acf_values = []
        for lag in range(1, max_lag + 1):
            ck = np.dot(y_demeaned[:-lag], y_demeaned[lag:]) / len(y)
            acf_values.append(ck / c0)

        avg_acf = np.mean(np.abs(acf_values))
        lambda_acf = np.clip(1.0 - avg_acf, 0.05, 0.5)

        return lambda_acf

    def _select_lambda(self, y: np.ndarray, kappa_t: float) -> float:
        """Select λ based on chosen strategy"""
        if self.lambda_strategy == 'fixed':
            return self.lambda_param

        elif self.lambda_strategy == 'adaptive_acf':
            return self._compute_acf_decay(y)

        elif self.lambda_strategy == 'coherence_based':
            lambda_max = self.lambda_param
            return lambda_max * (1.0 - kappa_t)

        else:
            return self.lambda_param

    def forecast(self, y: np.ndarray, horizon: int = 1) -> ForecastResult:
        base_result = super().forecast(y, horizon)

        if len(y) < 3:
            return base_result

        self.mean_estimate = self._estimate_mean(y)
        lambda_t = self._select_lambda(y, base_result.coherence)

        kinematic_forecast = base_result.point_forecast
        y_hat_hybrid = (1 - lambda_t) * kinematic_forecast + lambda_t * self.mean_estimate

        base_width = base_result.upper_bound - base_result.lower_bound
        shrinkage_factor = 1.0 - 0.3 * lambda_t
        adjusted_width = base_width * shrinkage_factor
        half_width = adjusted_width / 2

        return ForecastResult(
            point_forecast=y_hat_hybrid,
            lower_bound=y_hat_hybrid - half_width,
            upper_bound=y_hat_hybrid + half_width,
            velocity=base_result.velocity,
            acceleration=base_result.acceleration,
            coherence=base_result.coherence,
            regime=base_result.regime,
            lambda_t=lambda_t,
            model_name=self.name
        )


# =============================================================================
# ENSEMBLE ORACLE
# =============================================================================

class MultiLeavesEnsemble(BaseForecaster):
    """Multi-configuration ensemble (standard or hybrid)"""

    def __init__(self, 
                 configs: Optional[List[Tuple[int, int]]] = None,
                 alpha: float = 0.05,
                 uncertainty_multiplier: float = 2.0,
                 use_hybrid: bool = False,
                 lambda_param: float = 0.2,
                 lambda_strategy: str = 'fixed',
                 name: str = "MultiLeaves"):
        super().__init__(name)
        if configs is None:
            configs = [(3, 2), (5, 3), (7, 3), (10, 5)]

        if use_hybrid:
            self.oracles = [
                HybridLeavesOracle(w, w_prime, alpha, 
                                  uncertainty_multiplier=1.0,
                                  lambda_param=lambda_param,
                                  lambda_strategy=lambda_strategy,
                                  name=f"HybridLeaf_{w}_{w_prime}")
                for w, w_prime in configs
            ]
        else:
            self.oracles = [
                LeavesInWindOracle(w, w_prime, alpha, 
                                  uncertainty_multiplier=1.0, 
                                  name=f"Leaf_{w}_{w_prime}")
                for w, w_prime in configs
            ]

        self.alpha = alpha
        self.uncertainty_multiplier = uncertainty_multiplier
        self.use_hybrid = use_hybrid

    def forecast(self, y: np.ndarray, horizon: int = 1) -> ForecastResult:
        forecasts = [oracle.forecast(y, horizon) for oracle in self.oracles]
        point_forecasts = np.array([f.point_forecast for f in forecasts])
        y_hat = np.median(point_forecasts)

        individual_widths = np.array([f.upper_bound - f.lower_bound for f in forecasts])
        avg_individual_width = np.mean(individual_widths)

        forecast_std = np.std(point_forecasts)
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        disagreement_width = 2 * z_critical * forecast_std

        combined_width = max(avg_individual_width, disagreement_width) * self.uncertainty_multiplier
        half_width = combined_width / 2

        avg_velocity = np.mean([f.velocity for f in forecasts])
        avg_acceleration = np.mean([f.acceleration for f in forecasts])
        avg_coherence = np.mean([f.coherence for f in forecasts])
        avg_lambda = np.mean([f.lambda_t for f in forecasts])

        regimes = [f.regime for f in forecasts]
        regime = max(set(regimes), key=regimes.count)

        return ForecastResult(
            point_forecast=y_hat,
            lower_bound=y_hat - half_width,
            upper_bound=y_hat + half_width,
            velocity=avg_velocity,
            acceleration=avg_acceleration,
            coherence=avg_coherence,
            regime=regime,
            lambda_t=avg_lambda,
            model_name=self.name
        )


# =============================================================================
# BENCHMARK FORECASTERS
# =============================================================================

class NoChangeForecaster(BaseForecaster):
    """Naive random walk / no-change forecast"""

    def __init__(self, alpha: float = 0.05, name: str = "NoChange"):
        super().__init__(name)
        self.alpha = alpha

    def forecast(self, y: np.ndarray, horizon: int = 1) -> ForecastResult:
        y_hat = y[-1]

        if len(y) > 1:
            sigma = np.std(np.diff(y), ddof=1)
        else:
            sigma = 0.0

        z_critical = stats.norm.ppf(1 - self.alpha/2)
        interval_width = z_critical * sigma * np.sqrt(horizon)

        return ForecastResult(
            point_forecast=y_hat,
            lower_bound=y_hat - interval_width,
            upper_bound=y_hat + interval_width,
            velocity=0.0,
            acceleration=0.0,
            coherence=0.0,
            regime='none',
            lambda_t=0.0,
            model_name=self.name
        )


class AR1Forecaster(BaseForecaster):
    """AR(1) benchmark with OLS estimation"""

    def __init__(self, alpha: float = 0.05, window: Optional[int] = None, name: str = "AR1"):
        super().__init__(name)
        self.alpha = alpha
        self.phi0 = 0.0
        self.phi1 = 0.0
        self.sigma = 0.0
        self.window = window

    def forecast(self, y: np.ndarray, horizon: int = 1) -> ForecastResult:
        if len(y) < 10:
            return NoChangeForecaster(self.alpha, self.name).forecast(y, horizon)
        else:
            if self.window and len(y) > self.window:
                y_window = y[-self.window:]
            else:
                y_window = y
        # OLS estimation
        y_lag = y_window[:-1]
        y_current = y_window[1:]

        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            beta = np.linalg.lstsq(X, y_current, rcond=None)[0]
            self.phi0 = beta[0]
            self.phi1 = beta[1]
        except:
            self.phi0 = 0.0
            self.phi1 = 1.0

        y_fitted = self.phi0 + self.phi1 * y_lag
        residuals = y_current - y_fitted
        self.sigma = np.std(residuals, ddof=2)

        y_last = y_window[-1]

        if np.abs(self.phi1) < 0.999:
            mean_y = self.phi0 / (1 - self.phi1) if self.phi1 != 1 else y_last
            y_hat = mean_y + (self.phi1 ** horizon) * (y_last - mean_y)

            if self.phi1 != 1 and np.abs(self.phi1) < 1:
                var_h = self.sigma**2 * (1 - self.phi1**(2*horizon)) / (1 - self.phi1**2)
            else:
                var_h = self.sigma**2 * horizon
        else:
            y_hat = y_last + horizon * self.phi0
            var_h = self.sigma**2 * horizon

        z_critical = stats.norm.ppf(1 - self.alpha/2)
        interval_width = z_critical * np.sqrt(max(var_h, 1e-10))

        return ForecastResult(
            point_forecast=y_hat,
            lower_bound=y_hat - interval_width,
            upper_bound=y_hat + interval_width,
            velocity=0.0,
            acceleration=0.0,
            coherence=0.0,
            regime='none',
            lambda_t=0.0,
            model_name=self.name
        )


class MeanForecast(BaseForecaster):
    """Historical mean forecast"""

    def __init__(self, alpha: float = 0.05, window: Optional[int] = None, name: str = "Mean"):
        super().__init__(name)
        self.alpha = alpha
        self.window = window

    def forecast(self, y: np.ndarray, horizon: int = 1) -> ForecastResult:
        if self.window and len(y) > self.window:
            y_window = y[-self.window:]
        else:
            y_window = y

        y_hat = np.mean(y_window)
        sigma = np.std(y_window, ddof=1)

        z_critical = stats.norm.ppf(1 - self.alpha/2)
        interval_width = z_critical * sigma / np.sqrt(len(y_window))

        return ForecastResult(
            point_forecast=y_hat,
            lower_bound=y_hat - interval_width,
            upper_bound=y_hat + interval_width,
            velocity=0.0,
            acceleration=0.0,
            coherence=0.0,
            regime='none',
            lambda_t=0.0,
            model_name=self.name
        )


# =============================================================================
# ROLLING WINDOW BACKTEST ENGINE
# =============================================================================

class RollingWindowBacktest:
    """
    Rolling window backtesting engine with comprehensive evaluation
    """

    def __init__(self,
                 forecasters: List[BaseForecaster],
                 window_size: int = 60,
                 horizon: int = 1,
                 step_size: int = 1):
        """
        Parameters:
        -----------
        forecasters : list
            List of forecaster objects
        window_size : int
            Size of rolling estimation window
        horizon : int
            Forecast horizon
        step_size : int
            Step size for rolling window (1 = every period)
        """
        self.forecasters = forecasters
        self.window_size = window_size
        self.horizon = horizon
        self.step_size = step_size
        self.results_df = None

    def run(self, y: Union[np.ndarray, pd.Series], 
            dates: Optional[pd.DatetimeIndex] = None) -> pd.DataFrame:
        """
        Execute rolling window backtest

        Parameters:
        -----------
        y : array-like
            Time series data
        dates : DatetimeIndex, optional
            Date index for results
        """
        if isinstance(y, pd.Series):
            dates = y.index if dates is None else dates
            y = y.values

        n = len(y)
        n_forecasts = (n - self.window_size - self.horizon + 1) // self.step_size

        if n_forecasts <= 0:
            raise ValueError(f"Series too short. Need at least {self.window_size + self.horizon} observations.")

        results = []

        for i in range(0, n - self.window_size - self.horizon + 1, self.step_size):
            train_end = self.window_size + i
            y_train = y[i:train_end]
            y_actual = y[train_end + self.horizon - 1]

            forecast_date = dates[train_end + self.horizon - 1] if dates is not None else train_end + self.horizon - 1

            row = {
                'forecast_origin': train_end - 1,
                'forecast_target': train_end + self.horizon - 1,
                'date': forecast_date,
                'actual': y_actual
            }

            for forecaster in self.forecasters:
                result = forecaster.forecast(y_train, self.horizon)

                row[f'{forecaster.name}_forecast'] = result.point_forecast
                row[f'{forecaster.name}_lower'] = result.lower_bound
                row[f'{forecaster.name}_upper'] = result.upper_bound
                row[f'{forecaster.name}_error'] = y_actual - result.point_forecast
                row[f'{forecaster.name}_lambda'] = result.lambda_t
                row[f'{forecaster.name}_coherence'] = result.coherence
                row[f'{forecaster.name}_regime'] = result.regime

            results.append(row)

        self.results_df = pd.DataFrame(results)
        return self.results_df

    def evaluate(self, alpha: float = 0.05) -> pd.DataFrame:
        """Compute evaluation metrics for all forecasters"""
        if self.results_df is None:
            raise ValueError("Must run backtest first")

        actuals = self.results_df['actual'].values
        metrics_list = []

        for forecaster in self.forecasters:
            name = forecaster.name
            forecasts = self.results_df[f'{name}_forecast'].values
            lower = self.results_df[f'{name}_lower'].values
            upper = self.results_df[f'{name}_upper'].values

            # Error metrics
            errors = actuals - forecasts
            abs_errors = np.abs(errors)
            sq_errors = errors ** 2

            mae = np.mean(abs_errors)
            rmse = np.sqrt(np.mean(sq_errors))

            # MAPE (avoiding division by zero)
            mask = actuals != 0
            mape = np.mean(np.abs(errors[mask] / actuals[mask]) * 100) if np.any(mask) else np.inf

            # Coverage and sharpness
            in_interval = (actuals >= lower) & (actuals <= upper)
            coverage = np.mean(in_interval)
            sharpness = np.mean(upper - lower)

            # Direction accuracy (hit rate)
            if len(actuals) > 1:
                actual_direction = np.sign(np.diff(actuals))
                forecast_direction = np.sign(forecasts[1:] - actuals[:-1])
                hit_rate = np.mean(actual_direction == forecast_direction)
            else:
                hit_rate = 0.5

            # Oracle-specific metrics
            if f'{name}_lambda' in self.results_df.columns:
                avg_lambda = self.results_df[f'{name}_lambda'].mean()
                avg_coherence = self.results_df[f'{name}_coherence'].mean()
            else:
                avg_lambda = 0.0
                avg_coherence = 0.0

            metrics = BacktestMetrics(
                model_name=name,
                mae=mae,
                rmse=rmse,
                mape=mape,
                coverage=coverage,
                sharpness=sharpness,
                hit_rate=hit_rate,
                avg_lambda=avg_lambda,
                avg_coherence=avg_coherence,
                n_forecasts=len(actuals)
            )

            metrics_list.append(metrics.to_dict())

        return pd.DataFrame(metrics_list)

    def diebold_mariano_test(self, model1: str, model2: str) -> Dict[str, float]:
        """Diebold-Mariano test for forecast comparison"""
        if self.results_df is None:
            raise ValueError("Must run backtest first")

        e1 = self.results_df[f'{model1}_error'].values
        e2 = self.results_df[f'{model2}_error'].values

        d = e1**2 - e2**2  # Loss differential (squared errors)
        d_bar = np.mean(d)

        n = len(d)

        # HAC standard error (Newey-West with auto lag selection)
        nw_lag = int(np.floor(4 * (n/100)**(2/9)))

        gamma_0 = np.var(d, ddof=1)
        gamma_sum = 0
        for j in range(1, nw_lag + 1):
            gamma_j = np.cov(d[:-j], d[j:])[0, 1]
            weight = 1 - j / (nw_lag + 1)
            gamma_sum += 2 * weight * gamma_j

        var_d = gamma_0 + gamma_sum
        se_d = np.sqrt(var_d / n)

        dm_stat = d_bar / se_d if se_d > 0 else 0.0
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))

        return {
            'DM_statistic': dm_stat,
            'p_value': p_value,
            'mean_loss_diff': d_bar,
            'model1_wins': 'Yes' if d_bar < 0 and p_value < 0.05 else ('No' if d_bar > 0 and p_value < 0.05 else 'Tie')
        }


# =============================================================================
# COMPREHENSIVE ANALYSIS WRAPPER
# =============================================================================

class LeavesOracleAnalyzer:
    """
    Complete analysis wrapper for Leaves-in-the-Wind Oracle experiments
    """

    def __init__(self, 
                 start_date: str = '1990-01-01',
                 end_date: Optional[str] = None):
        """Initialize analyzer with optional FRED connection"""
        self.start_date = start_date
        self.end_date = end_date

        if FRED_AVAILABLE:
            self.fred = FREDDataManager(start_date, end_date)
        else:
            self.fred = None

        self.series_data = {}
        self.backtest_results = {}

    def load_fred_series(self, name: str, series_id: str, 
                         transform: Optional[str] = None, yoy_periods: int = 12) -> pd.Series:
        """Load a FRED series"""
        if self.fred is None:
            raise ImportError("FRED data access not available")

        series = self.fred.fetch_series(series_id, transform, yoy_periods)
        self.series_data[name] = series
        return series

    def load_inflation_unemployment(self) -> Dict[str, pd.Series]:
        """Load standard inflation and unemployment series"""
        inflation = self.fred.fetch_inflation_rate()
        unemployment = self.fred.fetch_unemployment_rate()

        self.series_data['Inflation'] = inflation
        self.series_data['Unemployment'] = unemployment

        return {'Inflation': inflation, 'Unemployment': unemployment}

    def create_forecaster_suite(self, 
                                include_standard: bool = True,
                                include_hybrid: bool = True,
                                include_ensemble: bool = True,
                                include_benchmarks: bool = True) -> List[BaseForecaster]:
        """Create a comprehensive suite of forecasters"""
        forecasters = []

        if include_standard:
            forecasters.extend([
                LeavesInWindOracle(velocity_window=5, acceleration_window=3, 
                                  name="Leaves_5_3"),
                LeavesInWindOracle(velocity_window=12, acceleration_window=6,
                                  name="Leaves_12_6"),
            ])

        if include_hybrid:
            forecasters.extend([
                HybridLeavesOracle(velocity_window=5, acceleration_window=3,
                                  lambda_param=0.2, lambda_strategy='fixed',
                                  name="Hybrid_Fixed_0.2"),
                HybridLeavesOracle(velocity_window=5, acceleration_window=3,
                                  lambda_param=0.5, lambda_strategy='adaptive_acf',
                                  name="Hybrid_ACF"),
                HybridLeavesOracle(velocity_window=12, acceleration_window=6,
                                  lambda_param=0.3, lambda_strategy='coherence_based',
                                  name="Hybrid_Coh"),
            ])

        if include_ensemble:
            forecasters.extend([
                MultiLeavesEnsemble(configs=[(3,2), (5,3), (7,3), (12,6)],
                                   use_hybrid=False, name="Ensemble_Std"),
                MultiLeavesEnsemble(configs=[(3,2), (5,3), (7,3), (12,6)],
                                   use_hybrid=True, lambda_strategy='adaptive_acf',
                                   name="Ensemble_Hybrid"),
            ])

        if include_benchmarks:
            forecasters.extend([
                NoChangeForecaster(name="NoChange"),
                AR1Forecaster(window=12, name="AR1"),
                MeanForecast(window=12, name="Mean_12"),
            ])

        return forecasters

    def run_analysis(self, 
                     series_name: str,
                     series: Optional[pd.Series] = None,
                     window_size: int = 60,
                     horizon: int = 1,
                     forecasters: Optional[List[BaseForecaster]] = None) -> Dict:
        """
        Run complete rolling window analysis on a series

        Returns dictionary with:
        - metrics: DataFrame of evaluation metrics
        - results: Full backtest results DataFrame
        - dm_tests: Diebold-Mariano test results
        """
        if series is None:
            series = self.series_data.get(series_name)
            if series is None:
                raise ValueError(f"Series '{series_name}' not found. Load it first.")

        if forecasters is None:
            forecasters = self.create_forecaster_suite()

        # Run backtest
        backtest = RollingWindowBacktest(
            forecasters=forecasters,
            window_size=window_size,
            horizon=horizon
        )

        results_df = backtest.run(series, series.index)
        metrics_df = backtest.evaluate()

        # Run DM tests (compare oracles to benchmarks)
        dm_tests = {}
        oracle_names = [f.name for f in forecasters if 'Leaves' in f.name or 'Hybrid' in f.name or 'Ensemble' in f.name]
        benchmark_names = [f.name for f in forecasters if f.name in ['NoChange', 'AR1']]

        for oracle in oracle_names:
            for benchmark in benchmark_names:
                try:
                    dm_result = backtest.diebold_mariano_test(oracle, benchmark)
                    dm_tests[f'{oracle}_vs_{benchmark}'] = dm_result
                except:
                    pass

        # Store results
        self.backtest_results[series_name] = {
            'metrics': metrics_df,
            'results': results_df,
            'dm_tests': dm_tests
        }

        return self.backtest_results[series_name]

    def print_report(self, series_name: str):
        """Print formatted analysis report"""
        if series_name not in self.backtest_results:
            print(f"No results for '{series_name}'. Run analysis first.")
            return

        results = self.backtest_results[series_name]
        metrics = results['metrics']
        dm_tests = results['dm_tests']

        print("=" * 80)
        print(f"LEAVES-IN-THE-WIND ORACLE ANALYSIS: {series_name.upper()}")
        print("=" * 80)
        print()

        # Series info
        if series_name in self.series_data:
            series = self.series_data[series_name]
            print(f"Series length: {len(series)} observations")
            print(f"Date range: {series.index[0].strftime('%Y-%m')} to {series.index[-1].strftime('%Y-%m')}")
            print(f"Mean: {series.mean():.3f}, Std: {series.std():.3f}")
            print()

        # Metrics table
        print("PERFORMANCE METRICS (Ranked by MAE)")
        print("-" * 80)
        metrics_sorted = metrics.sort_values('MAE')

        # Format for display
        display_cols = ['Model', 'MAE', 'RMSE', 'Coverage', 'Hit_Rate', 'Avg_Lambda', 'Avg_Coherence']
        display_metrics = metrics_sorted[display_cols].copy()
        display_metrics['MAE'] = display_metrics['MAE'].map('{:.4f}'.format)
        display_metrics['RMSE'] = display_metrics['RMSE'].map('{:.4f}'.format)
        display_metrics['Coverage'] = display_metrics['Coverage'].map('{:.3f}'.format)
        display_metrics['Hit_Rate'] = display_metrics['Hit_Rate'].map('{:.3f}'.format)
        display_metrics['Avg_Lambda'] = display_metrics['Avg_Lambda'].map('{:.3f}'.format)
        display_metrics['Avg_Coherence'] = display_metrics['Avg_Coherence'].map('{:.3f}'.format)

        print(display_metrics.to_string(index=False))
        print()

        # Ranking
        print("RANKING BY MAE:")
        for rank, row in enumerate(metrics_sorted.itertuples(), 1):
            print(f"  {rank:2d}. {row.Model:<20} MAE={row.MAE:.4f}")
        print()

        # DM tests summary
        if dm_tests:
            print("DIEBOLD-MARIANO TESTS (Oracle vs Benchmarks)")
            print("-" * 80)
            for comparison, result in dm_tests.items():
                sig = "***" if result['p_value'] < 0.01 else ("**" if result['p_value'] < 0.05 else ("*" if result['p_value'] < 0.10 else ""))
                print(f"  {comparison:<35} DM={result['DM_statistic']:7.3f} p={result['p_value']:.4f} {sig}")
            print()
            print("  Significance: *** p<0.01, ** p<0.05, * p<0.10")
            print("  Negative DM → first model better; Positive DM → second model better")

        print()
        print("=" * 80)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_fred_analysis(series_id: str = 'UNRATE', 
                        transform: Optional[str] = None,
                        yoy_periods: int = 12,
                        start_date: str = '1990-01-01',
                        window_size: int = 12,
                        horizon: int = 1) -> Dict:
    """
    Quick one-liner analysis of a FRED series

    Example:
    --------
    >>> results = quick_fred_analysis('UNRATE')
    >>> results = quick_fred_analysis('CPIAUCSL', transform='yoy')
    """
    analyzer = LeavesOracleAnalyzer(start_date=start_date)
    series = analyzer.load_fred_series('series', series_id, transform, yoy_periods)
    results = analyzer.run_analysis('series', window_size=window_size, horizon=horizon)
    analyzer.print_report('series')
    return results


def run_inflation_unemployment_analysis(start_date: str = '1990-01-01',
                                        window_size: int = 36,
                                        horizon: int = 1) -> LeavesOracleAnalyzer:
    """
    Run complete analysis on inflation and unemployment

    Returns the analyzer object with all results
    """
    print("=" * 80)
    print("LEAVES-IN-THE-WIND ORACLE: FRED DATA ANALYSIS")
    print("Inflation Rate (YoY CPI) & Unemployment Rate")
    print("=" * 80)
    print()

    # Initialize
    analyzer = LeavesOracleAnalyzer(start_date=start_date)

    # Load data
    print("Loading FRED data...")
    data = analyzer.load_inflation_unemployment()
    print(f"  Inflation: {len(data['Inflation'])} observations")
    print(f"  Unemployment: {len(data['Unemployment'])} observations")
    print()

    # Run analyses
    for series_name in ['Inflation', 'Unemployment']:
        print(f"\nAnalyzing {series_name}...")
        analyzer.run_analysis(series_name, window_size=window_size, horizon=horizon)
        analyzer.print_report(series_name)

    return analyzer


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("Leaves-in-the-Wind Oracle: Complete Analysis Framework")
    print("=" * 60)
    print()
    print("Available functions:")
    print("  - run_inflation_unemployment_analysis()")
    print("  - quick_fred_analysis(series_id, transform)")
    print()
    print("Example usage:")
    print("  >>> analyzer = run_inflation_unemployment_analysis()")
    print("  >>> results = quick_fred_analysis('FEDFUNDS')")
    print()
