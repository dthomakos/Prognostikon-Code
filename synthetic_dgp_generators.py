"""
SYNTHETIC DATA GENERATORS FOR LEAVES-IN-THE-WIND ORACLE
========================================================

This module provides data generating processes (DGPs) for simulation studies
as described in Section 7 (Experimental Design) of the paper.

Implements three canonical DGPs:
1. Stationary AR(1)
2. Trend plus cycle
3. Regime switching

Author: Dimitrios Thomakos
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import matplotlib.pyplot as plt


# =============================================================================
# DGP CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class AR1Config:
    """Configuration for stationary AR(1) process"""
    phi: float = 0.7  # AR coefficient
    sigma: float = 1.0  # Innovation std
    length: int = 200
    seed: Optional[int] = None
    burn_in: int = 100


@dataclass
class TrendCycleConfig:
    """Configuration for trend plus cycle process"""
    beta: float = 0.05  # Trend slope
    amplitude: float = 2.0  # Cycle amplitude
    period: float = 40.0  # Cycle period
    sigma: float = 0.5  # Noise std
    length: int = 200
    seed: Optional[int] = None


@dataclass
class RegimeSwitchConfig:
    """Configuration for regime switching process"""
    mu1: float = 0.0  # Mean in regime 1
    mu2: float = 3.0  # Mean in regime 2
    sigma1: float = 0.5  # Std in regime 1
    sigma2: float = 0.5  # Std in regime 2
    phi: float = 0.3  # Autoregressive component
    regime_duration: int = 50  # Duration of each regime
    length: int = 200
    seed: Optional[int] = None


# =============================================================================
# DATA GENERATING PROCESSES
# =============================================================================

class SyntheticDGP:
    """Base class for synthetic data generation"""

    def __init__(self, name: str):
        self.name = name
        self.y = None
        self.dates = None

    def generate(self) -> pd.Series:
        """Generate synthetic series"""
        raise NotImplementedError

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame with metadata"""
        if self.y is None:
            raise ValueError("Must call generate() first")
        return pd.DataFrame({'y': self.y}, index=self.dates)


class AR1Process(SyntheticDGP):
    """
    Stationary AR(1) process:
    Y_t = φ·Y_{t-1} + ε_t,  ε_t ~ N(0, σ²)

    As described in Section 7.1 of the paper.
    """

    def __init__(self, config: AR1Config):
        super().__init__(f"AR1(φ={config.phi})")
        self.config = config

    def generate(self) -> pd.Series:
        """Generate AR(1) series"""
        np.random.seed(self.config.seed)

        T = self.config.length + self.config.burn_in
        phi = self.config.phi
        sigma = self.config.sigma

        # Check stationarity
        if np.abs(phi) >= 1.0:
            raise ValueError(f"phi={phi} violates stationarity (|phi| < 1)")

        # Generate innovations
        eps = np.random.normal(0, sigma, T)

        # Generate AR(1) series
        y = np.zeros(T)
        y[0] = eps[0] / np.sqrt(1 - phi**2)  # Stationary initial condition

        for t in range(1, T):
            y[t] = phi * y[t-1] + eps[t]

        # Remove burn-in
        y = y[self.config.burn_in:]

        # Create date index
        self.dates = pd.date_range(start='2000-01-01', periods=len(y), freq='M')
        self.y = y

        return pd.Series(y, index=self.dates, name=self.name)


class TrendCycleProcess(SyntheticDGP):
    """
    Trend plus cycle process:
    Y_t = β·t + A·sin(2π·t/P) + ε_t

    Where:
    - β is the trend slope
    - A is the cycle amplitude
    - P is the cycle period
    - ε_t ~ N(0, σ²) is noise

    As described in Section 7.1 of the paper.
    """

    def __init__(self, config: TrendCycleConfig):
        super().__init__(f"TrendCycle(P={config.period})")
        self.config = config

    def generate(self) -> pd.Series:
        """Generate trend plus cycle series"""
        np.random.seed(self.config.seed)

        T = self.config.length
        t = np.arange(T)

        # Components
        trend = self.config.beta * t
        cycle = self.config.amplitude * np.sin(2 * np.pi * t / self.config.period)
        noise = np.random.normal(0, self.config.sigma, T)

        y = trend + cycle + noise

        # Create date index
        self.dates = pd.date_range(start='2000-01-01', periods=T, freq='M')
        self.y = y

        return pd.Series(y, index=self.dates, name=self.name)


class RegimeSwitchProcess(SyntheticDGP):
    """
    Regime switching process:
    Y_t = μ_{S_t} + η_t + φ·Y_{t-1}

    Where:
    - S_t ∈ {1,2} is regime indicator
    - μ_{S_t} is regime-specific mean
    - η_t ~ N(0, σ²_{S_t}) is regime-specific noise
    - φ is autoregressive coefficient

    Regimes switch every fixed duration.

    As described in Section 7.1 of the paper.
    """

    def __init__(self, config: RegimeSwitchConfig):
        super().__init__(f"RegimeSwitch(φ={config.phi})")
        self.config = config
        self.regime_sequence = None

    def generate(self) -> pd.Series:
        """Generate regime switching series"""
        np.random.seed(self.config.seed)

        T = self.config.length
        duration = self.config.regime_duration

        # Generate regime sequence
        n_regimes = int(np.ceil(T / duration))
        regime_seq = []
        for i in range(n_regimes):
            regime = 1 if i % 2 == 0 else 2
            regime_seq.extend([regime] * duration)
        regime_seq = regime_seq[:T]
        self.regime_sequence = np.array(regime_seq)

        # Generate series
        y = np.zeros(T)
        y[0] = self.config.mu1  # Start in regime 1

        for t in range(1, T):
            regime = regime_seq[t]

            if regime == 1:
                mu = self.config.mu1
                sigma = self.config.sigma1
            else:
                mu = self.config.mu2
                sigma = self.config.sigma2

            eta = np.random.normal(0, sigma)
            y[t] = mu + eta + self.config.phi * y[t-1]

        # Create date index
        self.dates = pd.date_range(start='2000-01-01', periods=T, freq='M')
        self.y = y

        return pd.Series(y, index=self.dates, name=self.name)


# =============================================================================
# SIMULATION STUDY RUNNER
# =============================================================================

class SimulationStudy:
    """
    Orchestrates complete simulation study as described in Section 7

    Runs multiple replications of each DGP, evaluates all forecasters,
    and aggregates results.
    """

    def __init__(self, 
                 forecasters: list,
                 window_size: int = 50,
                 horizon: int = 1,
                 n_replications: int = 100):
        """
        Parameters
        ----------
        forecasters : list
            List of forecaster objects
        window_size : int
            Initial training window size (T_0 in paper)
        horizon : int
            Forecast horizon
        n_replications : int
            Number of Monte Carlo replications
        """
        self.forecasters = forecasters
        self.window_size = window_size
        self.horizon = horizon
        self.n_replications = n_replications
        self.results = {}

    def run_dgp(self, dgp_generator, dgp_name: str) -> Dict:
        """
        Run simulation study for a single DGP

        Parameters
        ----------
        dgp_generator : callable
            Function that generates a new DGP instance
        dgp_name : str
            Name of the DGP for reporting

        Returns
        -------
        results : dict
            Aggregated results across replications
        """
        print(f"\nRunning simulation for {dgp_name}...")
        print(f"  Replications: {self.n_replications}")
        print(f"  Window size: {self.window_size}")
        print(f"  Horizon: {self.horizon}")

        # Storage for replication results
        replication_results = {f.name: [] for f in self.forecasters}

        for rep in range(self.n_replications):
            if (rep + 1) % 10 == 0:
                print(f"  Replication {rep + 1}/{self.n_replications}...", end='\r')

            # Generate data
            dgp = dgp_generator()
            series = dgp.generate()

            # Run backtest for each forecaster
            from leaves_oracle_fred_wrapper import RollingWindowBacktest

            backtest = RollingWindowBacktest(
                forecasters=self.forecasters,
                window_size=self.window_size,
                horizon=self.horizon,
                step_size=1
            )

            results_df = backtest.run(series)
            metrics_df = backtest.evaluate()

            # Store metrics for this replication
            for _, row in metrics_df.iterrows():
                model_name = row['Model']
                replication_results[model_name].append({
                    'MAE': row['MAE'],
                    'RMSE': row['RMSE'],
                    'Coverage': row['Coverage'],
                    'Sharpness': row['Sharpness'],
                    'Hit_Rate': row['Hit_Rate']
                })

        print(f"  Completed {self.n_replications} replications!        ")

        # Aggregate across replications
        aggregated = {}
        for model_name, rep_list in replication_results.items():
            metrics_array = {
                metric: [r[metric] for r in rep_list]
                for metric in ['MAE', 'RMSE', 'Coverage', 'Sharpness', 'Hit_Rate']
            }

            aggregated[model_name] = {
                metric: {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'q05': np.percentile(values, 5),
                    'q95': np.percentile(values, 95)
                }
                for metric, values in metrics_array.items()
            }

        return aggregated

    def run_all_dgps(self, dgp_configs: Dict) -> Dict:
        """
        Run simulation study for all DGPs

        Parameters
        ----------
        dgp_configs : dict
            Dictionary mapping DGP names to generator functions

        Returns
        -------
        all_results : dict
            Nested dictionary with results for each DGP
        """
        all_results = {}

        for dgp_name, dgp_generator in dgp_configs.items():
            all_results[dgp_name] = self.run_dgp(dgp_generator, dgp_name)

        self.results = all_results
        return all_results

    def print_summary(self):
        """Print summary tables of simulation results"""
        if not self.results:
            print("No results to display. Run simulation first.")
            return

        print("\n" + "=" * 80)
        print("SIMULATION STUDY RESULTS SUMMARY")
        print("=" * 80)

        for dgp_name, dgp_results in self.results.items():
            print(f"\n{dgp_name.upper()}")
            print("-" * 80)
            print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'Coverage':<12} {'Hit Rate':<12}")
            print("-" * 80)

            # Sort by MAE
            sorted_models = sorted(
                dgp_results.items(),
                key=lambda x: x[1]['MAE']['mean']
            )

            for model_name, metrics in sorted_models:
                mae_mean = metrics['MAE']['mean']
                rmse_mean = metrics['RMSE']['mean']
                cov_mean = metrics['Coverage']['mean']
                hit_mean = metrics['Hit_Rate']['mean']

                print(f"{model_name:<20} {mae_mean:>6.4f}±{metrics['MAE']['std']:<4.3f} "
                      f"{rmse_mean:>6.4f}±{metrics['RMSE']['std']:<4.3f} "
                      f"{cov_mean:>6.3f}±{metrics['Coverage']['std']:<4.3f} "
                      f"{hit_mean:>6.3f}±{metrics['Hit_Rate']['std']:<4.3f}")

        print("\n" + "=" * 80)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_standard_dgps(length: int = 200) -> Dict[str, SyntheticDGP]:
    """
    Generate all three standard DGPs as described in Section 7.1

    Returns dictionary of DGP instances
    """
    dgps = {
        'AR1': AR1Process(AR1Config(phi=0.7, sigma=1.0, length=length, seed=42)),
        'TrendCycle': TrendCycleProcess(TrendCycleConfig(
            beta=0.05, amplitude=2.0, period=40.0, sigma=0.5, length=length, seed=42
        )),
        'RegimeSwitch': RegimeSwitchProcess(RegimeSwitchConfig(
            mu1=0.0, mu2=3.0, sigma1=0.5, sigma2=0.5, 
            phi=0.3, regime_duration=50, length=length, seed=42
        ))
    }

    return dgps


def run_quick_simulation(forecasters: list, 
                        n_replications: int = 50,
                        window_size: int = 50) -> Dict:
    """
    Quick simulation study with standard DGPs

    Parameters
    ----------
    forecasters : list
        List of forecaster objects to evaluate
    n_replications : int
        Number of Monte Carlo replications
    window_size : int
        Initial training window

    Returns
    -------
    results : dict
        Simulation results for all DGPs
    """
    # Define DGP generators
    dgp_configs = {
        'AR1': lambda: AR1Process(AR1Config(phi=0.7, sigma=1.0, length=200)),
        'TrendCycle': lambda: TrendCycleProcess(TrendCycleConfig(
            beta=0.05, amplitude=2.0, period=40.0, sigma=0.5, length=200
        )),
        'RegimeSwitch': lambda: RegimeSwitchProcess(RegimeSwitchConfig(
            mu1=0.0, mu2=3.0, sigma1=0.5, sigma2=0.5,
            phi=0.3, regime_duration=50, length=200
        ))
    }

    # Run simulation study
    study = SimulationStudy(
        forecasters=forecasters,
        window_size=window_size,
        horizon=1,
        n_replications=n_replications
    )

    results = study.run_all_dgps(dgp_configs)
    study.print_summary()

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_dgp_realizations(n_realizations: int = 5, length: int = 200):
    """
    Plot sample realizations of all three DGPs

    Creates a 3x1 panel plot showing typical realizations
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # AR(1)
    ax = axes[0]
    for i in range(n_realizations):
        dgp = AR1Process(AR1Config(phi=0.7, sigma=1.0, length=length, seed=i))
        series = dgp.generate()
        ax.plot(series.values, alpha=0.6, linewidth=1)
    ax.set_title('AR(1) Process: $Y_t = 0.7 Y_{t-1} + \\varepsilon_t$', fontsize=12)
    ax.set_ylabel('$Y_t$')
    ax.grid(True, alpha=0.3)

    # Trend + Cycle
    ax = axes[1]
    for i in range(n_realizations):
        dgp = TrendCycleProcess(TrendCycleConfig(
            beta=0.05, amplitude=2.0, period=40.0, sigma=0.5, length=length, seed=i
        ))
        series = dgp.generate()
        ax.plot(series.values, alpha=0.6, linewidth=1)
    ax.set_title('Trend + Cycle: $Y_t = 0.05t + 2\\sin(2\\pi t/40) + \\varepsilon_t$', fontsize=12)
    ax.set_ylabel('$Y_t$')
    ax.grid(True, alpha=0.3)

    # Regime Switching
    ax = axes[2]
    for i in range(n_realizations):
        dgp = RegimeSwitchProcess(RegimeSwitchConfig(
            mu1=0.0, mu2=3.0, sigma1=0.5, sigma2=0.5,
            phi=0.3, regime_duration=50, length=length, seed=i
        ))
        series = dgp.generate()
        ax.plot(series.values, alpha=0.6, linewidth=1)
    ax.set_title('Regime Switching: $Y_t = \\mu_{S_t} + \\eta_t + 0.3 Y_{t-1}$', fontsize=12)
    ax.set_ylabel('$Y_t$')
    ax.set_xlabel('Time')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Synthetic DGP Module for Leaves-in-the-Wind Oracle")
    print("=" * 60)
    print()
    print("Available DGPs:")
    print("  1. AR1Process - Stationary AR(1)")
    print("  2. TrendCycleProcess - Deterministic trend + cycle")
    print("  3. RegimeSwitchProcess - Mean-switching with AR component")
    print()
    print("Usage:")
    print("  >>> dgps = generate_standard_dgps(length=200)")
    print("  >>> ar1_series = dgps['AR1'].generate()")
    print()
    print("  >>> results = run_quick_simulation(forecasters, n_replications=100)")
