#!/usr/bin/env python3
"""
LEAVES-IN-THE-WIND ORACLE EXPERIMENTATION SCRIPT
================================================

Ready-to-customize script for running parameter experiments.
Modify the parameters below to test different configurations.

Author: Dimitrios Thomakos
Date: February 15, 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from leaves_oracle_fred_wrapper import (
    LeavesInWindOracle,
    HybridLeavesOracle,
    MultiLeavesEnsemble,
    NoChangeForecaster,
    AR1Forecaster,
    MeanForecast,
    RollingWindowBacktest,
    FREDDataManager
)

# =============================================================================
# CONFIGURATION SECTION - MODIFY THESE PARAMETERS
# =============================================================================

# --- DATA CONFIGURATION ---
DATA_FILE = "your_data.csv"        # Path to your data file
DATE_COLUMN = None                 # Name of date column, or None if index
VALUE_COLUMN = "value"             # Name of value column
PARSE_DATES = True                 # Parse dates automatically?

# Or use FRED data (requires pandas_datareader):
USE_FRED = True                    # Set to True to use FRED
FRED_SERIES_ID = "GDPC1"          # FRED series code
FRED_START_DATE = "2000-01-01"
FRED_TRANSFORM = 'yoy'             # Transform to use for the series
FRED_YOY_PERIODS = 4              # Periods for yoy transform

# --- ORACLE CONFIGURATIONS ---
# Test different parameter combinations

# Standard oracles to test
STANDARD_ORACLES = [
    {'velocity_window': 3, 'acceleration_window': 2, 'name': 'Leaves_3_2'},
    {'velocity_window': 5, 'acceleration_window': 3, 'name': 'Leaves_5_3'},
    {'velocity_window': 7, 'acceleration_window': 3, 'name': 'Leaves_7_3'},
    {'velocity_window': 10, 'acceleration_window': 5, 'name': 'Leaves_10_5'},
]

# Hybrid oracles to test
HYBRID_ORACLES = [
    {
        'velocity_window': 5,
        'acceleration_window': 3,
        'lambda_param': 0.1,
        'lambda_strategy': 'fixed',
        'name': 'Hybrid_Lambda_0.1'
    },
    {
        'velocity_window': 5,
        'acceleration_window': 3,
        'lambda_param': 0.2,
        'lambda_strategy': 'fixed',
        'name': 'Hybrid_Lambda_0.2'
    },
    {
        'velocity_window': 5,
        'acceleration_window': 3,
        'lambda_param': 0.2,
        'lambda_strategy': 'adaptive_acf',
        'name': 'Hybrid_ACF_Adaptive'
    },
    {
        'velocity_window': 5,
        'acceleration_window': 3,
        'lambda_param': 0.2,
        'lambda_strategy': 'adaptive_coherence',
        'name': 'Hybrid_Coherence_Adaptive'
    },
]

# Ensemble configurations
ENSEMBLES = [
    {
        'configs': [(3,2), (5,3), (7,3), (10,5)],
        'use_hybrid': False,
        'name': 'Standard_Ensemble'
    },
    {
        'configs': [(3,2), (5,3), (7,3)],
        'use_hybrid': True,
        'lambda_param': 0.2,
        'lambda_strategy': 'adaptive_acf',
        'name': 'Hybrid_Ensemble'
    },
]

# Benchmark models
INCLUDE_BENCHMARKS = True          # Include NoChange, AR1, Mean?

# --- BACKTEST CONFIGURATION ---
WINDOW_SIZE = 60                   # Initial training window
HORIZON = 1                        # Forecast horizon (keep at 1)
STEP_SIZE = 1                      # Rolling step (1 = maximum forecasts)

# --- OUTPUT CONFIGURATION ---
SAVE_RESULTS = True                # Save metrics to CSV?
OUTPUT_FILE = "experiment_results.csv"
CREATE_PLOTS = True                # Create visualization plots?
PLOT_OUTPUT_DIR = "plots"          # Directory for plots

# --- EXPERIMENT TYPE ---
# Choose one:
EXPERIMENT_MODE = "standard"       # Options: "standard", "sensitivity", "lambda_sweep"

# For sensitivity experiments:
SENSITIVITY_PARAM = "velocity_window"  # Parameter to vary
SENSITIVITY_VALUES = [3, 4, 5, 6, 7, 8, 10, 12]  # Values to test

# For lambda sweep:
LAMBDA_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]

# =============================================================================
# DATA LOADING FUNCTION
# =============================================================================

def load_data():
    """Load data from file or FRED"""

    if USE_FRED:
        print(f"Fetching {FRED_SERIES_ID} from FRED...")
        from leaves_oracle_fred_wrapper import FREDDataManager
        fred = FREDDataManager(start_date=FRED_START_DATE, end_date=None)
        series = fred.fetch_series(series_id=FRED_SERIES_ID,
        transform=FRED_TRANSFORM,
        yoy_periods=FRED_YOY_PERIODS)
    else:
        print(f"Loading data from {DATA_FILE}...")
        if DATE_COLUMN is None:
            data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=PARSE_DATES)
        else:
            data = pd.read_csv(DATA_FILE, parse_dates=[DATE_COLUMN])
            data.set_index(DATE_COLUMN, inplace=True)
        series = data[VALUE_COLUMN]

    print(f"Loaded {len(series)} observations")
    print(f"Date range: {series.index[0]} to {series.index[-1]}")
    print(f"Series stats: mean={series.mean():.3f}, std={series.std():.3f}")
    print(f"              min={series.min():.3f}, max={series.max():.3f}")

    return series


# =============================================================================
# FORECASTER CREATION FUNCTIONS
# =============================================================================

def create_forecasters():
    """Create all forecaster objects based on configuration"""

    forecasters = []

    # Standard oracles
    print(f"\nCreating {len(STANDARD_ORACLES)} standard oracle(s)...")
    for config in STANDARD_ORACLES:
        forecasters.append(LeavesInWindOracle(**config))

    # Hybrid oracles
    print(f"Creating {len(HYBRID_ORACLES)} hybrid oracle(s)...")
    for config in HYBRID_ORACLES:
        forecasters.append(HybridLeavesOracle(**config))

    # Ensembles
    print(f"Creating {len(ENSEMBLES)} ensemble(s)...")
    for config in ENSEMBLES:
        forecasters.append(MultiLeavesEnsemble(**config))

    # Benchmarks
    if INCLUDE_BENCHMARKS:
        print("Creating benchmark models...")
        forecasters.extend([
            NoChangeForecaster(name="NoChange"),
            AR1Forecaster(window=36, name="AR1"),
            MeanForecast(window=4, name="Mean")
        ])

    print(f"\nTotal forecasters: {len(forecasters)}")
    return forecasters


def create_sensitivity_forecasters():
    """Create forecasters for sensitivity analysis"""

    forecasters = []

    if SENSITIVITY_PARAM == "velocity_window":
        for w in SENSITIVITY_VALUES:
            forecasters.append(
                LeavesInWindOracle(
                    velocity_window=w,
                    acceleration_window=3,
                    name=f"Leaves_w{w}"
                )
            )
    elif SENSITIVITY_PARAM == "acceleration_window":
        for wp in SENSITIVITY_VALUES:
            forecasters.append(
                LeavesInWindOracle(
                    velocity_window=5,
                    acceleration_window=wp,
                    name=f"Leaves_wp{wp}"
                )
            )
    else:
        raise ValueError(f"Unknown sensitivity parameter: {SENSITIVITY_PARAM}")

    return forecasters


def create_lambda_sweep_forecasters():
    """Create forecasters for lambda sensitivity analysis"""

    forecasters = []

    for lam in LAMBDA_VALUES:
        forecasters.append(
            HybridLeavesOracle(
                velocity_window=5,
                acceleration_window=3,
                lambda_param=lam,
                lambda_strategy='fixed',
                name=f"Hybrid_λ{lam:.2f}"
            )
        )

    return forecasters


# =============================================================================
# EXPERIMENT RUNNERS
# =============================================================================

def run_standard_experiment(series):
    """Run standard comparison of all configured models"""

    print("\n" + "="*80)
    print("RUNNING STANDARD EXPERIMENT")
    print("="*80)

    forecasters = create_forecasters()

    backtest = RollingWindowBacktest(
        forecasters=forecasters,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        step_size=STEP_SIZE
    )

    print(f"\nRunning backtest with window_size={WINDOW_SIZE}, horizon={HORIZON}...")
    results = backtest.run(series)

    print("Evaluating metrics...")
    metrics = backtest.evaluate()

    # Compute Diebold-Mariano tests
    print("Computing Diebold-Mariano tests...")
    baseline_model = "NoChange"
    all_models = metrics['Model'].tolist()
    dm_results = []

    for model in all_models:
        if model != baseline_model:
            try:
                # Get result DICTIONARY
                result = backtest.diebold_mariano_test(baseline_model, model)

                # Extract values using dictionary keys
                dm_stat = result['DM_statistic']
                p_value = result['p_value']
                d_bar = result['mean_loss_diff']
                model1_wins = result['model1_wins']

                dm_results.append({
                    'Model': model,
                    'DM_Statistic': dm_stat,
                    'P_Value': p_value,
                    'Mean_Loss_Diff': d_bar,
                    'Winner': model1_wins,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })

                print(f"  {model} vs {baseline_model}: DM={dm_stat:.3f}, p={p_value:.4f}, Winner={model1_wins}")

            except KeyError as e:
                print(f"  Warning: Missing key in DM result for {model}: {e}")
            except Exception as e:
                print(f"  Warning: DM test failed for {model}: {e}")

    if dm_results:
        dm_tests = pd.DataFrame(dm_results)
    else:
        dm_tests = None

    # Display results
    print("\n" + "="*80)
    print("RESULTS: FORECAST ACCURACY")
    print("="*80)
    print(metrics[['Model', 'MAE', 'RMSE', 'MAPE', 'Coverage', 'Sharpness', 'Hit_Rate']].to_string(index=False))

    print("\n" + "="*80)
    print("RESULTS: DIEBOLD-MARIANO TESTS (vs NoChange)")
    print("="*80)
    print(dm_tests[['DM_Statistic', 'P_Value', 'Winner', 'Significant']].to_string(index=False))

    # Find best model
    best_mae_idx = metrics['MAE'].idxmin()
    best_model = metrics.loc[best_mae_idx, 'Model']
    best_mae = metrics.loc[best_mae_idx, 'MAE']
    print(f"\n*** BEST MODEL (by MAE): {best_model} (MAE = {best_mae:.4f}) ***")

    # Save results
    if SAVE_RESULTS:
        metrics.to_csv(OUTPUT_FILE, index=False)
        print(f"\nResults saved to {OUTPUT_FILE}")

    return metrics, dm_tests, backtest


def run_sensitivity_experiment(series):
    """Run parameter sensitivity analysis"""

    print("\n" + "="*80)
    print(f"RUNNING SENSITIVITY ANALYSIS: {SENSITIVITY_PARAM}")
    print("="*80)

    forecasters = create_sensitivity_forecasters()

    backtest = RollingWindowBacktest(
        forecasters=forecasters,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        step_size=STEP_SIZE
    )

    print(f"\nTesting {len(SENSITIVITY_VALUES)} values...")
    results = backtest.run(series)
    metrics = backtest.evaluate()

    # Display results
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS RESULTS")
    print("="*80)
    print(metrics[['Model', 'MAE', 'RMSE', 'Coverage']].to_string(index=False))

    # Find optimal
    optimal_idx = metrics['MAE'].idxmin()
    optimal_value = SENSITIVITY_VALUES[optimal_idx]
    optimal_mae = metrics.loc[optimal_idx, 'MAE']
    print(f"\n*** OPTIMAL {SENSITIVITY_PARAM}: {optimal_value} (MAE = {optimal_mae:.4f}) ***")

    # Plot sensitivity curve
    if CREATE_PLOTS:
        import os
        os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(SENSITIVITY_VALUES, metrics['MAE'].values, 'o-', linewidth=2, markersize=8)
        plt.xlabel(SENSITIVITY_PARAM, fontsize=12)
        plt.ylabel('MAE', fontsize=12)
        plt.title(f'Sensitivity Analysis: MAE vs {SENSITIVITY_PARAM}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = f"{PLOT_OUTPUT_DIR}/sensitivity_{SENSITIVITY_PARAM}.png"
        plt.savefig(plot_file, dpi=300)
        print(f"\nPlot saved to {plot_file}")
        plt.close()

    return metrics


def run_lambda_sweep(series):
    """Run lambda parameter sweep"""

    print("\n" + "="*80)
    print("RUNNING LAMBDA SWEEP EXPERIMENT")
    print("="*80)

    forecasters = create_lambda_sweep_forecasters()

    backtest = RollingWindowBacktest(
        forecasters=forecasters,
        window_size=WINDOW_SIZE,
        horizon=HORIZON,
        step_size=STEP_SIZE
    )

    print(f"\nTesting {len(LAMBDA_VALUES)} lambda values...")
    results = backtest.run(series)
    metrics = backtest.evaluate()

    # Display results
    print("\n" + "="*80)
    print("LAMBDA SWEEP RESULTS")
    print("="*80)
    print(metrics[['Model', 'MAE', 'RMSE', 'Coverage', 'Sharpness']].to_string(index=False))

    # Find optimal
    optimal_idx = metrics['MAE'].idxmin()
    optimal_lambda = LAMBDA_VALUES[optimal_idx]
    optimal_mae = metrics.loc[optimal_idx, 'MAE']
    print(f"\n*** OPTIMAL LAMBDA: {optimal_lambda} (MAE = {optimal_mae:.4f}) ***")

    # Plot lambda curve
    if CREATE_PLOTS:
        import os
        os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # MAE vs lambda
        ax1.plot(LAMBDA_VALUES, metrics['MAE'].values, 'o-', linewidth=2, markersize=8, color='blue')
        ax1.axvline(optimal_lambda, color='red', linestyle='--', alpha=0.5, label=f'Optimal λ={optimal_lambda}')
        ax1.set_xlabel('Lambda (λ)', fontsize=12)
        ax1.set_ylabel('MAE', fontsize=12)
        ax1.set_title('MAE vs Lambda', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Coverage vs lambda
        ax2.plot(LAMBDA_VALUES, metrics['Coverage'].values, 'o-', linewidth=2, markersize=8, color='green')
        ax2.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Target 95%')
        ax2.set_xlabel('Lambda (λ)', fontsize=12)
        ax2.set_ylabel('Coverage', fontsize=12)
        ax2.set_title('Coverage vs Lambda', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        plot_file = f"{PLOT_OUTPUT_DIR}/lambda_sweep.png"
        plt.savefig(plot_file, dpi=300)
        print(f"\nPlot saved to {plot_file}")
        plt.close()

    return metrics


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""

    print("="*80)
    print("LEAVES ORACLE EXPERIMENTATION SCRIPT")
    print("="*80)

    # Load data
    series = load_data()

    # Run experiment based on mode
    if EXPERIMENT_MODE == "standard":
        metrics, dm_tests, backtest = run_standard_experiment(series)
    elif EXPERIMENT_MODE == "sensitivity":
        metrics = run_sensitivity_experiment(series)
    elif EXPERIMENT_MODE == "lambda_sweep":
        metrics = run_lambda_sweep(series)
    else:
        raise ValueError(f"Unknown experiment mode: {EXPERIMENT_MODE}")

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
