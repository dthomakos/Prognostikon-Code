# Code from the below post at the blog Prognostikon
#
# https://prognostikon.cce.uoa.gr/thomakos/gdp-forecasting-on-your-own/
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com
#

import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime
from statsmodels.api import OLS, add_constant
from scipy.optimize import minimize_scalar
from rich.console import Console
from rich.table import Table
from rich import box
import matplotlib.pyplot as plt

# Initialize rich console
console = Console()

# ================================ FRED SERIES CONFIGURATION ================================
FRED_SERIES = {
    'GDP': 'GDPC1',               # Real GDP
    'CUM': 'CUMFN',               # Capacity Utilization
    'INVRT': 'N382RX1Q020SBEA',   # Real Private Inventories
    'PCEPI': 'PCECTPI'            # PCE Price Index
}

# ================================ INDEX CONSTRUCTION =======================================
class CURPIndexBuilder:
    def __init__(self, growth_type='qoq', weights=None):
        self.growth_type = growth_type.lower()
        self.weights = self._validate_weights(weights) if weights else {
            'CUM': 0.4, 'INVRT': 0.4, 'PCEPI': 0.2}

    def _validate_weights(self, weights):
        total = sum(weights.values())
        if not np.isclose(total, 1.0):
            console.print(f"[yellow]Weights sum to {total:.2f}, normalizing to 1.0[/]")
            return {k: v/total for k,v in weights.items()}
        return weights

    def _safe_growth(self, series, periods):
        """Robust growth calculation with validation"""
        growth = series.pct_change(periods) #* 100 removed the scaling
        return growth.replace([np.inf, -np.inf], np.nan).dropna()

    def build_index(self, data):
        """Construct index with strict alignment"""
        components = data[list(self.weights.keys())].copy()
        normalized = components.div(components.iloc[0])
        index_level = normalized.dot(pd.Series(self.weights))
        return pd.DataFrame({
            'index_level': index_level,
            'index_growth': self._safe_growth(index_level,
                1 if self.growth_type == 'qoq' else 4)
        }).dropna()

# ================================ CORE FORECASTING =========================================
class GDPForecaster:
    def __init__(self, growth_type='qoq', tar_threshold=0, lag=1):
        self.growth_type = growth_type
        self.tar_threshold = tar_threshold
        self.lag = lag

    def prepare_data(self, gdp, index_growth, lag):
        """Create aligned dataset with threshold features"""
        y = self._safe_growth(gdp, 1 if self.growth_type == 'qoq' else 4)
        x = index_growth.shift(self.lag)

        df = pd.DataFrame({'y(t)': y, 'x(t)': x}).dropna()
        df['y(t+1)'] = df['y(t)'].shift(-1)
        df['I(t)'] = (df['x(t)'] < self.tar_threshold).astype(int)
        df['xI(t)'] = df['x(t)'] * df['I(t)']
        return df #.dropna() dropped the dropna() so as to not loose last observation

    def _safe_growth(self, series, periods):
        """Validated growth calculation"""
        growth = series.pct_change(periods) #* 100 removed the scaling
        return growth.replace([np.inf, -np.inf], np.nan).dropna()

    def train_model(self, data, model_type='arx'):
        """Robust model training with error handling"""
        try:
            if model_type == 'arx':
                X = add_constant(data[['y(t)', 'x(t)']])
                return OLS(data['y(t+1)'], X).fit()
            elif model_type == 'tarx':
                X = add_constant(data[['y(t)', 'x(t)', 'I(t)', 'xI(t)']])
                return OLS(data['y(t+1)'], X).fit()
        except Exception as e:
            console.print(f"[red]Model training failed: {str(e)}[/]")
            return None

# ================================ ENHANCED FEATURES ========================================
class EnhancedForecaster:
    def __init__(self, growth_type='qoq', lag=1, test_periods=40, do_plot=False):
        self.growth_type = growth_type
        self.lag = lag
        self.test_periods = test_periods

    def backtest(self, full_data, train_window=20):
        """Comprehensive backtesting with threshold optimization"""
        results = []
        test_window = self.test_periods
        start_idx = max(train_window, len(full_data)-test_window)

        for t in range(start_idx, len(full_data)-1):
            train_data = full_data.iloc[:(t+1)]

            # Optimize threshold using current training data
            opt_threshold = self.optimize_threshold(train_data)

            # Create new forecaster with optimized threshold
            forecaster = GDPForecaster(self.growth_type, opt_threshold)
            prepared_data = forecaster.prepare_data(
                train_data['y(t)'], train_data['x(t)'], self.lag
            )

            # Split data
            train_data = prepared_data.dropna(subset=['y(t+1)'])
            forecast_inputs = prepared_data.iloc[[-1]].drop(columns='y(t+1)')

            # Train models
            arx_model = forecaster.train_model(train_data, 'arx')
            tarx_model = forecaster.train_model(train_data, 'tarx')

            if arx_model and tarx_model:
                # Generate forecasts
                test_point = full_data.iloc[t+1]
                arx_fc = arx_model.predict(add_constant(forecast_inputs[['y(t)', 'x(t)']], has_constant='add'))
                tarx_fc = tarx_model.predict(add_constant(forecast_inputs[['y(t)', 'x(t)', 'I(t)', 'xI(t)']], has_constant='add'))

                results.append({
                    'date': full_data.index[t+1],
                    'actual': test_point['y(t+1)'],
                    'arx_fc': arx_fc.iloc[0],
                    'tarx_fc': tarx_fc.iloc[0],
                    'threshold': opt_threshold
                })

        return pd.DataFrame(results).set_index('date')

    def optimize_threshold(self, data):
        """Find optimal threshold using historical data"""
        def loss(threshold):
            # Create new forecaster with current threshold
            forecaster = GDPForecaster(self.growth_type, tar_threshold=threshold, lag=self.lag)
            prepared_data = forecaster.prepare_data(data['y(t)'], data['x(t)'], self.lag)
            train_data = prepared_data.dropna(subset=['y(t+1)'])
            model = forecaster.train_model(train_data, 'tarx')
            return model.mse_resid if model else np.inf

        res = minimize_scalar(
            loss,
            bounds=(data['x(t)'].quantile(0.1), data['x(t)'].quantile(0.9)),
            method='bounded'
        )
        return res.x if res.success else 0

    def format_output(self, forecast, opt_threshold, backtest_results, last_date, do_plot):
        """Create rich formatted output with performance metrics"""
        # Forecast Table
        forecast_table = Table(title="[bold green]Next Quarter Forecast[/]", box=box.ROUNDED)
        forecast_table.add_column("Date, Model & Threshold", style="cyan")
        forecast_table.add_column("Quarter & Forecast (%)", justify="right")
        #
        forecast_date = last_date + pd.DateOffset(months=3)
        quarter = (forecast_date.month - 1) // 3 + 1  # Calculate quarter manually
        forecast_table.add_row("Date", f"{forecast_date.strftime('%Y')}-Q{quarter}")
        forecast_table.add_row("ARX", f"{forecast['arx']*100:.2f}")
        forecast_table.add_row("TARX", f"{forecast['tarx']*100:.2f}")
        forecast_table.add_row("Threshold", f"{opt_threshold*100:.2f}")

        # Backtest Performance
        metrics_table = Table(title="[bold blue]Backtest Performance[/]", box=box.ROUNDED)
        metrics_table.add_column("Metric", style="magenta")
        metrics_table.add_column("ARX", justify="right")
        metrics_table.add_column("TARX", justify="right")

        metrics_table.add_row("MSE",
                            f"{self._calc_metric(backtest_results, 'arx_fc', 'mse'):.4f}",
                            f"{self._calc_metric(backtest_results, 'tarx_fc', 'mse'):.4f}")
        metrics_table.add_row("MAE",
                            f"{self._calc_metric(backtest_results, 'arx_fc', 'mae'):.4f}",
                            f"{self._calc_metric(backtest_results, 'tarx_fc', 'mae'):.4f}")
        metrics_table.add_row("Sign Accuracy (%)",
                            f"{self._calc_metric(backtest_results, 'arx_fc', 'sign'):.4f}",
                            f"{self._calc_metric(backtest_results, 'tarx_fc', 'sign'):.4f}")
        metrics_table.add_row("Precision (>= 0.0316)",
                            f"{self._calc_metric(backtest_results, 'arx_fc', 'ppc'):.4f}",
                            f"{self._calc_metric(backtest_results, 'tarx_fc', 'ppc'):.4f}")

        # Display output
        console.print(forecast_table)
        console.print(metrics_table)
        if do_plot:
            self._plot_thresholds(backtest_results['threshold'])

    def _calc_metric(self, data, col, metric='mse'):
        """Calculate performance metrics, including Thomakos' MSE-based decomposition"""
        err = data['actual'] - data[col]
        mse = (err ** 2).mean()
        mae = abs(err).mean()
        sign = (np.sign(data[col]) == np.sign(data['actual'])).mean()
        s_actual = ((err/data['actual']) ** 2).mean()
        s_forcst = ((err/data[col]) ** 2).mean()
        R_af = s_actual/s_forcst
        ppc = 1/np.sqrt(1000*R_af + 1)
        if metric == 'mse': return mse
        if metric == 'mae': return mae
        if metric == 'sign': return sign*100
        if metric == 'ppc': return ppc
        return 0

    def _plot_thresholds(self, thresholds):
        """Plot threshold distribution"""
        try:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.hist(thresholds, bins=20, color='blue', alpha=0.5)
            ax.set_title('Optimized Threshold Distribution')
            ax.set_xlabel('Threshold Value')
            ax.set_ylabel('Frequency')
            plt.show()
        except Exception as e:
            console.print(f"[yellow]Could not display plot: {str(e)}[/]")

# ===================================== MAIN PIPELINE =======================================
def generate_enhanced_forecast(weights=None, growth_type='qoq', lag=1, start_date='2000-01-01', test_periods=40, do_plot=False):
    """Complete forecasting pipeline"""
    # Fetch and preprocess data
    raw_data = web.DataReader(
        list(FRED_SERIES.values()),
        'fred',
        start=start_date,
        end=datetime.today()
    ).resample('QE').last().rename(
        columns={v:k for k,v in FRED_SERIES.items()}
    ).ffill().dropna()

    # Build CURP index
    index = CURPIndexBuilder(growth_type, weights).build_index(raw_data)

    # Prepare modeling data
    base_forecaster = GDPForecaster(growth_type, lag)
    full_data = base_forecaster.prepare_data(raw_data['GDP'], index['index_growth'], lag)

    # Backtesting and final forecast
    enhanced = EnhancedForecaster(growth_type, lag, test_periods)
    backtest_results = enhanced.backtest(full_data)

    # Current forecast with optimized threshold
    opt_threshold = enhanced.optimize_threshold(full_data)
    final_forecaster = GDPForecaster(growth_type, opt_threshold, lag)
    final_data = final_forecaster.prepare_data(raw_data['GDP'], index['index_growth'], lag)

    # Split data
    train_data = final_data.dropna(subset=['y(t+1)'])
    forecast_inputs = final_data.iloc[[-1]].drop(columns='y(t+1)')

    arx_model = final_forecaster.train_model(train_data, 'arx')
    tarx_model = final_forecaster.train_model(train_data, 'tarx')

    current = final_data.iloc[-1]
    forecast = {
        'arx': arx_model.predict(add_constant(forecast_inputs[['y(t)', 'x(t)']], has_constant='add')).iloc[0],
        'tarx': tarx_model.predict(add_constant(forecast_inputs[['y(t)', 'x(t)', 'I(t)', 'xI(t)']], has_constant='add')).iloc[0]
    }

    # Display results
    enhanced.format_output(forecast, opt_threshold, backtest_results, full_data.index[-1], do_plot)
    return forecast, backtest_results

# ===================================== EXECUTION =====================================
if __name__ == "__main__":
    console.print("[bold yellow]\n============ CURP US real GDP forecasting ============[/]\n")
    forecast, backtest = generate_enhanced_forecast(
        weights={'CUM': 0.5, 'INVRT': 0.2, 'PCEPI': 0.3}, # select the weights for the CURP index
        growth_type='yoy', # select type of forecast
        lag=3,# the lag can start at zero because the data are already pre-aligned, so lag 0 is essentially 1 etc.
        start_date='1995-01-01', # select the starting date
        test_periods=8, # select the number of periods for backtesting
        do_plot=False # plot the threshold distribution?
    )
    console.print("[bold yellow]\n============ CURP US real GDP forecasting ============[/]\n")
