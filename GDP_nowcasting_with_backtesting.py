#
# Python code adding for the post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-fractional-trader-supercharged/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

# ============================ IMPORTS =====================
import pandas as pd
import numpy as np
import pandas_datareader.data as web
from datetime import datetime, timedelta
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from statsmodels.tsa.ar_model import AutoReg
from rich.console import Console
from rich.table import Table
from rich.box import Box
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
console = Console()

# ================== GLOBAL CONFIGURATION ==================
DEBUG_MODE = True
START_DATE = '2000-01-01'
QUARTERLY_TARGET = 'GDPC1'
WEEKLY_SERIES = {
    'FF': 'DFF',
    'WGS10YR': 'DGS10',
    'DPSACBW027SBOG': 'DPSACBW027SBOG',
    'TOTCI': 'TOTCI',
    'CLSACBW027SBOG': 'CLSACBW027SBOG'
}
RAW_DATA = {}

# =============== ENHANCED MIDAS TRANSFORMER ===============
class MidasTransformer:
    def __init__(self, start_date=START_DATE, n_weeks=12, growth_type='qoq'):
        self.start_date = start_date
        self.n_weeks = n_weeks # Must be greater or equal to
        self.growth_type = growth_type.lower()
        self.scaler = StandardScaler()
        self.gdp_dates = None
        self.feature_names = []
        self.feature_registry = []
        self.expected_features = None  # Will be set after first transform

    def _safe_fetch(self, series_id):
        """Robust data fetching with validation"""
        try:
            if series_id == QUARTERLY_TARGET:
                series = web.DataReader(series_id, 'fred', start=self.start_date)
                return series.resample('QE').last().ffill()

            # Handle different series and frequencies
            if series_id in ['DFF', 'DGS10']:
                series = web.DataReader(series_id, 'fred', start=self.start_date)
                return series.squeeze().resample('W-WED').last().ffill()
            else:
                series = web.DataReader(series_id, 'fred', start=self.start_date)
                return series.resample('W-WED').last().ffill()

        except Exception as e:
            console.print(f"[red]Error fetching {series_id}: {str(e)}[/]")
            return pd.Series(name=series_id)

    def _calculate_growth(self, series, series_id):
        """Smart growth rate calculation with outlier handling"""
        if series.empty or len(series) < 5:
            return pd.Series()

        # Compute the growth rates, weekly variables have weekly growth
        try:
            if series_id == QUARTERLY_TARGET and self.growth_type == 'qoq':
                return series.pct_change(1, fill_method=None).dropna()
            elif series_id == QUARTERLY_TARGET and self.growth_type == 'yoy':
                return series.pct_change(4, fill_method=None).dropna()
            elif series_id != QUARTERLY_TARGET:
                if series_id in ['DFF', 'DGS10']:
                    return series.diff(1).dropna()
                else:
                    return series.pct_change(1, fill_method=None).dropna()

        except Exception as e:
            console.print(f"[red]Growth calc error {series_id}: {str(e)}[/]")
            return pd.Series()

    def _create_features(self, series, name):
        """Improved MIDAS feature engineering with proper alignment"""
        features = pd.DataFrame(index=self.gdp_dates)

        for q_end in self.gdp_dates:
            start_date = q_end - timedelta(weeks=self.n_weeks)
            window = series.loc[start_date:q_end]

            if len(window) >= 10:
                values = window.tail(self.n_weeks).values
                pad_left = max(self.n_weeks - len(values), 0)
                values = np.pad(values, (pad_left, 0),
                              mode='constant', constant_values=np.nan)

                for i in range(self.n_weeks):
                    features.loc[q_end, f'{name}_w{i+1}'] = values[-(i+1)]

        return features.ffill().dropna(thresh=int(self.n_weeks*0.8))

    def transform(self):
        """Enhanced transformation pipeline"""
        gdp = self._safe_fetch(QUARTERLY_TARGET)
        if gdp.empty:
            raise ValueError("Failed to fetch GDP data")

        self.gdp_dates = gdp.index
        gdp_growth = self._calculate_growth(gdp, QUARTERLY_TARGET)
        RAW_DATA[QUARTERLY_TARGET] = gdp

        panels = []
        console.print("[bold green]Processing weekly series:[/]")

        for name, series_id in WEEKLY_SERIES.items():
            console.print(f"  - {name} ({series_id})", end=" ")
            raw = self._safe_fetch(series_id)
            RAW_DATA[series_id] = raw

            if not raw.empty:
                transformed = self._calculate_growth(raw, series_id)
                if not transformed.empty:
                    features = self._create_features(transformed, name)
                    if not features.empty:
                        panels.append(features)
                        console.print("[green]✓[/]")
                        continue
            console.print("[red]×[/]")

        full_panel = pd.concat(panels, axis=1)
        full_panel.columns = full_panel.columns.astype(str)
        self.feature_names = full_panel.columns.tolist()

        # After creating full_panel
        self.expected_features = full_panel.columns.tolist()
        self.feature_registry = self.expected_features.copy()

        # Dynamic threshold for missing data
        valid_panel = full_panel.dropna(thresh=int(len(WEEKLY_SERIES)*self.n_weeks*0.8))

        return (
            self.scaler.fit_transform(valid_panel),
            valid_panel.index,
            gdp_growth.reindex(valid_panel.index),
            self.feature_names
        )

# ================== ROBUST NOWCASTING SYSTEM ==================
class RealTimeNowcaster:
    def __init__(self, n_lags=1, train_size=40):
        # The defaults for the Elastic Net are below and hardcoded
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', ElasticNetCV(l1_ratio=0.5, cv=5, random_state=0, max_iter=5000, fit_intercept=True, tol=1e-4, selection='cyclic'))
        ])
        self.n_lags = n_lags
        self.train_size = train_size
        self.feature_names = None
        self.last_trained = None

    def prepare_features(self, X, dates, gdp_growth):
        """Leakage-proof feature engineering"""
        features = pd.DataFrame(X, index=dates, columns=self.feature_names)

        # Add lagged GDP growth features
        for lag in range(1, self.n_lags+1):
            features[f'GDP_lag{lag}'] = gdp_growth.shift(lag).reindex(features.index)

        return features.dropna()

    def backtest(self, X, dates, gdp_growth):
        """Corrected backtesting with Series validation"""
        try:
            if not self.feature_names:
                raise ValueError("Feature names not set")

            features = self.prepare_features(X, dates, gdp_growth)

            # Ensure GDP growth is a Series
            y = gdp_growth.reindex(features.index).squeeze()  # Add .squeeze() here

            if len(features) < self.train_size + 1:
                console.print(f"[red]Insufficient data: {len(features)} quarters[/]")
                return pd.DataFrame()

            results = []
            for t in range(self.train_size, len(features)):
                X_train = features.iloc[:t]
                y_train = y.iloc[:t]
                X_test = features.iloc[t:t+1]

                # Validate test point using scalar checks
                if X_test.isna().any().any() or y_train.isna().any():
                    continue

                # Train model
                self.model.fit(X_train, y_train)
                nowcast = self.model.predict(X_test)[0]

                # AR(1) benchmark
                ar_model = AutoReg(y_train, lags=1, trend='c').fit()
                ar_fc = ar_model.forecast(steps=1).iloc[0]

                # Proper quarter formatting
                forecast_date = y.index[t]
                quarter = (forecast_date.month - 1) // 3 + 1
                date_str = f"{forecast_date.year}-Q{quarter}"

                results.append({
                    'date': date_str,
                    'actual': y.iloc[t],
                    'nowcast': nowcast,
                    'ar_fc': ar_fc
                })

            return pd.DataFrame(results).set_index('date')

        except Exception as e:
            console.print(f"[red]Backtest failed: {str(e)}[/]")
            return pd.DataFrame()

    def performance_report(self, results):
        """Enhanced diagnostics with statistical testing"""
        if results.empty:
            console.print("[red]No results to analyze[/]")
            return

        # Calculate metrics
        metrics = {
            'RMSE': lambda a, p: np.sqrt(((a - p)**2).mean()),
            'MAE': lambda a, p: (a - p).abs().mean(),
            'Direction Accuracy (%)': lambda a, p: 100 * (np.sign(a) == np.sign(p)).mean(),
            'Correlation': lambda a, p: a.corr(p)
        }

        # Create table
        table = Table(title="Model Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Nowcast", justify="right")
        table.add_column("AR(1)", justify="right")
        table.add_column("Improvement", justify="right")

        for name, func in metrics.items():
            now_val = func(results['actual'], results['nowcast'])
            ar_val = func(results['actual'], results['ar_fc'])
            improvement = (now_val - ar_val) if name != 'Direction Accuracy (%)' else (now_val - ar_val)
            table.add_row(
                name,
                f"{now_val:.4f}",
                f"{ar_val:.4f}",
                f"[green]{improvement:.4f}[/]" if improvement > 0 else f"[red]{improvement:.4f}[/]"
            )

        console.print(table)

# ========================== MAIN EXECUTION ==========================
if __name__ == "__main__":
    console.print("[bold]GDP Nowcasting System[/]", style="blue")
    console.print("=====================\n", style="blue")

    # Initialize data pipeline
    #
    # Change the start date in the globals section
    # Leave n_weeks as one of n_weeks = (9, 10, 11 or 12)
    set_n_weeks = 12
    # Set growht_type = 'qoq' or 'yoy'
    set_growth_type = 'qoq'
    d_growth = 1 if set_growth_type == 'qoq' else 4
    label_growth = 'Q-on-Q' if set_growth_type == 'qoq' else 'Y-on-Y'
    # Call the transformer
    mt = MidasTransformer(start_date=START_DATE, n_weeks=set_n_weeks, growth_type=set_growth_type)
    X_midas, dates, gdp_growth, feature_names = mt.transform()

    if not gdp_growth.empty:
        # Initialize nowcasting engine
        #
        # Select lags for model, benchmark is AR(1) always
        set_n_lags = 2
        # Select initial training size for the recursive backtesting
        set_init_train = 40
        # Call the nowcaster
        nowcaster = RealTimeNowcaster(n_lags=set_n_lags, train_size=set_init_train)
        nowcaster.feature_names = feature_names

        # Run backtest
        console.print("\n[bold green]Running Backtest...[/]")
        results = nowcaster.backtest(X_midas, dates, gdp_growth)

        if not results.empty:
            # Show performance
            nowcaster.performance_report(results)

            # Real-time nowcast, given the data already available and the lags
            new_gdp_growth_lags = RAW_DATA['GDPC1'].pct_change(d_growth).iloc[-set_n_lags:][::-1].values
            final = nowcaster.prepare_features(X_midas, dates, gdp_growth)
            final.iloc[-1,-set_n_lags:] = new_gdp_growth_lags.flatten()
            nowcast = nowcaster.model.predict(final.iloc[-1:])[0]
            console.print(f"\n[bold]Current Quarter Nowcast, "+label_growth+f" =[/] {nowcast*100:.4f}%")

