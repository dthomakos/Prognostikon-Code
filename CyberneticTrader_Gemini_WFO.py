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
import matplotlib.pyplot as plt
from nolds import sampen # For Sample Entropy
import yfinance as yf
import itertools # For parameter grid search
import warnings # Import the warnings module

# --- Silence specific RuntimeWarning from nolds ---
# This line will filter out RuntimeWarnings originating from the 'nolds.measures' module.
# Use this if you understand the warning and have handled the underlying numerical stability
# issues (as we have in _calculate_entropy by checking for near-zero std dev).
warnings.filterwarnings("ignore", category=RuntimeWarning, module='nolds.measures')

# -------------------------------------------------------------------------------------
# Class definition(s)
# -------------------------------------------------------------------------------------

# --- CyberneticTrader Class (Copied for self-contained WFO example) ---
class CyberneticTrader:
    def __init__(self, history_length=20, learning_rate=0.001,
                 order_wiener_expansion=2, regularization_strength=0.001,
                 normalize_returns=True):
        if not isinstance(history_length, int) or history_length <= 0:
            raise ValueError("history_length must be a positive integer.")
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive number.")
        if not isinstance(order_wiener_expansion, int) or order_wiener_expansion < 0:
            raise ValueError("order_wiener_expansion must be a non-negative integer.")
        if not isinstance(regularization_strength, (int, float)) or regularization_strength < 0:
            raise ValueError("regularization_strength must be a non-negative number.")
        if not isinstance(normalize_returns, bool):
            raise ValueError("normalize_returns must be a boolean.")

        self.history_length = history_length
        self.learning_rate = learning_rate
        self.order_wiener_expansion = order_wiener_expansion
        self.regularization_strength = regularization_strength
        self.normalize_returns = normalize_returns

        self.kernels = self._initialize_kernels(order_wiener_expansion)
        self.returns_history = np.zeros(self.history_length)
        self.last_predicted_return = 0.0
        self.is_initialized_for_forecast = False
        self.history_counter = 0

        self.rolling_mean = 0.0
        self.rolling_std = 1.0

    def _initialize_kernels(self, order):
        kernels = {}
        kernels[0] = 0.0
        if order >= 1:
            kernels[1] = np.zeros(self.history_length)
        if order >= 2:
            kernels[2] = np.zeros((self.history_length, self.history_length))
        if order > 2:
            print(f"Warning: Wiener expansion order {order} is set, but only orders up to 2 are explicitly implemented.")
        return kernels

    def _calculate_entropy(self, data):
        if len(data) < 2:
            return 0.0
        if np.isclose(np.std(data), 0):
            return 0.0
        try:
            tolerance_val = max(0.2 * np.std(data), 1e-9)
            return sampen(data, emb_dim=2, tolerance=tolerance_val)
        except Exception:
            return 0.0

    def _wiener_forecast(self, current_history_normalized):
        forecast = self.kernels[0]
        if 1 in self.kernels:
            forecast += np.dot(self.kernels[1], current_history_normalized)
        if 2 in self.kernels:
            forecast += np.sum(self.kernels[2] * np.outer(current_history_normalized, current_history_normalized))
        return forecast

    def _adaptive_kernel_adjustment(self, actual_return_normalized, predicted_return_normalized, history_at_prediction_time_normalized):
        error = actual_return_normalized - predicted_return_normalized
        self.kernels[0] += self.learning_rate * error - self.learning_rate * self.regularization_strength * self.kernels[0]
        if 1 in self.kernels:
            self.kernels[1] += self.learning_rate * error * history_at_prediction_time_normalized \
                               - self.learning_rate * self.regularization_strength * self.kernels[1]
        if 2 in self.kernels:
            outer_product = np.outer(history_at_prediction_time_normalized, history_at_prediction_time_normalized)
            self.kernels[2] += self.learning_rate * error * outer_product \
                               - self.learning_rate * self.regularization_strength * self.kernels[2]
            self.kernels[2] = (self.kernels[2] + self.kernels[2].T) / 2

    def process_data_point(self, current_return):
        history_for_feedback = np.copy(self.returns_history)
        if self.history_counter < self.history_length:
            self.returns_history[self.history_counter] = current_return
            self.history_counter += 1
        else:
            self.returns_history = np.roll(self.returns_history, -1)
            self.returns_history[-1] = current_return
            self.is_initialized_for_forecast = True

        signal = 0
        predicted_return_for_next_period = None
        current_entropy = None

        current_history_normalized = np.copy(self.returns_history)
        current_return_normalized = current_return
        history_for_feedback_normalized = np.copy(history_for_feedback)
        last_predicted_return_normalized = self.last_predicted_return

        if self.normalize_returns and self.history_counter > 1:
            self.rolling_mean = np.mean(self.returns_history[:self.history_counter])
            self.rolling_std = np.std(self.returns_history[:self.history_counter])
            if self.rolling_std == 0:
                self.rolling_std = 1e-6

            current_history_normalized = (self.returns_history - self.rolling_mean) / self.rolling_std
            current_return_normalized = (current_return - self.rolling_mean) / self.rolling_std
            history_for_feedback_normalized = (history_for_feedback - self.rolling_mean) / self.rolling_std

        if self.is_initialized_for_forecast:
            self._adaptive_kernel_adjustment(current_return_normalized, last_predicted_return_normalized, history_for_feedback_normalized)
            predicted_return_for_next_period_normalized = self._wiener_forecast(current_history_normalized)
            predicted_return_for_next_period = predicted_return_for_next_period_normalized * self.rolling_std + self.rolling_mean
            signal = np.sign(predicted_return_for_next_period)
            if signal == 0:
                signal = 0
            self.last_predicted_return = predicted_return_for_next_period_normalized
            current_entropy = self._calculate_entropy(self.returns_history[:self.history_counter])

        return signal, predicted_return_for_next_period, current_entropy

# --- BacktestingEngine Class (Copied for self-contained WFO example) ---
class BacktestingEngine:
    def __init__(self, trader_instance, initial_capital=10000.0, transaction_cost_bps=1.0,
                 max_position_size=1.0, stop_loss_pct=0.02, take_profit_pct=0.03):
        if not isinstance(trader_instance, CyberneticTrader):
            raise TypeError("trader_instance must be an instance of CyberneticTrader.")
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            raise ValueError("initial_capital must be a positive number.")
        if not isinstance(transaction_cost_bps, (int, float)) or transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be a non-negative number.")
        if not isinstance(max_position_size, (int, float)) or not (0.0 <= max_position_size <= 1.0):
            raise ValueError("max_position_size must be between 0.0 and 1.0.")
        if not isinstance(stop_loss_pct, (int, float)) or stop_loss_pct < 0:
            raise ValueError("stop_loss_pct must be a non-negative number.")
        if not isinstance(take_profit_pct, (int, float)) or take_profit_pct < 0:
            raise ValueError("take_profit_pct must be a non-negative number.")

        self.trader = trader_instance
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transaction_cost_rate = transaction_cost_bps / 10000.0

        self.portfolio_history = [initial_capital]
        self.trade_log = []

        self.current_position = 0
        self.position_entry_price = None
        self.position_size_value = 0

        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def _execute_trade(self, date, current_price, daily_return, signal_for_next_period, predicted_return, current_entropy):
        pnl_today = 0
        transaction_cost = 0
        action_today = "HOLD"

        if self.current_position != 0 and self.position_entry_price is not None:
            if self.current_position == 1:
                if current_price <= self.position_entry_price * (1 - self.stop_loss_pct):
                    action_today = "STOP_LOSS_LONG"
                    pnl_today = self.position_size_value * (current_price / self.position_entry_price - 1)
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                elif current_price >= self.position_entry_price * (1 + self.take_profit_pct):
                    action_today = "TAKE_PROFIT_LONG"
                    pnl_today = self.position_size_value * (current_price / self.position_entry_price - 1)
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                else:
                    pnl_today = self.position_size_value * daily_return
            elif self.current_position == -1:
                if current_price >= self.position_entry_price * (1 + self.stop_loss_pct):
                    action_today = "STOP_LOSS_SHORT"
                    pnl_today = self.position_size_value * (self.position_entry_price / current_price - 1)
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                elif current_price <= self.position_entry_price * (1 - self.take_profit_pct):
                    action_today = "TAKE_PROFIT_SHORT"
                    pnl_today = self.position_size_value * (self.position_entry_price / current_price - 1)
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                else:
                    pnl_today = self.position_size_value * (-daily_return)

        self.capital += pnl_today

        if self.current_position == 0 and signal_for_next_period != 0:
            action_today = "BUY" if signal_for_next_period == 1 else "SELL"
            self.current_position = signal_for_next_period
            self.position_entry_price = current_price
            self.position_size_value = self.capital * self.max_position_size
            transaction_cost += self.position_size_value * self.transaction_cost_rate
        elif self.current_position != 0 and signal_for_next_period == 0:
            action_today = "FLAT"
            transaction_cost += self.position_size_value * self.transaction_cost_rate
            self.current_position = 0
            self.position_entry_price = None
            self.position_size_value = 0
        elif self.current_position != 0 and signal_for_next_period != 0 and signal_for_next_period != self.current_position:
            action_today = "REVERSE"
            transaction_cost += self.position_size_value * self.transaction_cost_rate
            self.current_position = signal_for_next_period
            self.position_entry_price = current_price
            self.position_size_value = self.capital * self.max_position_size
            transaction_cost += self.position_size_value * self.transaction_cost_rate

        self.capital -= transaction_cost

        self.trade_log.append({
            'date': date,
            'daily_return': daily_return,
            'signal_for_next_period': int(signal_for_next_period),
            'predicted_return_for_next_period': predicted_return,
            'current_position': int(self.current_position),
            'action_today': action_today,
            'pnl_today': pnl_today,
            'transaction_cost': transaction_cost,
            'current_capital': self.capital,
            'current_entropy': current_entropy
        })
        self.portfolio_history.append(self.capital)


    def run_backtest(self, price_data_series):
        if not isinstance(price_data_series, pd.Series):
            raise TypeError("price_data_series must be a pandas Series.")
        if price_data_series.empty:
            print("Warning: Empty price_data_series provided for backtest.")
            return

        # Reset trader state for a new backtest run
        self.trader.returns_history = np.zeros(self.trader.history_length)
        self.trader.last_predicted_return = 0.0
        self.trader.is_initialized_for_forecast = False
        self.trader.history_counter = 0
        self.trader.kernels = self.trader._initialize_kernels(self.trader.order_wiener_expansion)
        self.trader.rolling_mean = 0.0
        self.trader.rolling_std = 1.0

        # Reset backtester state
        self.capital = self.initial_capital
        self.portfolio_history = [self.initial_capital]
        self.trade_log = []
        self.current_position = 0
        self.position_entry_price = None
        self.position_size_value = 0


        # print(f"Starting backtest with ${self.initial_capital:,.2f} capital over {len(price_data_series)-1} periods.")

        for i in range(1, len(price_data_series)):
            date = price_data_series.index[i]
            current_price = price_data_series.iloc[i]
            previous_price = price_data_series.iloc[i-1]
            daily_return = (current_price - previous_price) / previous_price

            signal_for_next_period, predicted_return, current_entropy = \
                self.trader.process_data_point(daily_return)

            if self.trader.is_initialized_for_forecast:
                self._execute_trade(date, current_price, daily_return, signal_for_next_period, predicted_return, current_entropy)
            else:
                self.trade_log.append({
                    'date': date,
                    'daily_return': daily_return,
                    'signal_for_next_period': None,
                    'predicted_return_for_next_period': None,
                    'current_position': 0,
                    'action_today': 'INITIALIZING',
                    'pnl_today': 0,
                    'transaction_cost': 0,
                    'current_capital': self.capital,
                    'current_entropy': current_entropy
                })
                self.portfolio_history.append(self.capital)

        # print(f"Backtest finished. Final capital: ${self.capital:,.2f}")

    def analyze_results(self):
        results_df = pd.DataFrame(self.trade_log)

        if not results_df.empty:
            all_portfolio_dates = [results_df['date'].iloc[0] - pd.Timedelta(days=1)] + list(results_df['date'])
            portfolio_series = pd.Series(self.portfolio_history, index=pd.to_datetime(all_portfolio_dates), name="Portfolio Value")
        else:
            portfolio_series = pd.Series([self.initial_capital], name="Portfolio Value")

        total_return = (self.capital / self.initial_capital) - 1
        num_trading_days = len(results_df[results_df['action_today'] != 'INITIALIZING'])

        if num_trading_days > 0:
            annualized_return = (1 + total_return)**(252 / num_trading_days) - 1
        else:
            annualized_return = 0.0

        portfolio_daily_returns = portfolio_series.pct_change().dropna()
        annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252) if not portfolio_daily_returns.empty else 0.0

        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

        peak = portfolio_series.expanding(min_periods=1).max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        # print("\n--- Backtest Summary ---")
        # print(f"Initial Capital:        ${self.initial_capital:,.2f}")
        # print(f"Final Capital:          ${self.capital:,.2f}")
        # print(f"Total Return:           {total_return:.2%}")
        # print(f"Annualized Return:      {annualized_return:.2%}")
        # print(f"Annualized Volatility:  {annualized_volatility:.2%}")
        # print(f"Sharpe Ratio:           {sharpe_ratio:.2f}")
        # print(f"Max Drawdown:           {max_drawdown:.2%}")
        # print(f"Total Transaction Costs: ${results_df['transaction_cost'].sum():,.2f}")
        # print(f"Number of Trading Days: {num_trading_days}")

        return results_df, portfolio_series, {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_transaction_costs': results_df['transaction_cost'].sum()
        }

# --- Walk-Forward Optimization Function ---

def walk_forward_optimization(
    price_data_series: pd.Series,
    trader_param_grid: dict,
    backtester_param_grid: dict,
    train_window_size: int,
    test_window_size: int,
    step_size: int,
    initial_capital: float = 10000.0,
    transaction_cost_bps: float = 1.0,
    optimization_metric: str = 'sharpe_ratio' # or 'annualized_return', 'total_return'
) -> dict:
    """
    Performs Walk-Forward Optimization (WFO) to find optimal parameters for the
    CyberneticTrader and BacktestingEngine.

    Args:
        price_data_series (pd.Series): Historical price data for the asset.
        trader_param_grid (dict): Dictionary where keys are CyberneticTrader
                                  __init__ parameters and values are lists of
                                  values to test for each parameter.
        backtester_param_grid (dict): Dictionary for BacktestingEngine __init__
                                      parameters (max_position_size, stop_loss_pct,
                                      take_profit_pct).
        train_window_size (int): Number of data points in the training window.
        test_window_size (int): Number of data points in the testing window.
        step_size (int): How many data points to advance the window in each step.
        initial_capital (float): Initial capital for each backtest run.
        transaction_cost_bps (float): Transaction costs in basis points.
        optimization_metric (str): The metric to optimize for ('sharpe_ratio',
                                   'annualized_return', 'total_return').

    Returns:
        dict: A dictionary containing WFO results, including best parameters
              for each window and overall out-of-sample performance.
    """
    if not isinstance(price_data_series, pd.Series):
        raise TypeError("price_data_series must be a pandas Series.")
    if price_data_series.empty:
        raise ValueError("price_data_series cannot be empty.")
    if train_window_size <= 0 or test_window_size <= 0 or step_size <= 0:
        raise ValueError("Window sizes and step size must be positive integers.")
    if train_window_size + test_window_size > len(price_data_series):
        raise ValueError("Total window size (train + test) exceeds data length.")

    wfo_results = []
    total_out_of_sample_portfolio_value = pd.Series([initial_capital], index=[price_data_series.index[0]])

    # Generate all combinations of parameters for the grid search
    trader_keys = trader_param_grid.keys()
    trader_values = trader_param_grid.values()
    trader_param_combinations = [dict(zip(trader_keys, v)) for v in itertools.product(*trader_values)]

    backtester_keys = backtester_param_grid.keys()
    backtester_values = backtester_param_grid.values()
    backtester_param_combinations = [dict(zip(backtester_keys, v)) for v in itertools.product(*backtester_values)]

    print(f"Starting Walk-Forward Optimization with {len(price_data_series)} data points.")
    print(f"Training Window: {train_window_size} periods, Testing Window: {test_window_size} periods, Step: {step_size} periods.")
    print(f"Optimizing for: {optimization_metric}")
    print(f"Total parameter combinations to test per window: {len(trader_param_combinations) * len(backtester_param_combinations)}")

    # Iterate through rolling windows
    start_index = 0
    while start_index + train_window_size + test_window_size <= len(price_data_series):
        train_end_index = start_index + train_window_size
        test_end_index = train_end_index + test_window_size

        train_data = price_data_series.iloc[start_index:train_end_index]
        test_data = price_data_series.iloc[train_end_index:test_end_index]

        print(f"\n--- WFO Window: {train_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ---")
        print(f"  Training Period: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Testing Period:  {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")

        best_metric_in_train = -np.inf # Maximize Sharpe, Return
        best_trader_params = None
        best_backtester_params = None

        # Grid search within the training window
        for t_params in trader_param_combinations:
            for b_params in backtester_param_combinations:
                try:
                    trader = CyberneticTrader(**t_params)
                    backtester = BacktestingEngine(
                        trader_instance=trader,
                        initial_capital=initial_capital,
                        transaction_cost_bps=transaction_cost_bps,
                        **b_params
                    )
                    backtester.run_backtest(train_data)
                    _, _, metrics = backtester.analyze_results()

                    current_metric = metrics.get(optimization_metric)

                    if current_metric is not None and not np.isnan(current_metric) and current_metric > best_metric_in_train:
                        best_metric_in_train = current_metric
                        best_trader_params = t_params
                        best_backtester_params = b_params
                except Exception as e:
                    # print(f"  Error during training backtest with params {t_params}, {b_params}: {e}")
                    continue # Skip this combination

        if best_trader_params is None:
            print("  No valid parameters found for this training window. Skipping to next window.")
            start_index += step_size
            continue

        print(f"  Best parameters found in training: Trader: {best_trader_params}, Backtester: {best_backtester_params}")
        print(f"  Best {optimization_metric} in training: {best_metric_in_train:.4f}")

        # Run out-of-sample test on the testing window with best parameters
        try:
            oos_trader = CyberneticTrader(**best_trader_params)
            oos_backtester = BacktestingEngine(
                trader_instance=oos_trader,
                initial_capital=initial_capital,
                transaction_cost_bps=transaction_cost_bps,
                **best_backtester_params
            )
            oos_backtester.run_backtest(test_data)
            oos_trade_log, oos_portfolio_series, oos_metrics = oos_backtester.analyze_results()

            oos_result = {
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_trader_params': best_trader_params,
                'best_backtester_params': best_backtester_params,
                'train_metric_value': best_metric_in_train,
                'oos_metric_value': oos_metrics.get(optimization_metric),
                'oos_total_return': oos_metrics.get('total_return'),
                'oos_annualized_return': oos_metrics.get('annualized_return'),
                'oos_sharpe_ratio': oos_metrics.get('sharpe_ratio'),
                'oos_max_drawdown': oos_metrics.get('max_drawdown'),
                'oos_portfolio_series': oos_portfolio_series # Store for overall performance
            }
            wfo_results.append(oos_result)

            print(f"  Out-of-sample {optimization_metric}: {oos_metrics.get(optimization_metric):.4f}")
            print(f"  Out-of-sample Total Return: {oos_metrics.get('total_return'):.2%}")

            # Concatenate out-of-sample portfolio values for overall performance visualization
            # Ensure correct concatenation by removing overlapping initial capital if already present
            if not total_out_of_sample_portfolio_value.empty:
                # Align indices for concatenation, drop first element if it's just initial capital
                # from the previous window's portfolio series.
                # The first value of oos_portfolio_series is the initial capital for that test window.
                # We want to append from the *second* value onwards, as the first value is effectively
                # the *last* value of the previous window's capital (if we were chaining).
                # A more robust way is to rebase the OOS portfolio to the last capital of the previous window.
                last_overall_capital = total_out_of_sample_portfolio_value.iloc[-1]
                rebased_oos_portfolio = oos_portfolio_series / oos_portfolio_series.iloc[0] * last_overall_capital
                # Remove the first point of rebased_oos_portfolio as it duplicates the last point of total_out_of_sample_portfolio_value
                total_out_of_sample_portfolio_value = pd.concat([total_out_of_sample_portfolio_value, rebased_oos_portfolio.iloc[1:]])
            else:
                total_out_of_sample_portfolio_value = oos_portfolio_series


        except Exception as e:
            print(f"  Error during out-of-sample backtest: {e}")
            wfo_results.append({
                'train_start': train_data.index[0],
                'train_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'best_trader_params': best_trader_params,
                'best_backtester_params': best_backtester_params,
                'train_metric_value': best_metric_in_train,
                'oos_metric_value': np.nan,
                'oos_total_return': np.nan,
                'oos_annualized_return': np.nan,
                'oos_sharpe_ratio': np.nan,
                'oos_max_drawdown': np.nan,
                'oos_portfolio_series': pd.Series([initial_capital], index=[test_data.index[0]]) # Placeholder
            })

        start_index += step_size

    print("\n--- Walk-Forward Optimization Summary ---")
    wfo_df = pd.DataFrame(wfo_results)
    print(f"Total WFO windows processed: {len(wfo_df)}")
    print(f"Average out-of-sample {optimization_metric}: {wfo_df['oos_metric_value'].mean():.4f}")
    print(f"Average out-of-sample Total Return: {wfo_df['oos_total_return'].mean():.2%}")

    # Calculate overall performance from concatenated portfolio series
    overall_total_return = (total_out_of_sample_portfolio_value.iloc[-1] / total_out_of_sample_portfolio_value.iloc[0]) - 1
    overall_num_trading_days = len(total_out_of_sample_portfolio_value) - 1 # Exclude initial capital point
    overall_annualized_return = (1 + overall_total_return)**(252 / overall_num_trading_days) - 1 if overall_num_trading_days > 0 else 0.0
    overall_portfolio_daily_returns = total_out_of_sample_portfolio_value.pct_change().dropna()
    overall_annualized_volatility = overall_portfolio_daily_returns.std() * np.sqrt(252) if not overall_portfolio_daily_returns.empty else 0.0
    overall_sharpe_ratio = overall_annualized_return / overall_annualized_volatility if overall_annualized_volatility != 0 else np.nan
    overall_peak = total_out_of_sample_portfolio_value.expanding(min_periods=1).max()
    overall_drawdown = (total_out_of_sample_portfolio_value - overall_peak) / overall_peak
    overall_max_drawdown = overall_drawdown.min() if not overall_drawdown.empty else 0.0

    print("\n--- Overall Out-of-Sample Performance ---")
    print(f"Overall Final Capital:          ${total_out_of_sample_portfolio_value.iloc[-1]:,.2f}")
    print(f"Overall Total Return:           {overall_total_return:.2%}")
    print(f"Overall Annualized Return:      {overall_annualized_return:.2%}")
    print(f"Overall Annualized Volatility:  {overall_annualized_volatility:.2%}")
    print(f"Overall Sharpe Ratio:           {overall_sharpe_ratio:.2f}")
    print(f"Overall Max Drawdown:           {overall_max_drawdown:.2%}")

    return {
        'wfo_results_df': wfo_df,
        'overall_portfolio_series': total_out_of_sample_portfolio_value,
        'overall_metrics': {
            'final_capital': total_out_of_sample_portfolio_value.iloc[-1],
            'total_return': overall_total_return,
            'annualized_return': overall_annualized_return,
            'annualized_volatility': overall_annualized_volatility,
            'sharpe_ratio': overall_sharpe_ratio,
            'max_drawdown': overall_max_drawdown
        }
    }


# -------------------------------------------------------------------------------------
# A simple real world WFO example
# -------------------------------------------------------------------------------------

if __name__ == "__main__":
    # Download real-world data for AAPL
    ticker = "TNA"
    start_date = "2021-01-01" # Extended history for WFO
    end_date = "2025-06-30"
    print(f"\n--- Downloading {ticker} data for WFO ({start_date} to {end_date}) ---")
    data = yf.download(ticker, start=start_date, end=end_date)
    price_series_wfo = data['Close'].dropna().squeeze()

    if not isinstance(price_series_wfo, pd.Series) or price_series_wfo.empty:
        print(f"Could not load valid price series for {ticker}. Exiting WFO example.")
    else:
        # Define parameter grids for optimization
        trader_param_grid = {
            'history_length': [10, 20, 30],
            'learning_rate': [0.0005, 0.001, 0.002],
            'regularization_strength': [0.0001, 0.0005, 0.001],
            'order_wiener_expansion': [2], # Keeping fixed for now
            'normalize_returns': [True]
        }

        backtester_param_grid = {
            'max_position_size': [0.5, 0.8, 1.0],
            'stop_loss_pct': [0.03, 0.05],
            'take_profit_pct': [0.05, 0.10]
        }

        # WFO window settings (e.g., 2 quarters train, 1 quarter test, slide by 1 quarter)
        train_window_size = 126 # ~2 quarters of trading days
        test_window_size = 63   # ~1 quarter of trading days
        step_size = 63         # Re-optimize every quarter

        wfo_results = walk_forward_optimization(
            price_data_series=price_series_wfo,
            trader_param_grid=trader_param_grid,
            backtester_param_grid=backtester_param_grid,
            train_window_size=train_window_size,
            test_window_size=test_window_size,
            step_size=step_size,
            initial_capital=10000.0,
            transaction_cost_bps=0.5,
            optimization_metric='sharpe_ratio'
        )

        # Plot the overall out-of-sample portfolio performance
        plt.figure(figsize=(14, 7))
        plt.plot(wfo_results['overall_portfolio_series'].index, wfo_results['overall_portfolio_series'], label='Overall OOS Portfolio Value')
        plt.title(f'Walk-Forward Optimized Portfolio Performance for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Capital ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # You can further analyze wfo_results['wfo_results_df']
        # to see which parameters were chosen in each window and their OOS performance.
        # print("\nDetailed WFO Window Results (first 5 rows):")
        # print(wfo_results['wfo_results_df'].head())
        # print("\nDetailed WFO Window Results (last 5 rows):")
        # print(wfo_results['wfo_results_df'].tail())
