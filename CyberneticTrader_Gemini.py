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
# For entropy calculations, you might need to install these libraries:
# pip install nolds
from nolds import sampen # For Sample Entropy

# For real-world data fetching:
# pip install yfinance
import yfinance as yf

class CyberneticTrader:
    """
    Implements a cybernetic forecasting and trading agent based on Norbert Wiener's
    principles of control, communication, and non-linear system identification.

    It uses a simplified Volterra series (approximating Wiener integrals) to forecast
    financial returns and adaptively adjusts its internal 'kernels' (coefficients)
    based on feedback from actual market outcomes.
    """

    def __init__(self, history_length=20, learning_rate=0.001,
                 order_wiener_expansion=2, regularization_strength=0.001,
                 normalize_returns=True):
        """
        Initializes the CyberneticTrader with key parameters for its forecasting model
        and adaptive learning mechanism.

        Args:
            history_length (int): The number of past return observations (memory)
                                  used to make a future forecast. This defines the
                                  dimension of our input feature space.
            learning_rate (float): A small positive value controlling the step size
                                   for kernel adjustments during adaptation. A higher
                                   value leads to faster but potentially unstable learning.
            order_wiener_expansion (int): The maximum order of non-linearity to model.
                                          - 0: Constant forecast (K0)
                                          - 1: Linear forecast (K0 + K1)
                                          - 2: Quadratic forecast (K0 + K1 + K2), etc.
                                          Higher orders capture more complex non-linearities
                                          but increase computational complexity and risk of overfitting.
            regularization_strength (float): L2 regularization strength. A positive value
                                             penalizes large kernel values, preventing overfitting.
            normalize_returns (bool): If True, returns history will be normalized to
                                      mean 0 and std 1 before forecasting and learning.
                                      This can improve stability of gradient descent.
        """
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

        # Initialize kernel functions (coefficients) for the Wiener expansion.
        self.kernels = self._initialize_kernels(order_wiener_expansion)

        # Stores the most recent 'history_length' financial returns.
        self.returns_history = np.zeros(self.history_length)

        # Stores the last predicted return. Crucial for the feedback loop.
        self.last_predicted_return = 0.0

        # Flag to indicate if enough data has been accumulated to start forecasting
        self.is_initialized_for_forecast = False
        self.history_counter = 0 # To track how many data points have been accumulated

        # For rolling normalization
        self.rolling_mean = 0.0
        self.rolling_std = 1.0 # Avoid division by zero initially

    def _initialize_kernels(self, order):
        """
        Initializes the kernel functions (coefficients) with zeros.
        These will be adaptively learned during the trading process.
        """
        kernels = {}
        kernels[0] = 0.0 # K0 (constant term)

        if order >= 1:
            kernels[1] = np.zeros(self.history_length) # K1 (linear term)

        if order >= 2:
            # K2 (quadratic term): captures pairwise interaction effects.
            # Initialized as a symmetric matrix.
            kernels[2] = np.zeros((self.history_length, self.history_length))

        if order > 2:
            print(f"Warning: Wiener expansion order {order} is set, but only orders up to 2 are explicitly implemented.")
            print("Higher order kernels will not be initialized or used in this version.")

        return kernels

    def _calculate_entropy(self, data):
        """
        Calculates Sample Entropy for the given financial data.
        Higher entropy indicates higher unpredictability (more noise, less information).
        Lower entropy suggests more structure/predictability (more information).
        Requires 'nolds' library.
        """
        if len(data) < 2: # Sample Entropy needs at least 2 points
            return 0.0

        # Check if data is essentially constant (zero variance)
        if np.isclose(np.std(data), 0):
            return 0.0 # A constant series has zero entropy (perfect predictability)

        try:
            # emb_dim (m): embedding dimension, tolerance (r): similarity criterion
            # r is typically 0.1 to 0.25 * standard deviation of the data
            # Ensure tolerance is not too small to avoid numerical issues
            tolerance_val = max(0.2 * np.std(data), 1e-9)
            return sampen(data, emb_dim=2, tolerance=tolerance_val)
        except Exception as e:
            # print(f"Error calculating Sample Entropy: {e}") # For debugging
            return 0.0 # Return 0.0 instead of NaN on error for numerical stability


    def _wiener_forecast(self, current_history_normalized):
        """
        Performs a non-linear forecast using the estimated Wiener kernels.
        This is a discrete approximation of the Volterra series expansion.
        Expects `current_history_normalized` to be already normalized if self.normalize_returns is True.
        """
        forecast = self.kernels[0] # K0 term (constant bias)

        # K1 term (linear component)
        if 1 in self.kernels:
            forecast += np.dot(self.kernels[1], current_history_normalized)

        # K2 term (quadratic component - pairwise interactions)
        if 2 in self.kernels:
            # Efficient computation for K2 term
            forecast += np.sum(self.kernels[2] * np.outer(current_history_normalized, current_history_normalized))

        return forecast

    def _adaptive_kernel_adjustment(self, actual_return_normalized, predicted_return_normalized, history_at_prediction_time_normalized):
        """
        The core cybernetic feedback mechanism.
        Adjusts the kernel coefficients based on the error between the actual observed
        return and the model's prediction, incorporating L2 regularization.

        Args:
            actual_return_normalized (float): The true market return (normalized).
            predicted_return_normalized (float): The return that the model predicted (normalized).
            history_at_prediction_time_normalized (np.array): The historical data (normalized)
                                                               used to make the `predicted_return_normalized`.
        """
        error = actual_return_normalized - predicted_return_normalized

        # Update K0 with regularization
        self.kernels[0] += self.learning_rate * error - self.learning_rate * self.regularization_strength * self.kernels[0]

        # Update K1 with regularization
        if 1 in self.kernels:
            self.kernels[1] += self.learning_rate * error * history_at_prediction_time_normalized \
                               - self.learning_rate * self.regularization_strength * self.kernels[1]

        # Update K2 with regularization
        if 2 in self.kernels:
            # Element-wise update with regularization
            outer_product = np.outer(history_at_prediction_time_normalized, history_at_prediction_time_normalized)
            self.kernels[2] += self.learning_rate * error * outer_product \
                               - self.learning_rate * self.regularization_strength * self.kernels[2]
            # Ensure symmetry after update
            self.kernels[2] = (self.kernels[2] + self.kernels[2].T) / 2


    def process_data_point(self, current_return):
        """
        Processes a new financial return data point. This involves:
        1. Updating the historical data buffer.
        2. If sufficient history, making a new forecast.
        3. If a previous forecast exists, applying the cybernetic feedback
           to adapt the model's kernels.

        Args:
            current_return (float): The latest observed financial return.

        Returns:
            tuple: (signal, predicted_return, current_entropy)
                   - signal (int): 1 for BUY, -1 for SELL, 0 for HOLD/initializing.
                   - predicted_return (float): The forecast for the *next* period (unnormalized).
                   - current_entropy (float): Entropy of the current history window.
                   Returns (None, None, None) if not enough history to forecast yet.
        """
        # Store the current state of history *before* adding the new return.
        # This is the history that was used to make the `last_predicted_return`.
        history_for_feedback = np.copy(self.returns_history)

        # Update the returns history buffer (acts as a sliding window)
        if self.history_counter < self.history_length:
            self.returns_history[self.history_counter] = current_return
            self.history_counter += 1
        else:
            self.returns_history = np.roll(self.returns_history, -1)
            self.returns_history[-1] = current_return
            self.is_initialized_for_forecast = True

        signal = 0 # Default to HOLD
        predicted_return_for_next_period = None
        current_entropy = None

        # Normalize the history and current return for internal model use if enabled
        current_history_normalized = np.copy(self.returns_history)
        current_return_normalized = current_return
        history_for_feedback_normalized = np.copy(history_for_feedback)
        last_predicted_return_normalized = self.last_predicted_return # This is already normalized if `normalize_returns` was True for previous step

        if self.normalize_returns and self.history_counter > 1:
            # Update rolling mean and std
            self.rolling_mean = np.mean(self.returns_history[:self.history_counter])
            self.rolling_std = np.std(self.returns_history[:self.history_counter])
            if self.rolling_std == 0:
                self.rolling_std = 1e-6 # Avoid division by zero

            current_history_normalized = (self.returns_history - self.rolling_mean) / self.rolling_std
            current_return_normalized = (current_return - self.rolling_mean) / self.rolling_std
            history_for_feedback_normalized = (history_for_feedback - self.rolling_mean) / self.rolling_std


        if self.is_initialized_for_forecast:
            # 1. Cybernetic Feedback (Adaptation):
            # Use the `current_return` (actual outcome) and the `last_predicted_return`
            # (prediction made for this period) to adjust kernels.
            self._adaptive_kernel_adjustment(current_return_normalized, last_predicted_return_normalized, history_for_feedback_normalized)

            # 2. Forecast for the *next* period:
            # Use the *updated* kernels and the *latest* normalized history to make a new prediction.
            predicted_return_for_next_period_normalized = self._wiener_forecast(current_history_normalized)

            # Denormalize the prediction before returning
            predicted_return_for_next_period = predicted_return_for_next_period_normalized * self.rolling_std + self.rolling_mean

            # 3. Determine Trading Signal:
            signal = np.sign(predicted_return_for_next_period)
            if signal == 0: # If prediction is exactly zero, default to hold
                signal = 0

            # 4. Update last_predicted_return for the next feedback cycle (store normalized for consistency)
            self.last_predicted_return = predicted_return_for_next_period_normalized

            # 5. Calculate entropy of the current history window (on unnormalized data)
            current_entropy = self._calculate_entropy(self.returns_history[:self.history_counter])

        return signal, predicted_return_for_next_period, current_entropy


class BacktestingEngine:
    """
    Simulates the trading process using a CyberneticTrader instance over historical data.
    It tracks portfolio value, trade actions, and provides performance analysis.
    Incorporates dynamic position sizing, stop-loss, and take-profit mechanisms.
    """

    def __init__(self, trader_instance, initial_capital=10000.0, transaction_cost_bps=1.0,
                 max_position_size=1.0, stop_loss_pct=0.02, take_profit_pct=0.03):
        """
        Initializes the backtesting engine.

        Args:
            trader_instance (CyberneticTrader): An instance of the CyberneticTrader.
            initial_capital (float): The starting capital for the backtest.
            transaction_cost_bps (float): Transaction cost in basis points (e.g., 1.0 for 0.01%).
                                          Applied on each side of a trade (buy and sell).
            max_position_size (float): Maximum fraction of capital to allocate to a position (0.0 to 1.0).
                                       If 1.0, it's 100% capital.
            stop_loss_pct (float): Percentage loss at which to close a position (e.g., 0.02 for 2%).
            take_profit_pct (float): Percentage gain at which to close a position (e.g., 0.03 for 3%).
        """
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

        self.current_position = 0 # 1 for long, -1 for short, 0 for flat
        self.position_entry_price = None # Price at which the current position was opened
        self.position_size_value = 0 # Absolute value of capital allocated to current position

        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

    def _execute_trade(self, date, current_price, daily_return, signal_for_next_period, predicted_return, current_entropy):
        """
        Internal helper to execute trading logic for a single day.
        Handles position changes, P&L, transaction costs, stop-loss/take-profit.
        """
        pnl_today = 0
        transaction_cost = 0
        action_today = "HOLD" # Default action if no change or no position

        # Check for stop-loss or take-profit triggers if a position is open
        if self.current_position != 0 and self.position_entry_price is not None:
            if self.current_position == 1: # Long position
                if current_price <= self.position_entry_price * (1 - self.stop_loss_pct):
                    action_today = "STOP_LOSS_LONG"
                    # Close position at current price
                    pnl_today = self.position_size_value * (current_price / self.position_entry_price - 1)
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                elif current_price >= self.position_entry_price * (1 + self.take_profit_pct):
                    action_today = "TAKE_PROFIT_LONG"
                    # Close position at current price
                    pnl_today = self.position_size_value * (current_price / self.position_entry_price - 1)
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                else: # Position still open, calculate daily P&L
                    pnl_today = self.position_size_value * daily_return

            elif self.current_position == -1: # Short position
                if current_price >= self.position_entry_price * (1 + self.stop_loss_pct):
                    action_today = "STOP_LOSS_SHORT"
                    # Close position at current price
                    pnl_today = self.position_size_value * (self.position_entry_price / current_price - 1) # Inverse return for short
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                elif current_price <= self.position_entry_price * (1 - self.take_profit_pct):
                    action_today = "TAKE_PROFIT_SHORT"
                    # Close position at current price
                    pnl_today = self.position_size_value * (self.position_entry_price / current_price - 1)
                    transaction_cost += self.position_size_value * self.transaction_cost_rate
                    self.current_position = 0
                    self.position_entry_price = None
                    self.position_size_value = 0
                else: # Position still open, calculate daily P&L
                    pnl_today = self.position_size_value * (-daily_return) # Inverse return for short

        self.capital += pnl_today

        # Decide on new position based on signal_for_next_period
        # Only change position if signal differs from current_position AND no SL/TP triggered
        if self.current_position == 0 and signal_for_next_period != 0: # No position, signal to open
            action_today = "BUY" if signal_for_next_period == 1 else "SELL"
            self.current_position = signal_for_next_period
            self.position_entry_price = current_price
            self.position_size_value = self.capital * self.max_position_size # Allocate max allowed capital
            transaction_cost += self.position_size_value * self.transaction_cost_rate
        elif self.current_position != 0 and signal_for_next_period == 0: # Has position, signal to close
            action_today = "FLAT"
            transaction_cost += self.position_size_value * self.transaction_cost_rate
            self.current_position = 0
            self.position_entry_price = None
            self.position_size_value = 0
        elif self.current_position != 0 and signal_for_next_period != 0 and signal_for_next_period != self.current_position: # Change direction
            action_today = "REVERSE"
            # Close old position
            transaction_cost += self.position_size_value * self.transaction_cost_rate
            # Open new position
            self.current_position = signal_for_next_period
            self.position_entry_price = current_price
            self.position_size_value = self.capital * self.max_position_size
            transaction_cost += self.position_size_value * self.transaction_cost_rate
        # If signal_for_next_period matches current_position and no SL/TP, action_today remains "HOLD"

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
        """
        Runs the backtest simulation over the provided price data.

        Args:
            price_data_series (pd.Series): A pandas Series of asset prices,
                                           indexed by date/time.
        """
        if not isinstance(price_data_series, pd.Series):
            raise TypeError("price_data_series must be a pandas Series.")
        if price_data_series.empty:
            print("Warning: Empty price_data_series provided for backtest.")
            return

        print(f"Starting backtest with ${self.initial_capital:,.2f} capital over {len(price_data_series)-1} periods.")

        # Iterate through prices to simulate daily returns and trading
        for i in range(1, len(price_data_series)):
            date = price_data_series.index[i]
            current_price = price_data_series.iloc[i]
            previous_price = price_data_series.iloc[i-1]
            daily_return = (current_price - previous_price) / previous_price

            # Process the daily return with the CyberneticTrader
            # This will update the trader's history, adapt kernels, and generate a signal
            # for the *next* period based on data *up to and including* current_price.
            signal_for_next_period, predicted_return, current_entropy = \
                self.trader.process_data_point(daily_return)

            # Only start executing trades once the trader has enough history to forecast
            if self.trader.is_initialized_for_forecast:
                self._execute_trade(date, current_price, daily_return, signal_for_next_period, predicted_return, current_entropy)
            else:
                # Still initializing history, no trading yet
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


        print(f"Backtest finished. Final capital: ${self.capital:,.2f}")

    def analyze_results(self):
        """
        Analyzes and presents the backtest results, calculating key performance indicators.
        """
        results_df = pd.DataFrame(self.trade_log)

        # Ensure portfolio history is a Series with a proper index
        # The portfolio_history includes the initial capital, so it's one element longer than trade_log
        if not results_df.empty:
            # Create a full date range for the portfolio history, starting from the day before the first trade
            # This handles the initial_capital point correctly.
            all_portfolio_dates = [results_df['date'].iloc[0] - pd.Timedelta(days=1)] + list(results_df['date'])
            portfolio_series = pd.Series(self.portfolio_history, index=pd.to_datetime(all_portfolio_dates), name="Portfolio Value")
        else:
            portfolio_series = pd.Series([self.initial_capital], name="Portfolio Value")


        total_return = (self.capital / self.initial_capital) - 1
        num_trading_days = len(results_df[results_df['action_today'] != 'INITIALIZING'])

        if num_trading_days > 0:
            # Calculate annualized return (assuming daily data and 252 trading days in a year)
            annualized_return = (1 + total_return)**(252 / num_trading_days) - 1
        else:
            annualized_return = 0.0

        # Calculate volatility (standard deviation of daily returns of the strategy)
        portfolio_daily_returns = portfolio_series.pct_change().dropna()
        annualized_volatility = portfolio_daily_returns.std() * np.sqrt(252) if not portfolio_daily_returns.empty else 0.0

        # Sharpe Ratio (assuming 0 risk-free rate for simplicity)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else np.nan

        # Max Drawdown
        peak = portfolio_series.expanding(min_periods=1).max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min() if not drawdown.empty else 0.0

        print("\n--- Backtest Summary ---")
        print(f"Initial Capital:        ${self.initial_capital:,.2f}")
        print(f"Final Capital:          ${self.capital:,.2f}")
        print(f"Total Return:           {total_return:.2%}")
        print(f"Annualized Return:      {annualized_return:.2%}")
        print(f"Annualized Volatility:  {annualized_volatility:.2%}")
        print(f"Sharpe Ratio:           {sharpe_ratio:.2f}")
        print(f"Max Drawdown:           {max_drawdown:.2%}")
        print(f"Total Transaction Costs: ${results_df['transaction_cost'].sum():,.2f}")
        print(f"Number of Trading Days: {num_trading_days}")

        return results_df, portfolio_series

# --- Example Usage ---
if __name__ == "__main__":
    # 1. Generate Synthetic Data (Brownian Motion for benchmarking)
    np.random.seed(42)
    num_days = 500
    initial_price = 100.0
    daily_returns_bm = np.random.normal(loc=0.0001, scale=0.01, size=num_days)
    prices_bm = initial_price * (1 + daily_returns_bm).cumprod()
    dates = pd.date_range(start='2020-01-01', periods=num_days, freq='B')
    price_series_bm = pd.Series(prices_bm, index=dates)

    print("--- Running Backtest on Synthetic Brownian Motion Data ---")
    trader_bm = CyberneticTrader(history_length=20, learning_rate=0.0001,
                                 order_wiener_expansion=2, regularization_strength=0.0001,
                                 normalize_returns=True)
    backtester_bm = BacktestingEngine(trader_bm, initial_capital=10000,
                                      transaction_cost_bps=1.0, max_position_size=0.8,
                                      stop_loss_pct=0.05, take_profit_pct=0.10)
    backtester_bm.run_backtest(price_series_bm)
    trade_log_bm, portfolio_series_bm = backtester_bm.analyze_results()

    # Plotting portfolio value for Brownian Motion
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series_bm.index, portfolio_series_bm, label='Portfolio Value (Brownian Motion)')
    plt.title('Portfolio Value Over Time (Brownian Motion Data)')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # 2. Example with a slightly more predictable (simulated) series (Mean-Reverting AR(1))
    print("\n--- Running Backtest on Simulated Data with Some Structure (Mean-Reverting) ---")
    phi_mr = -0.2 # Stronger mean-reverting tendency
    c_mr = 0.0001
    epsilon_mr = np.random.normal(loc=0, scale=0.005, size=num_days)
    simulated_returns_mr = np.zeros(num_days)
    simulated_returns_mr[0] = epsilon_mr[0]

    for t in range(1, num_days):
        simulated_returns_mr[t] = c_mr + phi_mr * simulated_returns_mr[t-1] + epsilon_mr[t]

    prices_structured_mr = initial_price * (1 + simulated_returns_mr).cumprod()
    price_series_structured_mr = pd.Series(prices_structured_mr, index=dates)

    trader_structured_mr = CyberneticTrader(history_length=20, learning_rate=0.005,
                                            order_wiener_expansion=2, regularization_strength=0.0005,
                                            normalize_returns=True)
    backtester_structured_mr = BacktestingEngine(trader_structured_mr, initial_capital=10000,
                                                 transaction_cost_bps=1.0, max_position_size=0.8,
                                                 stop_loss_pct=0.05, take_profit_pct=0.10)
    backtester_structured_mr.run_backtest(price_series_structured_mr)
    trade_log_structured_mr, portfolio_series_structured_mr = backtester_structured_mr.analyze_results()

    # Plotting portfolio value for structured data
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series_structured_mr.index, portfolio_series_structured_mr, label='Portfolio Value (Structured Data)')
    plt.title('Portfolio Value Over Time (Structured Data - Mean Reverting)')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3. Example with a slightly more predictable (simulated) series (Trending AR(1))
    print("\n--- Running Backtest on Simulated Data with Some Structure (Trending) ---")
    phi_trend = 0.05 # Slight trending tendency
    c_trend = 0.0005 # Small positive drift
    epsilon_trend = np.random.normal(loc=0, scale=0.005, size=num_days)
    simulated_returns_trend = np.zeros(num_days)
    simulated_returns_trend[0] = epsilon_trend[0]

    for t in range(1, num_days):
        simulated_returns_trend[t] = c_trend + phi_trend * simulated_returns_trend[t-1] + epsilon_trend[t]

    prices_structured_trend = initial_price * (1 + simulated_returns_trend).cumprod()
    price_series_structured_trend = pd.Series(prices_structured_trend, index=dates)

    trader_structured_trend = CyberneticTrader(history_length=20, learning_rate=0.005,
                                            order_wiener_expansion=2, regularization_strength=0.0005,
                                            normalize_returns=True)
    backtester_structured_trend = BacktestingEngine(trader_structured_trend, initial_capital=10000,
                                                 transaction_cost_bps=1.0, max_position_size=0.8,
                                                 stop_loss_pct=0.05, take_profit_pct=0.10)
    backtester_structured_trend.run_backtest(price_series_structured_trend)
    trade_log_structured_trend, portfolio_series_structured_trend = backtester_structured_trend.analyze_results()

    # Plotting portfolio value for structured data
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_series_structured_trend.index, portfolio_series_structured_trend, label='Portfolio Value (Structured Data)')
    plt.title('Portfolio Value Over Time (Structured Data - Trending)')
    plt.xlabel('Date')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # 4. Real-World Data Example (using yfinance)
    print("\n--- Running Backtest on Real-World Data (AAPL) ---")
    try:
        # Download historical data for Apple (AAPL)
        ticker = "TNA"
        start_date = "2020-01-01"
        end_date = "2025-06-30"
        data = yf.download(ticker, start=start_date, end=end_date)

        if data.empty:
            print(f"No data downloaded for {ticker} between {start_date} and {end_date}. Check ticker or date range.")
        elif 'Close' not in data.columns:
            print(f"'{ticker}' data downloaded but 'Close' column not found.")
        else:
            # Explicitly select 'Close' column and convert to Series using .squeeze()
            price_series_real = data['Close'].dropna().squeeze()

            # After squeeze(), it should always be a Series if there's data.
            # If it's still a DataFrame (e.g., if .squeeze() didn't work as expected
            # due to some edge case, though highly unlikely for a single column),
            # the next check will catch it.
            if not isinstance(price_series_real, pd.Series):
                print(f"Error: price_series_real is not a pandas Series after squeeze(). Type: {type(price_series_real)}")
                # Fallback: try to convert explicitly again, though this indicates a deeper issue
                price_series_real = pd.Series(price_series_real)

            if not price_series_real.empty:
                trader_real = CyberneticTrader(history_length=30, learning_rate=0.0005,
                                                order_wiener_expansion=2, regularization_strength=0.0001,
                                                normalize_returns=True)
                backtester_real = BacktestingEngine(trader_real, initial_capital=10000,
                                                    transaction_cost_bps=1.0, max_position_size=0.8,
                                                    stop_loss_pct=0.03, take_profit_pct=0.05)
                backtester_real.run_backtest(price_series_real)
                trade_log_real, portfolio_series_real = backtester_real.analyze_results()

                plt.figure(figsize=(12, 6))
                plt.plot(portfolio_series_real.index, portfolio_series_real, label=f'Portfolio Value ({ticker} Real Data)')
                plt.title(f'Portfolio Value Over Time ({ticker} Real-World Data)')
                plt.xlabel('Date')
                plt.ylabel('Capital ($)')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print(f"Price series for {ticker} is empty after dropping NaNs. Check data quality or date range.")

    except Exception as e:
        print(f"\nError loading or processing real data with yfinance: {e}")
        print("Please ensure 'yfinance' is installed (`pip install yfinance`).")

