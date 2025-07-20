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
from nolds import sampen # For Sample Entropy

class CyberneticForecaster:
    """
    A generalized forecasting agent based on Norbert Wiener's cybernetic principles.
    It uses a simplified Volterra series (approximating Wiener integrals) to forecast
    stationary time series and adaptively adjusts its internal 'kernels' (coefficients)
    based on feedback from actual observed values.
    """

    def __init__(self, history_length=20, learning_rate=0.001,
                 order_wiener_expansion=2, regularization_strength=0.001,
                 normalize_data=True):
        """
        Initializes the CyberneticForecaster with key parameters for its forecasting model
        and adaptive learning mechanism.

        Args:
            history_length (int): The number of past data points (memory)
                                  used to make a future forecast.
            learning_rate (float): Controls the adaptation speed of the model's kernels.
            order_wiener_expansion (int): The maximum order of non-linearity to model.
                                          (0: Constant, 1: Linear, 2: Quadratic).
            regularization_strength (float): L2 regularization strength to prevent overfitting.
            normalize_data (bool): If True, input data history will be normalized to
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
        if not isinstance(normalize_data, bool):
            raise ValueError("normalize_data must be a boolean.")

        self.history_length = history_length
        self.learning_rate = learning_rate
        self.order_wiener_expansion = order_wiener_expansion
        self.regularization_strength = regularization_strength
        self.normalize_data = normalize_data

        # Initialize kernel functions (coefficients) for the Wiener expansion.
        self.kernels = self._initialize_kernels(order_wiener_expansion)

        # Stores the most recent 'history_length' data points.
        self.data_history = np.zeros(self.history_length)

        # Stores the last predicted value. Crucial for the feedback loop.
        self.last_predicted_value = 0.0

        # Flag to indicate if enough data has been accumulated to start forecasting
        self.is_initialized_for_forecast = False
        self.history_counter = 0 # To track how many data points have been accumulated

        # For rolling normalization
        self.rolling_mean = 0.0
        self.rolling_std = 1.0 # Avoid division by zero initially

    def _initialize_kernels(self, order):
        """
        Initializes the kernel functions (coefficients) with zeros.
        These will be adaptively learned during the forecasting process.
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
            # print(f"Warning: Wiener expansion order {order} is set, but only orders up to 2 are explicitly implemented.")
            pass # Silencing this warning as it's handled in the manual

        return kernels

    def _calculate_entropy(self, data):
        """
        Calculates Sample Entropy for the given data.
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
        except Exception:
            return 0.0 # Return 0.0 instead of NaN on error for numerical stability


    def _wiener_forecast(self, current_history_normalized):
        """
        Performs a non-linear forecast using the estimated Wiener kernels.
        This is a discrete approximation of the Volterra series expansion.
        Expects `current_history_normalized` to be already normalized if self.normalize_data is True.
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

    def _adaptive_kernel_adjustment(self, actual_value_normalized, predicted_value_normalized, history_at_prediction_time_normalized):
        """
        The core cybernetic feedback mechanism.
        Adjusts the kernel coefficients based on the error between the actual observed
        value and the model's prediction, incorporating L2 regularization.

        Args:
            actual_value_normalized (float): The true observed value (normalized).
            predicted_value_normalized (float): The value that the model predicted (normalized).
            history_at_prediction_time_normalized (np.array): The historical data (normalized)
                                                               used to make the `predicted_value_normalized`.
        """
        error = actual_value_normalized - predicted_value_normalized

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


    def process_data_point(self, current_value):
        """
        Processes a new data point. This involves:
        1. Updating the historical data buffer.
        2. If sufficient history, making a new forecast.
        3. If a previous forecast exists, applying the cybernetic feedback
           to adapt the model's kernels.

        Args:
            current_value (float): The latest observed data point.

        Returns:
            tuple: (predicted_value, current_entropy)
                   - predicted_value (float): The forecast for the *next* period (unnormalized).
                   - current_entropy (float): Entropy of the current history window.
                   Returns (None, None) if not enough history to forecast yet.
        """
        # Store the current state of history *before* adding the new value.
        # This is the history that was used to make the `last_predicted_value`.
        history_for_feedback = np.copy(self.data_history)

        # Update the data history buffer (acts as a sliding window)
        if self.history_counter < self.history_length:
            self.data_history[self.history_counter] = current_value
            self.history_counter += 1
        else:
            self.data_history = np.roll(self.data_history, -1)
            self.data_history[-1] = current_value
            self.is_initialized_for_forecast = True

        predicted_value_for_next_period = None
        current_entropy = None

        # Normalize the history and current value for internal model use if enabled
        current_history_normalized = np.copy(self.data_history)
        current_value_normalized = current_value
        history_for_feedback_normalized = np.copy(history_for_feedback)
        last_predicted_value_normalized = self.last_predicted_value # This is already normalized if `normalize_data` was True for previous step

        if self.normalize_data and self.history_counter > 1:
            # Update rolling mean and std
            self.rolling_mean = np.mean(self.data_history[:self.history_counter])
            self.rolling_std = np.std(self.data_history[:self.history_counter])
            if self.rolling_std == 0:
                self.rolling_std = 1e-6 # Avoid division by zero

            current_history_normalized = (self.data_history - self.rolling_mean) / self.rolling_std
            current_value_normalized = (current_value - self.rolling_mean) / self.rolling_std
            history_for_feedback_normalized = (history_for_feedback - self.rolling_mean) / self.rolling_std


        if self.is_initialized_for_forecast:
            # 1. Cybernetic Feedback (Adaptation):
            # Use the `current_value` (actual outcome) and the `last_predicted_value`
            # (prediction made for this period) to adjust kernels.
            self._adaptive_kernel_adjustment(current_value_normalized, last_predicted_value_normalized, history_for_feedback_normalized)

            # 2. Forecast for the *next* period:
            # Use the *updated* kernels and the *latest* normalized history to make a new prediction.
            predicted_value_for_next_period_normalized = self._wiener_forecast(current_history_normalized)

            # Denormalize the prediction before returning
            predicted_value_for_next_period = predicted_value_for_next_period_normalized * self.rolling_std + self.rolling_mean

            # 3. Update last_predicted_value for the next feedback cycle (store normalized for consistency)
            self.last_predicted_value = predicted_value_for_next_period_normalized

            # 4. Calculate entropy of the current history window (on unnormalized data)
            current_entropy = self._calculate_entropy(self.data_history[:self.history_counter])

        return predicted_value_for_next_period, current_entropy

    def reset_state(self):
        """Resets the internal state of the forecaster for a new training/evaluation run."""
        self.kernels = self._initialize_kernels(self.order_wiener_expansion)
        self.data_history = np.zeros(self.history_length)
        self.last_predicted_value = 0.0
        self.is_initialized_for_forecast = False
        self.history_counter = 0
        self.rolling_mean = 0.0
        self.rolling_std = 1.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings # Import the warnings module
# For FRED data:
# pip install pandas_datareader
from pandas_datareader import data as pdr
# For AR(1) model:
# pip install statsmodels
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler # For AR(1) input scaling if needed

# --- Silence specific RuntimeWarning from nolds ---
warnings.filterwarnings("ignore", category=RuntimeWarning, module='nolds.measures')
# --- Silence specific ValueWarning from statsmodels ---
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels.tsa.base.tsa_model')

# --- Import CyberneticForecaster Class (assuming it's in the same environment or imported) ---
# If CyberneticForecaster is in a separate file, you'd import it like:
# from cybernetic_forecaster import CyberneticForecaster
# For this self-contained example, we'll assume it's available.
# (The previous immersive block defines it, so it's available in this context.)

# --- Evaluation Function ---
def rolling_forecast_evaluation(
    data_series: pd.Series,
    forecaster_params: dict,
    ar_order: tuple = (1, 0, 0), # AR(1) model order
    train_window_size: int = 120, # e.g., 10 years for monthly data
    test_window_size: int = 12,  # e.g., 1 year for monthly data
    step_size: int = 12,         # e.g., advance by 1 year
    initial_forecast_history: int = 20 # Minimum history needed for CyberneticForecaster
) -> dict:
    """
    Performs rolling window evaluation for CyberneticForecaster vs. AR(1) benchmark.

    Args:
        data_series (pd.Series): The stationary time series to forecast (e.g., inflation rate).
                                 Must be indexed by date/time.
        forecaster_params (dict): Parameters for CyberneticForecaster initialization.
        ar_order (tuple): ARIMA order for the AR(1) benchmark (p, d, q). Default (1,0,0) for AR(1).
        train_window_size (int): Number of data points in the training window.
        test_window_size (int): Number of data points in the testing window.
        step_size (int): How many data points to advance the window in each step.
        initial_forecast_history (int): Minimum data points required by CyberneticForecaster
                                        before it can make a forecast.

    Returns:
        dict: A dictionary containing evaluation results, including:
              - 'forecast_df': DataFrame with actual, Cybernetic, and AR(1) forecasts.
              - 'metrics_df': DataFrame with MSE, MAE, Sign Accuracy for each model per window.
    """
    if not isinstance(data_series, pd.Series):
        raise TypeError("data_series must be a pandas Series.")
    if data_series.empty:
        raise ValueError("data_series cannot be empty.")
    if train_window_size <= 0 or test_window_size <= 0 or step_size <= 0:
        raise ValueError("Window sizes and step size must be positive integers.")
    if train_window_size + test_window_size > len(data_series):
        raise ValueError("Total window size (train + test) exceeds data length.")
    if initial_forecast_history > train_window_size:
        raise ValueError("initial_forecast_history cannot be greater than train_window_size.")

    results = []
    all_forecasts = []

    print(f"Starting Rolling Forecast Evaluation for {data_series.name}.")
    print(f"Total data points: {len(data_series)}")
    print(f"Training Window: {train_window_size} periods, Testing Window: {test_window_size} periods, Step: {step_size} periods.")

    start_index = 0
    while start_index + train_window_size + test_window_size <= len(data_series):
        train_end_index = start_index + train_window_size
        test_end_index = train_end_index + test_window_size

        train_data = data_series.iloc[start_index:train_end_index]
        test_data = data_series.iloc[train_end_index:test_end_index]

        print(f"\n--- Evaluation Window: {train_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')} ---")
        print(f"  Training Period: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"  Testing Period:  {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")

        # --- Cybernetic Forecaster ---
        cyber_forecaster = CyberneticForecaster(**forecaster_params)
        cyber_forecasts = []
        actual_values_cyber = []
        cyber_entropy_values = []

        # Train and forecast with CyberneticForecaster
        # We need to feed data point by point to allow for adaptive learning
        # The first `initial_forecast_history` points are for initialization
        # The first prediction will be for the `initial_forecast_history + 1`-th point
        temp_history_for_cyber = pd.Series([], dtype=float) # Use a temporary series to build history
        for j in range(len(train_data) + len(test_data)):
            if j < len(train_data):
                current_point = train_data.iloc[j]
            else:
                current_point = test_data.iloc[j - len(train_data)]

            predicted_val, entropy_val = cyber_forecaster.process_data_point(current_point)
            cyber_entropy_values.append(entropy_val)

            # Collect forecasts only for the test period
            if j >= len(train_data) and cyber_forecaster.is_initialized_for_forecast:
                # The prediction `predicted_val` is for the *next* step.
                # So we collect it, and the actual value it corresponds to will be the *next* current_point.
                # This means the actual_values_cyber should be current_point from the *next* iteration
                # or the test_data[j - len(train_data)]
                # Let's adjust for one-step-ahead forecasting:
                # The `process_data_point` returns prediction for next period.
                # So, if we process data up to `t`, it predicts for `t+1`.
                # We need to store this prediction and compare it to the actual value at `t+1`.

                # To avoid forward-looking bias, the forecast for test_data[k] must be based on data up to test_data[k-1].
                # The `process_data_point` method already handles this by returning a forecast for the *next* period
                # after processing `current_value`.

                # The `predicted_val` returned by `process_data_point(current_point)` is the forecast for the point *after* `current_point`.
                # So, if `current_point` is `test_data.iloc[k]`, `predicted_val` is for `test_data.iloc[k+1]`.
                # This means we need to align carefully.

                # Let's collect forecasts when `cyber_forecaster.process_data_point` is called with a point *from the training data*
                # that results in a prediction for the *first point of the test data*.
                # And then continue this for the entire test data.

                # A simpler approach for rolling evaluation:
                # 1. Reset forecaster.
                # 2. Feed ALL train_data to forecaster (it adapts).
                # 3. For each point in test_data:
                #    a. Get forecast for test_data[i] based on history up to test_data[i-1] (which includes train_data and previous test_data points).
                #    b. Feed test_data[i] to forecaster (it adapts for next prediction).

                # Reset forecaster for this window's training
                cyber_forecaster.reset_state()
                # Feed training data to cyber forecaster to train its kernels
                for val in train_data:
                    cyber_forecaster.process_data_point(val) # Adapt on training data

                # Now, forecast for the test data
                current_cyber_forecasts = []
                current_actual_values = []
                for k in range(len(test_data)):
                    # Get the point that needs to be forecasted
                    point_to_forecast = test_data.iloc[k]

                    # The forecaster's `process_data_point` method will take the *previous* point
                    # and return a prediction for the *current* point.
                    # So, to predict `test_data.iloc[k]`, we need to call `process_data_point`
                    # with `test_data.iloc[k-1]` (or the last train data point if k=0).

                    # Let's modify the loop to predict *before* processing the actual value for feedback.
                    # This ensures no look-ahead bias.

                    # Predict for the current test point `test_data.iloc[k]`
                    # The `process_data_point` method returns the prediction for the *next* step
                    # after processing the current input.
                    # So, to get a prediction for `test_data.iloc[k]`, we need the prediction that
                    # was generated when `test_data.iloc[k-1]` (or last train value) was processed.

                    # To get a forecast for `test_data.iloc[k]`:
                    # 1. Call `process_data_point` with `test_data.iloc[k-1]` (or last train data for k=0).
                    # 2. The returned `predicted_value` is for `test_data.iloc[k]`.
                    # 3. Store this prediction.
                    # 4. Then, `process_data_point` with `test_data.iloc[k]` for next prediction and adaptation.

                    # Corrected logic for one-step-ahead forecasting in test window:
                    # After training, the cyber_forecaster is ready to predict the first test point.
                    # Its `last_predicted_value` is the prediction for the first test point.
                    if k == 0:
                        # After training, the last call to process_data_point was with train_data.iloc[-1]
                        # and its output `predicted_value_for_next_period` is the forecast for test_data.iloc[0].
                        forecast_for_current_test_point = cyber_forecaster.last_predicted_value * cyber_forecaster.rolling_std + cyber_forecaster.rolling_mean
                    else:
                        # For subsequent test points, the prediction for test_data.iloc[k]
                        # was generated when test_data.iloc[k-1] was processed.
                        forecast_for_current_test_point = cyber_forecaster.last_predicted_value * cyber_forecaster.rolling_std + cyber_forecaster.rolling_mean

                    current_cyber_forecasts.append(forecast_for_current_test_point)
                    current_actual_values.append(point_to_forecast)

                    # Now, feed the actual value for adaptation and to get prediction for next step
                    cyber_forecaster.process_data_point(point_to_forecast)


        # --- AR(1) Benchmark ---
        ar_forecasts = []
        # Fit AR(1) model on training data
        try:
            # ARIMA(1,0,0) is an AR(1) model
            ar_model = ARIMA(train_data, order=ar_order)
            ar_model_fit = ar_model.fit()

            # Forecast for the test period
            # Use `predict` for in-sample and out-of-sample forecasting
            # start=len(train_data) gives the first out-of-sample forecast
            # end=len(train_data) + len(test_data) - 1 gives the last out-of-sample forecast
            # dynamic=False means one-step-ahead forecasts (uses actual previous values if available)
            # For pure out-of-sample, we need to re-fit or use dynamic=True for multi-step.
            # For rolling one-step-ahead, we fit on train + previously observed test points.

            # To avoid look-ahead bias, we will re-fit the AR(1) model for each forecast in the test window
            # using the expanding window of data (train_data + already observed test_data).
            # This is computationally intensive but ensures fairness.
            current_ar_forecasts = []
            expanding_data_for_ar = train_data.copy()
            for k in range(len(test_data)):
                try:
                    ar_model_expanding = ARIMA(expanding_data_for_ar, order=ar_order)
                    ar_model_fit_expanding = ar_model_expanding.fit()
                    # Predict the next step (k+1 from current expanding_data_for_ar length)
                    forecast_ar = ar_model_fit_expanding.forecast(steps=1).iloc[0]
                    current_ar_forecasts.append(forecast_ar)
                except Exception as e:
                    # print(f"  AR(1) forecasting error at step {k}: {e}")
                    current_ar_forecasts.append(np.nan) # Append NaN on error

                # Add the actual value to the expanding data for the next iteration
                expanding_data_for_ar = pd.concat([expanding_data_for_ar, pd.Series([test_data.iloc[k]], index=[test_data.index[k]])])

            ar_forecasts = current_ar_forecasts

        except Exception as e:
            print(f"  AR(1) model fitting error: {e}")
            ar_forecasts = [np.nan] * len(test_data) # Fill with NaNs on error

        # Ensure all lists are of the same length for test_data
        min_len = min(len(current_actual_values), len(current_cyber_forecasts), len(ar_forecasts))
        if min_len == 0:
            print("  Warning: No valid forecasts generated for this window.")
            continue

        current_actual_values = current_actual_values[:min_len]
        current_cyber_forecasts = current_cyber_forecasts[:min_len]
        ar_forecasts = ar_forecasts[:min_len]

        # --- Evaluate Performance ---
        # Filter out NaNs for metric calculation
        valid_indices = ~np.isnan(current_cyber_forecasts) & ~np.isnan(ar_forecasts) & ~np.isnan(current_actual_values)
        if not np.any(valid_indices):
            print("  No valid data points for metric calculation in this window.")
            continue

        actual_filtered = np.array(current_actual_values)[valid_indices]
        cyber_forecasts_filtered = np.array(current_cyber_forecasts)[valid_indices]
        ar_forecasts_filtered = np.array(ar_forecasts)[valid_indices]

        # Mean Squared Error (MSE)
        mse_cyber = mean_squared_error(actual_filtered, cyber_forecasts_filtered)
        mse_ar = mean_squared_error(actual_filtered, ar_forecasts_filtered)

        # Mean Absolute Error (MAE)
        mae_cyber = mean_absolute_error(actual_filtered, cyber_forecasts_filtered)
        mae_ar = mean_absolute_error(actual_filtered, ar_forecasts_filtered)

        # Sign Prediction Accuracy
        sign_accuracy_cyber = np.mean(np.sign(actual_filtered) == np.sign(cyber_forecasts_filtered))
        sign_accuracy_ar = np.mean(np.sign(actual_filtered) == np.sign(ar_forecasts_filtered))

        # Store results for this window
        results.append({
            'train_start': train_data.index[0],
            'train_end': train_data.index[-1],
            'test_start': test_data.index[0],
            'test_end': test_data.index[-1],
            'mse_cyber': mse_cyber,
            'mae_cyber': mae_cyber,
            'sign_acc_cyber': sign_accuracy_cyber,
            'mse_ar': mse_ar,
            'mae_ar': mae_ar,
            'sign_acc_ar': sign_accuracy_ar,
        })

        # Store forecasts for overall plotting
        for k in range(min_len):
            all_forecasts.append({
                'date': test_data.index[k],
                'actual': current_actual_values[k],
                'cyber_forecast': current_cyber_forecasts[k],
                'ar_forecast': ar_forecasts[k]
            })

        start_index += step_size

    metrics_df = pd.DataFrame(results)
    forecast_df = pd.DataFrame(all_forecasts).set_index('date')

    print("\n--- Rolling Forecast Evaluation Summary ---")
    print("Average Metrics Across Test Windows:")
    print(metrics_df[['mse_cyber', 'mae_cyber', 'sign_acc_cyber',
                      'mse_ar', 'mae_ar', 'sign_acc_ar']].mean())

    return {'forecast_df': forecast_df, 'metrics_df': metrics_df}


# --- Main Execution Block ---
if __name__ == "__main__":
    # 1. Download US CPI data from FRED
    print("\n--- Downloading US CPI Data from FRED ---")
    try:
        # CPIAUCSL: Consumer Price Index for All Urban Consumers: All Items in U.S. City Average, Seasonally Adjusted
        cpi_data = pdr.DataReader('CPIAUCSL', 'fred', start='2000-01-01', end='2025-05-31')
        cpi_data = cpi_data.dropna()
        print(f"Downloaded CPI data from {cpi_data.index.min().strftime('%Y-%m-%d')} to {cpi_data.index.max().strftime('%Y-%m-%d')}")

        # Calculate Monthly Inflation Rate (Monthly-over-Monthly Percentage Change)
        # (Current CPI / CPI 1 months ago - 1) * 100
        inflation_rate = cpi_data['CPIAUCSL'].pct_change(periods=1).dropna() * 100
        inflation_rate.name = 'Monthly US Inflation Rate'
        print(f"Calculated inflation rate from {inflation_rate.index.min().strftime('%Y-%m-%d')} to {inflation_rate.index.max().strftime('%Y-%m-%d')}")
        print("Inflation Rate Head:\n", inflation_rate.head())
        print("Inflation Rate Tail:\n", inflation_rate.tail())

    except Exception as e:
        print(f"Error downloading or processing FRED data: {e}")
        print("Please ensure 'pandas_datareader' and 'statsmodels' are installed.")
        exit()

    # 2. Define Cybernetic Forecaster Parameters
    cyber_forecaster_params = {
        'history_length': 36, # 3 years of monthly data
        'learning_rate': 0.0001,
        'order_wiener_expansion': 3,
        'regularization_strength': 0.0001,
        'normalize_data': True
    }

    # 3. Define Rolling Window Settings
    # For monthly data:
    # Train on 3 years (36 months), test on 1 year (12 months), step by 1 year (12 months)
    train_window_months = 36
    test_window_months = 12
    step_window_months = 12

    # 4. Run Rolling Forecast Evaluation
    evaluation_results = rolling_forecast_evaluation(
        data_series=inflation_rate,
        forecaster_params=cyber_forecaster_params,
        ar_order=(1, 0, 0), # AR(1)
        train_window_size=train_window_months,
        test_window_size=test_window_months,
        step_size=step_window_months,
        initial_forecast_history=cyber_forecaster_params['history_length']
    )

    forecast_df = evaluation_results['forecast_df']
    metrics_df = evaluation_results['metrics_df']

    # 5. Plotting Results
    if not forecast_df.empty:
        plt.figure(figsize=(15, 8))
        plt.plot(forecast_df.index, forecast_df['actual'], label='Actual Inflation', color='black', linewidth=2)
        plt.plot(forecast_df.index, forecast_df['cyber_forecast'], label='Cybernetic Forecaster', color='blue', linestyle='--')
        plt.plot(forecast_df.index, forecast_df['ar_forecast'], label='AR(1) Benchmark', color='red', linestyle=':')
        plt.title('US Monthly Inflation Rate: Actual vs. Forecasts (Rolling Windows)')
        plt.xlabel('Date')
        plt.ylabel('Annual Inflation Rate (%)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    else:
        print("No forecasts were generated for plotting.")

    print("\n--- Detailed Metrics per Test Window ---")
    print(metrics_df.to_string()) # Use to_string() to display full DataFrame
