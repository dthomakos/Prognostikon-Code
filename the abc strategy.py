#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-abc-of-speculation/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
#

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from itertools import product

class The_ABC_Trader:
    def __init__(self, ticker, start_date, end_date, frequency='daily'):
        """
        Initialize the ABC Trader class.

        Parameters:
        - ticker: The ticker symbol of the ETF or asset (e.g., 'SPY' for S&P 500 ETF).
        - start_date: The start date for fetching historical data (format: 'YYYY-MM-DD').
        - end_date: The end date for fetching historical data (format: 'YYYY-MM-DD').
        - frequency: The frequency of returns ('daily', 'weekly', or 'monthly').
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.frequency = frequency
        self.returns_df = self._download_data()

    def _download_data(self):
        """
        Download historical price data and calculate returns based on the specified frequency.

        Returns:
        - A DataFrame with returns.
        """
        print(f"Downloading data for {self.ticker} from {self.start_date} to {self.end_date}...")

        # Download historical price data using yfinance with auto_adjust=False
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=False)

        if self.frequency == 'daily':
            # Calculate daily returns using 'Close' column
            data['Returns'] = data['Close'].pct_change()
        elif self.frequency == 'weekly':
            # Calculate weekly returns
            data['Returns'] = data['Close'].resample('W-FRI').last().pct_change()
        elif self.frequency == 'monthly':
            # Calculate monthly returns
            data['Returns'] = data['Close'].resample('ME').last().pct_change()

        # Drop NaN values (first row will be NaN due to pct_change)
        data.dropna(inplace=True)  # Drop all NaN rows

        # Keep only the 'Returns' column
        returns_df = data[['Returns']]

        print("Data downloaded successfully!")

        return returns_df

    def apply_trading_strategy(self, alpha, beta, gamma, delay, ewma_span=None):
        """
        Apply a parametrized trading strategy to the returns DataFrame with a delay in signals.

        Parameters:
        - alpha: Number of past returns to check for buying.
        - beta: Number of past returns to check for selling.
        - gamma: Number of past returns to check for buying again after a sell.
        - delay: Number of periods to delay the signals.
        - ewma_span: Span for exponentially weighted moving average (EWMA).

        Returns:
        - A DataFrame with signals for buying and selling.
        """
        if ewma_span is not None:
            # Calculate EWMA returns
            ewma_returns = self.returns_df['Returns'].ewm(span=ewma_span, adjust=False).mean()
            returns_to_use = ewma_returns
        else:
            returns_to_use = self.returns_df['Returns']

        signals = pd.DataFrame(index=self.returns_df.index, columns=['Signal'])

        # Initialize signals to zero (no action)
        signals['Signal'] = 0

        # Iterate over the returns to generate signals
        for i in range(len(self.returns_df)):
            if i < max(alpha, beta, gamma):  # Not enough history to apply the strategy
                continue

            # Check past alpha returns
            past_alpha_returns = returns_to_use.iloc[i-alpha:i].values

            # Check past beta returns
            past_beta_returns = returns_to_use.iloc[i-beta:i].values

            # Check past gamma returns
            past_gamma_returns = returns_to_use.iloc[i-gamma:i].values

            # Apply the strategy
            if np.all(past_alpha_returns > 0):  # Past alpha returns are positive
                signals.loc[signals.index[i], 'Signal'] = 1  # Buy signal
            elif np.all(past_beta_returns > 0):  # Past beta returns are positive
                signals.loc[signals.index[i], 'Signal'] = -1  # Sell signal
            elif np.all(past_alpha_returns < 0) and np.any(past_gamma_returns > 0):  # Past alpha negative and past gamma positive
                signals.loc[signals.index[i], 'Signal'] = 1  # Buy signal again

        # Apply delay to signals
        signals['Signal'] = signals['Signal'].shift(delay).fillna(0)

        return signals

    def calculate_strategy_performance(self, signals):
        """
        Calculate the performance of the trading strategy.

        Parameters:
        - signals: DataFrame with buy/sell signals.

        Returns:
        - A DataFrame with strategy performance metrics.
        """
        # Calculate strategy returns based on signals
        strategy_returns = self.returns_df['Returns'] * signals['Signal']

        # Calculate cumulative strategy returns
        cumulative_strategy_returns = (1 + strategy_returns).cumprod()

        # Calculate buy-and-hold returns
        buy_and_hold_returns = (1 + self.returns_df['Returns']).cumprod()

        # Calculate maximum drawdown
        def calculate_drawdown(cumulative_returns):
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = drawdown.min()
            return max_drawdown

        strategy_max_drawdown = calculate_drawdown(cumulative_strategy_returns)
        buy_and_hold_max_drawdown = calculate_drawdown(buy_and_hold_returns)

        # Performance metrics
        if self.frequency == 'daily':
            scaling_factor = np.sqrt(252)
            annualization_factor = 252
        elif self.frequency == 'weekly':
            scaling_factor = np.sqrt(52)
            annualization_factor = 52
        elif self.frequency == 'monthly':
            scaling_factor = np.sqrt(12)
            annualization_factor = 12

        strategy_mean_return_annualized = strategy_returns.mean() * annualization_factor
        strategy_volatility_annualized = strategy_returns.std() * scaling_factor
        buy_and_hold_mean_return_annualized = self.returns_df['Returns'].mean() * annualization_factor
        buy_and_hold_volatility_annualized = self.returns_df['Returns'].std() * scaling_factor

        performance_metrics = {
            'Cumulative Strategy Returns': cumulative_strategy_returns.iloc[-1] - 1,
            'Cumulative Buy-and-Hold Returns': buy_and_hold_returns.iloc[-1] - 1,
            'Strategy Mean Return (Annualized)': strategy_mean_return_annualized,
            'Strategy Volatility (Annualized)': strategy_volatility_annualized,
            'Strategy Sharpe Ratio': strategy_mean_return_annualized / strategy_volatility_annualized if strategy_volatility_annualized != 0 else 0,
            'Buy-and-Hold Mean Return (Annualized)': buy_and_hold_mean_return_annualized,
            'Buy-and-Hold Volatility (Annualized)': buy_and_hold_volatility_annualized,
            'Buy-and-Hold Sharpe Ratio': buy_and_hold_mean_return_annualized / buy_and_hold_volatility_annualized if buy_and_hold_volatility_annualized != 0 else 0,
            'Strategy Max Drawdown': strategy_max_drawdown,
            'Buy-and-Hold Max Drawdown': buy_and_hold_max_drawdown,
        }

        return pd.DataFrame(performance_metrics, index=['Performance'])

    def plot_cumulative_returns(self, signals):
        """
        Plot the cumulative returns of the strategy and the buy-and-hold approach.

        Parameters:
        - signals: DataFrame with buy/sell signals.
        """
        # Calculate strategy returns based on signals
        strategy_returns = self.returns_df['Returns'] * signals['Signal']

        # Calculate cumulative strategy returns
        cumulative_strategy_returns = (1 + strategy_returns).cumprod()

        # Calculate buy-and-hold returns
        buy_and_hold_returns = (1 + self.returns_df['Returns']).cumprod()

        # Plot cumulative returns
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_strategy_returns-1, label='Strategy Returns', color='blue')
        plt.plot(buy_and_hold_returns-1, label='Buy-and-Hold Returns', color='red')
        plt.title('Cumulative Returns for '+self.ticker)
        plt.xlabel('Time')
        plt.ylabel('Cumulative Returns, in decimal')
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()

    def optimize_strategy(self, alpha_range, beta_range, gamma_range, delay_range, ewma_span_range=None):
        """
        Optimize the trading strategy by finding the best combination of alpha, beta, gamma, delay, and EWMA span.

        Parameters:
        - alpha_range: Range of values for alpha.
        - beta_range: Range of values for beta.
        - gamma_range: Range of values for gamma.
        - delay_range: Range of values for delay.
        - ewma_span_range: Range of values for EWMA span (optional).

        Returns:
        - The best combination of parameters based on cumulative strategy returns.
        """
        best_performance = -np.inf
        best_params = None

        if ewma_span_range is None:
            # Without EWMA
            for alpha, beta, gamma, delay in product(alpha_range, beta_range, gamma_range, delay_range):
                signals = self.apply_trading_strategy(alpha, beta, gamma, delay)
                performance = self.calculate_strategy_performance(signals)
                cumulative_returns = performance.loc['Performance', 'Cumulative Strategy Returns']

                if cumulative_returns > best_performance:
                    best_performance = cumulative_returns
                    best_params = (alpha, beta, gamma, delay)
        else:
            # With EWMA
            for alpha, beta, gamma, delay, ewma_span in product(alpha_range, beta_range, gamma_range, delay_range, ewma_span_range):
                signals = self.apply_trading_strategy(alpha, beta, gamma, delay, ewma_span=ewma_span)
                performance = self.calculate_strategy_performance(signals)
                cumulative_returns = performance.loc['Performance', 'Cumulative Strategy Returns']

                if cumulative_returns > best_performance:
                    best_performance = cumulative_returns
                    best_params = (alpha, beta, gamma, delay, ewma_span)

        return best_params, best_performance

    def calculate_expected_return_and_volatility(self, signals):
        """
        Calculate theoretical expected return and volatility of the strategy.

        Parameters:
        - signals: DataFrame with buy/sell/re-entry signals.

        Returns:
        - A dictionary containing expected return and volatility.
        """
        # Extract returns associated with each type of signal
        buy_returns = self.returns_df['Returns'][signals['Signal'] == 1]
        sell_returns = self.returns_df['Returns'][signals['Signal'] == -1]
        reentry_returns = self.returns_df['Returns'][(signals['Signal'] == 1) & (signals['Signal'].shift(1) == -1)]

        # Probabilities of each type of signal
        total_signals = len(signals)
        p_buy = len(buy_returns) / total_signals
        p_sell = len(sell_returns) / total_signals
        p_reentry = len(reentry_returns) / total_signals

        # Expected returns for each type of signal
        r_buy = buy_returns.mean() if len(buy_returns) > 0 else 0
        r_sell = sell_returns.mean() if len(sell_returns) > 0 else 0
        r_reentry = reentry_returns.mean() if len(reentry_returns) > 0 else 0

        # Variances for each type of signal
        sigma_buy = buy_returns.var() if len(buy_returns) > 0 else 0
        sigma_sell = sell_returns.var() if len(sell_returns) > 0 else 0
        sigma_reentry = reentry_returns.var() if len(reentry_returns) > 0 else 0

        # Covariances between signal types (approximated as zero for simplicity)
        cov_buy_sell = cov_buy_reentry = cov_sell_reentry = 0

        # Expected Return
        expected_return = p_buy * r_buy + p_sell * r_sell + p_reentry * r_reentry

        # Variance and Volatility
        variance = (
            p_buy * sigma_buy +
            p_sell * sigma_sell +
            p_reentry * sigma_reentry +
            2 * (p_buy * p_sell * cov_buy_sell +
                 p_buy * p_reentry * cov_buy_reentry +
                 p_sell * p_reentry * cov_sell_reentry)
        )

        volatility = np.sqrt(variance)

        # Frequency scaling factor (annualization)
        if self.frequency == 'daily':
            scaling_factor = 252
        elif self.frequency == 'weekly':
            scaling_factor = 52
        elif self.frequency == 'monthly':
            scaling_factor = 12
        else:
            scaling_factor = 1

        # Annualized Expected Return and Volatility
        expected_return_annualized = expected_return * scaling_factor
        volatility_annualized = volatility * np.sqrt(scaling_factor)

        # Theoretical Sharpe Ratio (annualized)
        sharpe_ratio = (expected_return_annualized / volatility_annualized) if volatility_annualized != 0 else 0

        return {
            'Expected Return (Annualized)': expected_return * scaling_factor,
            'Volatility (Annualized)': volatility * scaling_factor,
            'Theoretical Sharpe Ratio': sharpe_ratio
        }

# Example Usage
if __name__ == "__main__":
    # Parameters for downloading ETF data and applying the strategy
    ticker = "SPY"  # S&P 500 ETF as an example
    start_date = "2000-01-01"
    end_date = "2025-03-31"
    frequency = 'monthly'

    # Create trader instance and fetch data
    trader = The_ABC_Trader(ticker, start_date, end_date, frequency=frequency)

    # Define ranges for parameters
    alpha_range = range(2, 5)  # Example range for alpha
    beta_range = range(2, 5)    # Example range for beta
    gamma_range = range(2, 5)  # Example range for gamma
    delay_range = range(1, 2)    # Example range for delay
    ewma_span_range = range(1, 3)  # Example range for EWMA span

    # Optimize the strategy
    if ewma_span_range is not None:
        best_params, best_performance = trader.optimize_strategy(alpha_range, beta_range, gamma_range, delay_range, ewma_span_range=ewma_span_range)

        print(f"Best Parameters: alpha={best_params[0]}, beta={best_params[1]}, gamma={best_params[2]}, delay={best_params[3]}, ewma_span={best_params[4]}")
        print(f"Best Cumulative Returns: {best_performance}")

        # Apply the best strategy
        signals = trader.apply_trading_strategy(*best_params[:4], ewma_span=best_params[4])
    else:
        best_params, best_performance = trader.optimize_strategy(alpha_range, beta_range, gamma_range, delay_range, ewma_span_range=ewma_span_range)

        print(f"Best Parameters: alpha={best_params[0]}, beta={best_params[1]}, gamma={best_params[2]}, delay={best_params[3]}")
        print(f"Best Cumulative Returns: {best_performance}")

        # Apply the best strategy
        signals = trader.apply_trading_strategy(*best_params[:4], ewma_span=None)
    performance = trader.calculate_strategy_performance(signals)

    print("\nStrategy Performance with Best Parameters:")
    print(performance.T)

    # Plot cumulative returns
    trader.plot_cumulative_returns(signals)

    # Calculate theoretical expected return and volatility
    stats = trader.calculate_expected_return_and_volatility(signals)
    print('\n')
    print("Theoretical Expected Return:", round(stats['Expected Return (Annualized)'],6))
    print("Theoretical Volatility:", round(stats['Volatility (Annualized)'], 6))
    print("Theoretical Sharpe Ratio:", round(stats['Theoretical Sharpe Ratio'], 6))