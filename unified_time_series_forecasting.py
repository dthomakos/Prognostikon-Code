import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from tabulate import tabulate
from statsmodels.multivariate.manova import MANOVA
from scipy.stats import kstest, normaltest, anderson, expon, gamma, beta
from statsmodels.tsa.ar_model import AutoReg
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                            mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from scipy.linalg import pinv
from itertools import combinations
from typing import List, Optional, Dict, Tuple

# --- 1. Plotting Classes ---

class TimeSeriesPlotter:
    """
    A class for plotting various types of time series visualizations.
    """
    def __init__(self, data: pd.DataFrame, title: str = "Time Series Plot",
                 xlabel: str = "Time", ylabel: str = "Value"):
        """
        Initialize the TimeSeriesPlotter class.

        Parameters:
        - data: DataFrame containing time series data with a 'time' column.
        - title: The title of the plot.
        - xlabel: The label for the x-axis.
        - ylabel: The label for the y-axis.
        """
        if 'time' not in data.columns:
            raise ValueError("DataFrame must contain a 'time' column.")
        self.data = data
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot_single_series(self, series_name: str, secondary_series: Optional[str] = None,
                           secondary_ylabel: Optional[str] = None):
        """
        Plot a single time series from a DataFrame, with an optional secondary y-axis.

        Parameters:
        - series_name: The name of the column in the DataFrame to plot.
        - secondary_series: Optional secondary series to plot on the right axis.
        - secondary_ylabel: Label for the secondary y-axis.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.data['time'], self.data[series_name], marker='o', color='blue')
        ax1.set_title(self.title)
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel(self.ylabel, color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.grid(True)

        if secondary_series:
            ax2 = ax1.twinx()
            ax2.plot(self.data['time'], self.data[secondary_series], marker='o', color='red')
            ax2.set_ylabel(secondary_ylabel, color='red')
            ax2.tick_params(axis='y', labelcolor='red')

        plt.show()

    def plot_multiple_series(self, series_names: Optional[List[str]] = None,
                             secondary_series: Optional[str] = None,
                             secondary_ylabel: Optional[str] = None):
        """
        Plot multiple time series from the DataFrame.

        Parameters:
        - series_names: List of column names to plot. If None, all numeric columns except 'time' are plotted.
        - secondary_series: Optional secondary series to plot on the right axis.
        - secondary_ylabel: Label for the secondary y-axis.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        cols_to_plot = series_names if series_names is not None else [col for col in self.data.columns if col != 'time' and pd.api.types.is_numeric_dtype(self.data[col])]

        for column in cols_to_plot:
            ax1.plot(self.data['time'], self.data[column], label=column)
        ax1.set_title(self.title)
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel(self.ylabel)
        ax1.legend(loc='upper left')
        ax1.grid(True)

        if secondary_series:
            ax2 = ax1.twinx()
            ax2.plot(self.data['time'], self.data[secondary_series], label=secondary_series, color='red')
            ax2.set_ylabel(secondary_ylabel, color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            ax2.legend(loc='upper right')

        plt.show()

    def customize_plot(self, series_name: str, color: str = 'blue', marker: str = 'o',
                       linestyle: str = '-', secondary_series: Optional[str] = None,
                       secondary_ylabel: Optional[str] = None):
        """
        Customize the plot for a single time series.

        Parameters:
        - series_name: The name of the column in the DataFrame to plot.
        - color: The color of the line.
        - marker: The marker style.
        - linestyle: The line style.
        - secondary_series: Optional secondary series to plot on the right axis.
        - secondary_ylabel: Label for the secondary y-axis.
        """
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.data['time'], self.data[series_name], color=color, marker=marker, linestyle=linestyle)
        ax1.set_title(self.title)
        ax1.set_xlabel(self.xlabel)
        ax1.set_ylabel(self.ylabel, color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True)

        if secondary_series:
            ax2 = ax1.twinx()
            ax2.plot(self.data['time'], self.data[secondary_series], color='red', marker='o')
            ax2.set_ylabel(secondary_ylabel, color='red')
            ax2.tick_params(axis='y', labelcolor='red')

        plt.show()

    def plot_with_shading(self, series_name: str, recession_periods: List[Tuple[float, float]],
                           recession_colors: List[str]):
        """
        Plot a time series with multiple shaded regions for different types of recessions.

        Parameters:
        - series_name: The name of the column in the DataFrame to plot.
        - recession_periods: List of tuples, where each tuple contains the start and end times of a recession period.
                             Example: [(start1, end1), (start2, end2)]
        - recession_colors: List of colors corresponding to each recession period.
                            Example: ['gray', 'red', 'blue']
        """
        if len(recession_periods) != len(recession_colors):
            raise ValueError("The number of recession periods must match the number of colors.")

        plt.figure(figsize=(12, 8))

        # Plot the time series
        plt.plot(self.data['time'], self.data[series_name], marker='o', color='black', label=series_name)

        # Apply shading for each recession period
        for (start, end), color in zip(recession_periods, recession_colors):
            plt.fill_between(
                self.data['time'],
                self.data[series_name].min(), # Shade from min to max of the series
                self.data[series_name].max(),
                where=(self.data['time'] >= start) & (self.data['time'] <= end),
                color=color,
                alpha=0.3,
                label=f"Recession ({start}-{end})"
            )

        # Add labels and title
        plt.title(self.title)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)

        # Add legend and grid
        plt.legend(loc="upper left")
        plt.grid(True)

        plt.show()

    def plot_rolling_statistics(self, series_name: str, window_size: int = 12):
        """
        Plot rolling statistics: mean, standard deviation, skewness, and kurtosis.

        Parameters:
        - series_name: The name of the column in the DataFrame to plot.
        - window_size: The size of the rolling window.
        """
        rolling_mean = self.data[series_name].rolling(window_size).mean()
        rolling_std = self.data[series_name].rolling(window_size).std()
        rolling_skew = self.data[series_name].rolling(window_size).skew() # Use pandas built-in skew
        rolling_kurt = self.data[series_name].rolling(window_size).kurt() # Use pandas built-in kurt

        fig, axs = plt.subplots(4, 1, figsize=(12, 12))

        axs[0].plot(self.data['time'], rolling_mean)
        axs[0].set_title(f"Rolling Mean ({window_size} periods)")
        axs[0].set_ylabel("Mean")
        axs[0].grid(True)

        axs[1].plot(self.data['time'], rolling_std)
        axs[1].set_title(f"Rolling Standard Deviation ({window_size} periods)")
        axs[1].set_ylabel("Std. Dev.")
        axs[1].grid(True)

        axs[2].plot(self.data['time'], rolling_skew)
        axs[2].set_title(f"Rolling Skewness ({window_size} periods)")
        axs[2].set_ylabel("Skewness")
        axs[2].grid(True)

        axs[3].plot(self.data['time'], rolling_kurt)
        axs[3].set_title(f"Rolling Kurtosis ({window_size} periods)")
        axs[3].set_ylabel("Kurtosis")
        axs[3].set_xlabel(self.xlabel)
        axs[3].grid(True)

        fig.tight_layout()
        plt.show()

    def plot_rolling_correlation(self, window_size: int = 12):
        """
        Plot rolling correlations among all pairs of columns in the DataFrame (excluding 'time').

        Parameters:
        - window_size: The size of the rolling window.
        """
        numeric_cols = [col for col in self.data.columns if col != 'time' and pd.api.types.is_numeric_dtype(self.data[col])]
        col_pairs = list(combinations(numeric_cols, 2))

        if not col_pairs:
            print("No numeric column pairs found for rolling correlation.")
            return

        num_pairs = len(col_pairs)
        fig, axs = plt.subplots(num_pairs, 1, figsize=(12, 4 * num_pairs), sharex=True)

        # Ensure axs is iterable even if there's only one subplot
        if num_pairs == 1:
            axs = [axs]

        for ax, (col1, col2) in zip(axs, col_pairs):
            rolling_corr = (
                self.data[col1]
                .rolling(window=window_size)
                .corr(self.data[col2])
            )

            ax.plot(self.data['time'], rolling_corr, label=f"{col1} vs {col2}", color="blue")
            ax.set_title(f"Rolling Correlation ({window_size} periods): {col1} vs {col2}")
            ax.set_ylabel("Correlation")
            ax.grid(True)
            ax.legend(loc="upper left")

        plt.xlabel(self.xlabel)
        fig.tight_layout()
        plt.show()


class TimeSeriesWithHistogram:
    """
    A class for plotting a time series along with its histogram and empirical density function.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def plot(self, series_name: str, bins: int = 30, kde: bool = True, title: Optional[str] = None,
             xlabel: Optional[str] = None, ylabel: Optional[str] = None,
             background_color: str = 'white', save_as: Optional[str] = None):
        """
        Plot a time series along with its histogram and empirical density function.

        Parameters:
        - series_name: Name of the series to plot.
        - bins: Number of bins for the histogram (default is 30).
        - kde: Whether to plot the Kernel Density Estimate (default is True).
        - title: Title of the plot (default is None).
        - xlabel: Label for the x-axis (default is None).
        - ylabel: Label for the y-axis (default is None).
        - background_color: Background color of the plot (default is 'white').
        - save_as: File name to save the plot (default is None, options are 'png' or 'pdf').
        """
        if series_name not in self.df.columns:
            raise ValueError(f"Series '{series_name}' not found in DataFrame.")

        series = self.df[series_name]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        plt.style.use('ggplot')

        # Plot time series
        axes[0].plot(self.df.index, series, label=series_name, color='blue')
        axes[0].set_title(f'Time Series: {series_name}')
        if xlabel:
            axes[0].set_xlabel(xlabel)
        if ylabel:
            axes[0].set_ylabel(ylabel)
        axes[0].grid(True)
        axes[0].legend()
        axes[0].set_facecolor(background_color)

        # Plot histogram and KDE
        sns.histplot(series, bins=bins, kde=kde, ax=axes[1], color='blue')
        axes[1].set_title(f'Histogram and Empirical Density Function: {series_name}')
        if xlabel:
            axes[1].set_xlabel('Ordered observations')
        if ylabel:
            axes[1].set_ylabel(ylabel)
        axes[1].grid(True)
        axes[1].set_facecolor(background_color)

        plt.tight_layout()

        if title:
            fig.suptitle(title, fontsize=16)
            fig.subplots_adjust(top=0.9)

        if save_as:
            plt.savefig(f'plot.{save_as}', format=save_as)

        plt.show()


# --- 2. Descriptive Statistics & PCA Classes ---

class DataAnalyzer:
    """
    A class for performing descriptive statistics, correlation analysis, PCA, and MANOVA.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def descriptive_statistics(self, save_as: Optional[str] = None):
        """
        Generate descriptive statistics for the dataframe.

        Parameters:
        - save_as: File name to save the descriptive statistics table (default is None, options are 'csv' or 'xlsx').
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric columns found for descriptive statistics.")
            return None

        desc_stats = numeric_df.describe().T
        desc_stats['skew'] = numeric_df.skew()
        desc_stats['kurtosis'] = numeric_df.kurtosis()

        if save_as:
            if save_as == 'csv':
                desc_stats.to_csv('descriptive_statistics.csv')
                print("Descriptive statistics saved to descriptive_statistics.csv")
            elif save_as == 'xlsx':
                desc_stats.to_excel('descriptive_statistics.xlsx')
                print("Descriptive statistics saved to descriptive_statistics.xlsx")
            else:
                print("Invalid save_as format. Use 'csv' or 'xlsx'.")

        print("\nDescriptive Statistics:")
        print(tabulate(desc_stats, headers='keys', tablefmt='grid'))
        return desc_stats

    def correlation_matrix(self, save_as: Optional[str] = None):
        """
        Generate a correlation matrix for the dataframe and display it as a heatmap.

        Parameters:
        - save_as: File name to save the correlation matrix plot (default is None, options are 'png' or 'pdf').
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric columns found for correlation matrix.")
            return None

        corr_matrix = numeric_df.corr()

        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
        plt.title('Correlation Matrix Heatmap')
        plt.tight_layout()

        if save_as:
            if save_as in ['png', 'pdf']:
                plt.savefig(f'correlation_matrix.{save_as}', format=save_as)
                print(f"Correlation matrix plot saved to correlation_matrix.{save_as}")
            else:
                print("Invalid save_as format. Use 'png' or 'pdf'.")
        plt.show()
        return corr_matrix

    def principal_component_analysis(self, n_components: Optional[int] = None, save_as: Optional[str] = None):
        """
        Perform Principal Component Analysis (PCA) on the dataframe and plot explained variance.

        Parameters:
        - n_components: Number of principal components to keep (default is None, which keeps all components).
        - save_as: File name to save the PCA plots (default is None, options are 'png' or 'pdf').
        """
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric columns found for PCA.")
            return None
        if numeric_df.shape[1] < 2:
            print("At least two numeric columns are required for PCA.")
            return None

        pca = PCA(n_components=n_components)
        pca.fit(numeric_df)
        explained_variance = pca.explained_variance_ratio_
        components = pca.components_

        pca_df = pd.DataFrame(components, columns=numeric_df.columns)
        pca_df.index = [f'PC{i+1}' for i in range(pca_df.shape[0])]

        print("\nPrincipal Components:")
        print(tabulate(pca_df, headers='keys', tablefmt='grid'))

        # Scree plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        if save_as:
            if save_as in ['png', 'pdf']:
                plt.savefig(f'scree_plot.{save_as}', format=save_as)
                print(f"Scree plot saved to scree_plot.{save_as}")
            else:
                print("Invalid save_as format for scree plot. Use 'png' or 'pdf'.")
        plt.show()

        # Bar plot of explained variance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(explained_variance) + 1), explained_variance)
        plt.title('Explained Variance by Principal Components')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.grid(True)
        if save_as:
            if save_as in ['png', 'pdf']:
                plt.savefig(f'explained_variance.{save_as}', format=save_as)
                print(f"Explained variance plot saved to explained_variance.{save_as}")
            else:
                print("Invalid save_as format for explained variance plot. Use 'png' or 'pdf'.")
        plt.show()

        return pca_df

    def manova(self, dependent_vars: List[str], independent_var: str, save_as: Optional[str] = None):
        """
        Perform Multivariate Analysis of Variance (MANOVA) on the dataframe.

        Parameters:
        - dependent_vars: List of dependent variable names.
        - independent_var: Name of the independent variable.
        - save_as: File name to save the MANOVA results (default is None, options are 'csv' or 'txt').
                   Note: 'xlsx' is not directly supported for MANOVA summary text.
        """
        for var in dependent_vars + [independent_var]:
            if var not in self.df.columns:
                raise ValueError(f"Variable '{var}' not found in DataFrame.")

        formula = f"{' + '.join(dependent_vars)} ~ {independent_var}"
        manova = MANOVA.from_formula(formula, data=self.df)
        manova_results = manova.mv_test()

        if save_as:
            if save_as == 'csv':
                with open('manova_results.csv', 'w') as f:
                    f.write(manova_results.summary().as_text())
                print("MANOVA results saved to manova_results.csv")
            elif save_as == 'txt':
                with open('manova_results.txt', 'w') as f:
                    f.write(manova_results.summary().as_text())
                print("MANOVA results saved to manova_results.txt")
            else:
                print("Invalid save_as format. Use 'csv' or 'txt'.")

        print("\nMANOVA Results:")
        print(manova_results)
        return manova_results


class TimeSeriesBootstrap:
    """
    A class for performing time series bootstrapping and testing distributional fit.
    """
    def __init__(self, data: np.ndarray, num_samples: int, block_size: int):
        """
        Initialize the TimeSeriesBootstrap class.

        Parameters:
        - data: The original time series data (1D numpy array).
        - num_samples: The number of bootstrap samples to generate.
        - block_size: The size of blocks to use for block bootstrapping.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 1:
            raise ValueError("Data must be a 1D numpy array.")
        if num_samples <= 0 or block_size <= 0:
            raise ValueError("num_samples and block_size must be positive integers.")
        if block_size > len(data):
            raise ValueError("block_size cannot be greater than the length of the data.")

        self.data = data
        self.num_samples = num_samples
        self.block_size = block_size
        self.bootstrap_samples = self._generate_bootstrap_samples()

    def _generate_bootstrap_samples(self) -> np.ndarray:
        """
        Generates bootstrap samples using block bootstrapping.
        """
        n = len(self.data)
        num_blocks = int(np.ceil(n / self.block_size))

        bootstrap_samples = np.zeros((self.num_samples, n))

        for i in range(self.num_samples):
            sample = []
            for _ in range(num_blocks):
                # Ensure start_idx doesn't go out of bounds
                start_idx = np.random.randint(0, n - self.block_size + 1) if n - self.block_size + 1 > 0 else 0
                sample.extend(self.data[start_idx:start_idx + self.block_size])
            bootstrap_samples[i, :] = sample[:n] # Trim to original length

        return bootstrap_samples

    def _anderson_p_value(self, anderson_result) -> float:
        """
        Convert Anderson-Darling test result to p-value.
        This is an approximation as Anderson-Darling test does not directly return a p-value.
        It compares the statistic to critical values at given significance levels.
        """
        significance_levels = anderson_result.significance_level
        critical_values = anderson_result.critical_values
        statistic = anderson_result.statistic

        for i, critical_value in enumerate(critical_values):
            if statistic < critical_value:
                # Return the significance level corresponding to the critical value
                # This means the p-value is greater than this significance level
                return significance_levels[i] / 100.0
        # If statistic is greater than all critical values, p-value is less than the smallest significance level
        return significance_levels[-1] / 100.0 # Or a very small value, depending on interpretation

    def test_distribution_fit(self, dist: str = 'norm') -> Dict[str, float]:
        """
        Test the distributional fit of the original data using bootstrap samples.

        Parameters:
        - dist: The distribution to test against ('norm', 'expon', 'gamma', 'beta').

        Returns:
        - p_values: A dictionary containing average p-values from different tests.
        """
        p_values = {
            'kstest': [],
            'normaltest': [], # Only applicable for 'norm'
            'anderson': []
        }

        for sample in self.bootstrap_samples:
            if len(sample) < 2: # Skip if sample is too small for tests
                continue

            # Kolmogorov-Smirnov test
            try:
                if dist == 'norm':
                    ks_stat, ks_p_value = kstest(sample, 'norm', args=(np.mean(sample), np.std(sample)))
                elif dist == 'expon':
                    ks_stat, ks_p_value = kstest(sample, 'expon', args=(np.min(sample), np.mean(sample) - np.min(sample))) # scale is mean-loc
                elif dist == 'gamma':
                    # Fit gamma distribution parameters
                    shape, loc, scale = gamma.fit(sample, floc=0) # floc=0 assumes data starts from 0
                    ks_stat, ks_p_value = kstest(sample, 'gamma', args=(shape, loc, scale))
                elif dist == 'beta':
                    # Fit beta distribution parameters (requires data in [0,1])
                    # Scale data to [0,1] if not already
                    min_val, max_val = np.min(sample), np.max(sample)
                    scaled_sample = (sample - min_val) / (max_val - min_val) if (max_val - min_val) > 0 else np.zeros_like(sample)
                    a, b, loc, scale = beta.fit(scaled_sample)
                    ks_stat, ks_p_value = kstest(scaled_sample, 'beta', args=(a, b, loc, scale))
                else:
                    raise ValueError(f"Unsupported distribution: {dist}")
                p_values['kstest'].append(ks_p_value)
            except Exception as e:
                # print(f"KS Test failed for {dist} with sample of size {len(sample)}: {e}")
                pass # Append nothing, or NaN, depending on desired behavior

            # D'Agostino and Pearson's test (for normality only)
            if dist == 'norm':
                try:
                    _, normaltest_p_value = normaltest(sample)
                    p_values['normaltest'].append(normaltest_p_value)
                except Exception as e:
                    # print(f"Normal Test failed: {e}")
                    pass

            # Anderson-Darling test
            try:
                if dist in ['norm', 'expon']: # Anderson-Darling in scipy.stats supports specific distributions
                    anderson_result = anderson(sample, dist=dist)
                    p_values['anderson'].append(self._anderson_p_value(anderson_result))
                else:
                    # Anderson-Darling for other distributions would require custom implementation or external libraries
                    p_values['anderson'].append(np.nan) # Mark as not applicable
            except Exception as e:
                # print(f"Anderson-Darling Test failed for {dist}: {e}")
                pass

        # Calculate average p-values, handling cases where lists might be empty
        avg_p_values = {}
        for key, values in p_values.items():
            avg_p_values[key] = np.mean(values) if values else np.nan

        return avg_p_values

    def print_results_table(self):
        """
        Print the results of distribution fit tests in a table format.
        """
        distributions = ['norm', 'expon', 'gamma', 'beta']
        results_table = []

        headers = ["Distribution", "KS Test P-Value", "Normal Test P-Value", "Anderson Test P-Value"]

        for dist in distributions:
            p_values = self.test_distribution_fit(dist=dist)
            row = [
                dist.capitalize(),
                f"{p_values.get('kstest', np.nan):.4f}" if not np.isnan(p_values.get('kstest', np.nan)) else "N/A",
                f"{p_values.get('normaltest', np.nan):.4f}" if not np.isnan(p_values.get('normaltest', np.nan)) else "N/A",
                f"{p_values.get('anderson', np.nan):.4f}" if not np.isnan(p_values.get('anderson', np.nan)) else "N/A"
            ]
            results_table.append(row)

        print("\nDistributional Fit Test Results (Average P-values from Bootstrap Samples):")
        print(tabulate(results_table, headers=headers, tablefmt="grid"))


# --- 3. Forecasting Models ---

class MDL_AR_Model:
    """
    Minimum Description Length (MDL) based Autoregressive (AR) Model.
    Selects the optimal lag order based on the MDL principle.
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        self.model = None
        self.lags = None

    def fit(self):
        """
        Fits the AR model by selecting the optimal lag order using the MDL criterion.
        """
        n = len(self.data)
        if n < 2:
            raise ValueError("Data length must be at least 2 to fit an AR model.")

        def mdl(lags_val):
            lags_val = int(lags_val)
            if lags_val == 0: # Handle lag 0 case, which means just mean
                residuals = self.data - np.mean(self.data)
            else:
                try:
                    model_fit = AutoReg(self.data, lags=lags_val, trend='c').fit() # 'c' for constant trend
                    residuals = model_fit.resid
                except Exception as e:
                    # print(f"AutoReg fit failed for lags={lags_val}: {e}")
                    return np.inf # Return infinity for failed fits

            sigma2 = np.mean(residuals ** 2)
            if sigma2 <= 0: # Avoid log of non-positive numbers
                return np.inf
            mdl_value = n * np.log(sigma2) + lags_val * np.log(n)
            return mdl_value

        mdl_values = []
        # Iterate through possible lags, up to n/4 as a common heuristic
        max_lags = max(1, n // 4) # Ensure at least 1 lag is considered if data allows
        for lags in range(1, max_lags + 1):
            try:
                value = mdl(lags)
                if not np.isinf(value): # Only append if fit was successful
                    mdl_values.append((lags, value))
            except Exception as e:
                # print(f"MDL calculation failed for lags={lags}: {e}")
                continue

        if not mdl_values:
            # Fallback to AR(1) if no valid MDL values found
            self.lags = 1
            print("Warning: No valid MDL values found. Defaulting to AR(1) model.")
        else:
            mdl_values.sort(key=lambda x: x[1])
            self.lags = mdl_values[0][0]

        self.model = AutoReg(self.data, lags=self.lags, trend='c').fit()

    def forecast(self, steps: int):
        """
        Generates forecasts for the specified number of steps.

        Parameters:
        - steps: The number of future steps to forecast.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call the fit method first.")
        return self.model.predict(start=len(self.data), end=len(self.data) + steps - 1)


class AR1_Model:
    """
    A simple Autoregressive (AR) model of order 1.
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        self.model = None

    def fit(self):
        """
        Fits the AR(1) model to the data.
        """
        if len(self.data) < 2:
            raise ValueError("Data length must be at least 2 to fit an AR(1) model.")
        self.model = AutoReg(self.data, lags=1, trend='c').fit()

    def forecast(self, steps: int):
        """
        Generates forecasts for the specified number of steps.

        Parameters:
        - steps: The number of future steps to forecast.
        """
        if self.model is None:
            raise ValueError("Model is not fitted yet. Call the fit method first.")
        return self.model.predict(start=len(self.data), end=len(self.data) + steps - 1)


class Backtesting:
    """
    Performs backtesting for forecasting models using either expanding or rolling windows.
    Compares MDL_AR_Model and AR1_Model.
    """
    def __init__(self, data: np.ndarray, sample_type: str = 'expanding', window_size: Optional[int] = None):
        """
        Initialize the Backtesting class.

        Parameters:
        - data: The time series data for backtesting.
        - sample_type: 'expanding' or 'rolling'.
        - window_size: Required if sample_type is 'rolling'.
        """
        self.data = data
        self.sample_type = sample_type
        self.window_size = window_size
        self.forecasts_mdl = []
        self.forecasts_ar1 = []
        self.actuals = []

    def run_backtest(self, forecast_steps: int = 1):
        """
        Runs the backtest simulation.

        Parameters:
        - forecast_steps: Number of steps to forecast into the future.
        """
        n = len(self.data)

        if self.sample_type == 'rolling' and (self.window_size is None or self.window_size <= 0):
            raise ValueError("Window size must be specified and positive for rolling sample type.")
        if self.window_size is not None and self.window_size > n:
            raise ValueError("Window size cannot be greater than the total data length.")
        if n < (self.window_size if self.sample_type == 'rolling' else 2) + forecast_steps:
             print("Warning: Data too short for backtesting with current window size/forecast steps. Skipping.")
             return # Do not raise error, just return if data is too short

        # Determine the starting point for backtesting
        # Need at least window_size data points for rolling, or 2 for expanding to fit AR(1)
        start_idx_for_training = self.window_size if self.sample_type == 'rolling' else 2

        for i in range(start_idx_for_training, n - forecast_steps + 1):
            if self.sample_type == 'expanding':
                train_data = self.data[:i]
            elif self.sample_type == 'rolling':
                train_data = self.data[max(0, i - self.window_size):i]

            # Ensure train_data is long enough for model fitting
            if len(train_data) < 2:
                # print(f"Skipping iteration {i}: Training data too short ({len(train_data)} points).")
                continue

            model_mdl = MDL_AR_Model(train_data)
            model_ar1 = AR1_Model(train_data)

            try:
                model_mdl.fit()
                forecast_mdl = model_mdl.forecast(forecast_steps)
                self.forecasts_mdl.append(forecast_mdl[-1])
            except ValueError as e:
                # print(f"MDL model fit/forecast failed at iteration {i}: {e}. Appending NaN.")
                self.forecasts_mdl.append(np.nan)
            except Exception as e:
                # print(f"An unexpected error occurred with MDL model at iteration {i}: {e}. Appending NaN.")
                self.forecasts_mdl.append(np.nan)


            try:
                model_ar1.fit()
                forecast_ar1 = model_ar1.forecast(forecast_steps)
                self.forecasts_ar1.append(forecast_ar1[-1])
            except ValueError as e:
                # print(f"AR(1) model fit/forecast failed at iteration {i}: {e}. Appending NaN.")
                self.forecasts_ar1.append(np.nan)
            except Exception as e:
                # print(f"An unexpected error occurred with AR(1) model at iteration {i}: {e}. Appending NaN.")
                self.forecasts_ar1.append(np.nan)

            self.actuals.append(self.data[i + forecast_steps -1]) # Adjusted index for actuals

        # Filter out NaNs from forecasts and actuals for evaluation
        valid_indices = ~np.isnan(self.forecasts_mdl) & ~np.isnan(self.forecasts_ar1) & ~np.isnan(self.actuals)
        self.forecasts_mdl = np.array(self.forecasts_mdl)[valid_indices]
        self.forecasts_ar1 = np.array(self.forecasts_ar1)[valid_indices]
        self.actuals = np.array(self.actuals)[valid_indices]

        if len(self.actuals) == 0:
            print("No valid forecasts generated for evaluation.")
            return

        self.evaluate()

    def evaluate(self):
        """
        Evaluates the performance of the models and prints a results table.
        """
        if len(self.actuals) == 0:
            print("No actuals available for evaluation.")
            return

        mse_mdl = mean_squared_error(self.actuals, self.forecasts_mdl)
        mae_mdl = mean_absolute_error(self.actuals, self.forecasts_mdl)

        # Ensure that actuals and forecasts are not all zero or constant for sign comparison
        if np.all(self.actuals == 0) or np.all(self.forecasts_mdl == 0):
            percent_correct_sign_predictions_mdl = np.nan
        else:
            correct_sign_predictions_mdl = np.sum(np.sign(self.forecasts_mdl) == np.sign(self.actuals))
            percent_correct_sign_predictions_mdl = correct_sign_predictions_mdl / len(self.actuals) * 100

        mse_ar1 = mean_squared_error(self.actuals, self.forecasts_ar1)
        mae_ar1 = mean_absolute_error(self.actuals, self.forecasts_ar1)

        if np.all(self.actuals == 0) or np.all(self.forecasts_ar1 == 0):
            percent_correct_sign_predictions_ar1 = np.nan
        else:
            correct_sign_predictions_ar1 = np.sum(np.sign(self.forecasts_ar1) == np.sign(self.actuals))
            percent_correct_sign_predictions_ar1 = correct_sign_predictions_ar1 / len(self.actuals) * 100

        mse_relative_to_ar1 = mse_mdl / mse_ar1 if mse_ar1 != 0 else np.inf
        mae_relative_to_ar1 = mae_mdl / mae_ar1 if mae_ar1 != 0 else np.inf
        percent_correct_sign_relative_to_ar1 = percent_correct_sign_predictions_mdl / percent_correct_sign_predictions_ar1 if percent_correct_sign_predictions_ar1 != 0 else np.inf

        results_table = [
            ["Metric", "MDL Model", "AR(1) Model", "Relative to AR(1)"],
            ["MSE", f"{mse_mdl:.4f}", f"{mse_ar1:.4f}", f"{mse_relative_to_ar1:.4f}"],
            ["MAE", f"{mae_mdl:.4f}", f"{mae_ar1:.4f}", f"{mae_relative_to_ar1:.4f}"],
            ["Percent Correct Sign Predictions", f"{percent_correct_sign_predictions_mdl:.2f}%",
             f"{percent_correct_sign_predictions_ar1:.2f}%", f"{percent_correct_sign_relative_to_ar1:.4f}"]
        ]

        print(f"\nBacktesting Results ({self.sample_type.capitalize()} Sample):")
        print(tabulate(results_table, headers="firstrow", tablefmt="grid"))

    def plot_forecasts_and_errors(self):
        """
        Plots the actual vs. forecasted values and the forecast errors.
        """
        if len(self.actuals) == 0:
            print("No actuals or forecasts to plot.")
            return

        plt.figure(figsize=(14, 10))

        # Plot actual vs. forecasted values
        plt.subplot(2, 1, 1)
        plt.plot(self.actuals, label='Actual', color='blue', linewidth=2)
        plt.plot(self.forecasts_mdl, label='MDL Forecast', color='orange', linestyle='--')
        plt.plot(self.forecasts_ar1, label='AR(1) Forecast', color='green', linestyle='-.')
        plt.legend()
        plt.title('Actual vs. Forecasted Values')
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.grid(True)

        # Plot forecast errors
        errors_mdl = np.array(self.actuals) - np.array(self.forecasts_mdl)
        errors_ar1 = np.array(self.actuals) - np.array(self.forecasts_ar1)

        plt.subplot(2, 1, 2)
        plt.plot(errors_mdl, label='MDL Forecast Errors', color='orange')
        plt.plot(errors_ar1, label='AR(1) Forecast Errors', color='green')
        plt.legend()
        plt.title('Forecast Errors')
        plt.xlabel('Time Index')
        plt.ylabel('Error')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


class RealTimeForecasting:
    """
    Performs real-time forecasting using MDL_AR_Model and AR1_Model on different sample sizes.
    """
    def __init__(self, data: np.ndarray):
        self.data = data

    def compute_forecasts(self, forecast_steps: int = 1):
        """
        Computes forecasts for the next `forecast_steps` using different sample sizes.
        """
        n = len(self.data)
        if n < 4: # Need at least 4 data points for 25% sample
            raise ValueError("Data length must be at least 4 for real-time forecasting with multiple samples.")

        # Full Sample
        full_sample_forecast_mdl_value = np.nan
        full_sample_forecast_ar1_value = np.nan
        try:
            full_sample_forecast_mdl = MDL_AR_Model(self.data)
            full_sample_forecast_mdl.fit()
            full_sample_forecast_mdl_value = full_sample_forecast_mdl.forecast(steps=forecast_steps)[-1]
        except Exception as e:
            print(f"Full sample MDL forecast failed: {e}")

        try:
            full_sample_forecast_ar1 = AR1_Model(self.data)
            full_sample_forecast_ar1.fit()
            full_sample_forecast_ar1_value = full_sample_forecast_ar1.forecast(steps=forecast_steps)[-1]
        except Exception as e:
            print(f"Full sample AR(1) forecast failed: {e}")

        # Last 25%, 50%, 75% samples
        sample_sizes = {'25%': n//4, '50%': n//2, '75%': 3*n//4}
        forecasts_mdl_per_sample = [full_sample_forecast_mdl_value]
        forecasts_ar1_per_sample = [full_sample_forecast_ar1_value]

        results_table = [
            ["Sample", "MDL Model Forecast", "AR(1) Model Forecast"]
        ]
        results_table.append(["Full Sample", f"{full_sample_forecast_mdl_value:.4f}" if not np.isnan(full_sample_forecast_mdl_value) else "N/A",
                              f"{full_sample_forecast_ar1_value:.4f}" if not np.isnan(full_sample_forecast_ar1_value) else "N/A"])

        for label, size in sample_sizes.items():
            if size == 0: # Ensure sample size is not zero
                mdl_val, ar1_val = np.nan, np.nan
            else:
                current_data = self.data[-size:]
                mdl_val, ar1_val = np.nan, np.nan
                try:
                    mdl_model = MDL_AR_Model(current_data)
                    mdl_model.fit()
                    mdl_val = mdl_model.forecast(steps=forecast_steps)[-1]
                except Exception as e:
                    print(f"Last {label} MDL forecast failed: {e}")
                try:
                    ar1_model = AR1_Model(current_data)
                    ar1_model.fit()
                    ar1_val = ar1_model.forecast(steps=forecast_steps)[-1]
                except Exception as e:
                    print(f"Last {label} AR(1) forecast failed: {e}")

            forecasts_mdl_per_sample.append(mdl_val)
            forecasts_ar1_per_sample.append(ar1_val)
            results_table.append([f"Last {label} Sample", f"{mdl_val:.4f}" if not np.isnan(mdl_val) else "N/A",
                                  f"{ar1_val:.4f}" if not np.isnan(ar1_val) else "N/A"])

        # Filter out NaN values before averaging
        valid_mdl_forecasts = [f for f in forecasts_mdl_per_sample if not np.isnan(f)]
        valid_ar1_forecasts = [f for f in forecasts_ar1_per_sample if not np.isnan(f)]

        average_forecast_mdl = np.mean(valid_mdl_forecasts) if valid_mdl_forecasts else np.nan
        average_forecast_ar1 = np.mean(valid_ar1_forecasts) if valid_ar1_forecasts else np.nan

        results_table.append(["Average Forecast", f"{average_forecast_mdl:.4f}" if not np.isnan(average_forecast_mdl) else "N/A",
                              f"{average_forecast_ar1:.4f}" if not np.isnan(average_forecast_ar1) else "N/A"])

        print("\nReal-Time Forecasting Results:")
        print(tabulate(results_table, headers="firstrow", tablefmt="grid"))


class UnifiedForecaster:
    """
    A unified forecaster class supporting both AR models (with lag selection) and SVR.
    """
    def __init__(self, y: np.ndarray, R: int, lag_criterion: str = 'AIC',
                 use_svr: bool = False, svr_params: Optional[Dict] = None, verbose: bool = False):
        """
        Initialize the UnifiedForecaster.

        Parameters:
        - y: The time series data (1D numpy array).
        - R: The window size for historical observations for prediction.
        - lag_criterion: Criterion for AR lag selection ('AIC', 'BIC', 'HQIC').
        - use_svr: If True, uses SVR for forecasting; otherwise, uses AR models.
        - svr_params: Dictionary of SVR parameters if use_svr is True.
        - verbose: If True, prints additional debug information.
        """
        self.y = np.asarray(y)
        self.R = R
        self.lag_criterion = lag_criterion
        self.use_svr = use_svr
        self.verbose = verbose

        if len(self.y) <= R:
            raise ValueError(f"Time series length ({len(self.y)}) must be greater than R ({R}) to generate forecasts.")

        if use_svr:
            self.svr_params = svr_params or {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1, 'gamma': 'scale'}
            self.scaler = StandardScaler()
            self.svr_model = SVR(**self.svr_params)
            self.forecasts_svr, self.actual_values = self._compute_svr_forecast()
        else:
            self.forecasts_baseline = self._compute_baseline_forecast()
            self.forecasts_ar1 = self._compute_ar1_forecast()
            self.forecasts_ar_selected = self._compute_ar_selected_forecast()
            self.actual_values = self.y[R:]

    # === Traditional Forecasting Methods ===
    def _compute_baseline_forecast(self) -> np.ndarray:
        """Computes a simple baseline forecast (e.g., last observed value or mean of window)."""
        forecasts = []
        for i in range(len(self.y) - self.R):
            window = self.y[i:i+self.R]
            forecasts.append(window[-1]) # Using last value as baseline
        return np.array(forecasts)

    def _compute_ar1_forecast(self) -> np.ndarray:
        """Computes AR(1) forecasts using a rolling window."""
        forecasts = []
        for i in range(len(self.y) - self.R):
            window = self.y[i:i+self.R]
            try:
                model = AutoReg(window, lags=1, trend='c')
                forecast = model.fit().predict(start=len(window), end=len(window))[0]
                forecasts.append(forecast)
            except Exception as e:
                if self.verbose:
                    print(f"AR(1) forecast failed at index {i}: {e}")
                forecasts.append(np.nan)
        return np.array(forecasts)

    def _compute_ar_selected_forecast(self) -> np.ndarray:
        """Computes AR forecasts with lag selection using a rolling window."""
        forecasts = []
        for i in range(len(self.y) - self.R):
            window = self.y[i:i+self.R]
            p = self._select_best_lag(window)
            try:
                model = AutoReg(window, lags=p, trend='c')
                forecast = model.fit().predict(start=len(window), end=len(window))[0]
                forecasts.append(forecast)
            except Exception as e:
                if self.verbose:
                    print(f"AR({p}) forecast failed at index {i}: {e}")
                forecasts.append(np.nan)
        return np.array(forecasts)

    def _select_best_lag(self, window: np.ndarray) -> int:
        """
        Selects the best lag for an AR model based on the specified criterion.
        """
        criteria = []
        max_lag_to_consider = max(1, len(window) - 1) # Must be at least 1, and less than window size
        for p in range(1, max_lag_to_consider + 1):
            try:
                model = AutoReg(window, lags=p, trend='c')
                fit = model.fit()
                value = getattr(fit, self.lag_criterion.lower()) # AIC, BIC, HQIC
                criteria.append((p, value))
            except Exception as e:
                if self.verbose:
                    print(f"Lag selection failed for p={p}: {e}")
                continue

        if not criteria:
            return 1 # Fallback to AR(1) if no valid lags found
        return min(criteria, key=lambda x: x[1])[0]

    # === SVR Forecasting (rolling) ===
    def _compute_svr_forecast(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes rolling SVR forecasts.
        """
        forecasts = []
        actuals = []

        # Minimum data points needed for a window + target
        if len(self.y) < self.R + 1:
            return np.array([]), np.array([])

        for i in range(len(self.y) - self.R):
            X_window = self.y[i:i+self.R].reshape(1, -1) # Input features are the R previous values
            y_target = self.y[i + self.R] # Target is the next value

            # Fit scaler on the current window and transform
            X_scaled = self.scaler.fit_transform(X_window)

            # Train the SVR model on the current window's data and its target
            # For rolling forecast, we effectively train on (R previous values -> current value)
            # and then predict the next value based on the current R values.
            # This implementation trains on the current window to predict the *next* value.
            # A more robust rolling SVR would train on an expanding or rolling window of past (X,y) pairs.
            # For simplicity, this version trains on the current window and predicts the immediate next.
            # A better approach for SVR in a rolling window context would be to collect X,y pairs
            # from the historical window and train on those.

            # Let's revise this to be more standard for rolling forecast:
            # We use data from (i to i+R-1) to predict (i+R)
            # The training data for SVR should be pairs (X_t, y_t) where X_t is a window and y_t is the next value.
            # We will use an expanding window for training the SVR model itself.

            if i == 0:
                # For the very first prediction, we train on the first R points to predict R+1
                # This is a bit of a cold start. We'll use the first R points as X and the Rth point as y for initial fit.
                # Then predict the R+1th point.
                initial_X_train = self.y[:self.R].reshape(1, -1)
                initial_y_train = self.y[self.R]
                self.scaler.fit(initial_X_train)
                self.svr_model.fit(self.scaler.transform(initial_X_train), np.array([initial_y_train]))

                # Predict the first point using the just-fitted model
                forecast = self.svr_model.predict(self.scaler.transform(X_window))[0]
                forecasts.append(forecast)
            else:
                # For subsequent steps, we predict based on the current window
                forecast = self.svr_model.predict(X_scaled)[0]
                forecasts.append(forecast)

                # Then, we update the model by fitting it with the new actual data point.
                # The training data for the SVR model itself should be an expanding window of (X, y) pairs.
                # X for training will be (y[k:k+R]), and y for training will be (y[k+R])

                # Collect all available past (X,y) pairs up to current point i
                past_X_train = []
                past_y_train = []
                for j in range(i + 1): # Up to current window's target
                    if j + self.R < len(self.y):
                        past_X_train.append(self.y[j:j+self.R])
                        past_y_train.append(self.y[j+self.R])

                if past_X_train and past_y_train:
                    past_X_train = np.array(past_X_train)
                    past_y_train = np.array(past_y_train)

                    self.scaler.fit(past_X_train) # Fit scaler on all past X
                    self.svr_model.fit(self.scaler.transform(past_X_train), past_y_train)

            actuals.append(y_target)

        return np.array(forecasts), np.array(actuals)


    # === Evaluation ===
    def evaluate(self):
        """
        Evaluates the forecasting performance using various metrics.
        """
        if self.use_svr:
            if len(self.actual_values) == 0:
                return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R²": np.nan}

            # Filter out NaNs from SVR forecasts
            valid_indices = ~np.isnan(self.forecasts_svr)
            actuals_filtered = self.actual_values[valid_indices]
            forecasts_filtered = self.forecasts_svr[valid_indices]

            if len(actuals_filtered) == 0:
                return {"MSE": np.nan, "RMSE": np.nan, "MAE": np.nan, "MAPE": np.nan, "R²": np.nan}

            metrics = {
                'MSE': mean_squared_error(actuals_filtered, forecasts_filtered),
                'RMSE': np.sqrt(mean_squared_error(actuals_filtered, forecasts_filtered)),
                'MAE': mean_absolute_error(actuals_filtered, forecasts_filtered),
                'R²': r2_score(actuals_filtered, forecasts_filtered)
            }
            try:
                metrics['MAPE'] = mean_absolute_percentage_error(actuals_filtered, forecasts_filtered)
            except ValueError:
                metrics['MAPE'] = np.nan # Handle cases where actuals contain zeros for MAPE
            return metrics
        else:
            if len(self.actual_values) == 0:
                return pd.DataFrame() # Return empty DataFrame if no actuals

            results = []
            models_to_evaluate = {
                "Baseline (Last Value)": self.forecasts_baseline,
                "AR(1)": self.forecasts_ar1,
                f"AR({self.lag_criterion})": self.forecasts_ar_selected
            }

            for name, forecast in models_to_evaluate.items():
                # Filter out NaNs for each forecast
                valid_indices = ~np.isnan(forecast)
                actuals_filtered = self.actual_values[valid_indices]
                forecast_filtered = forecast[valid_indices]

                if len(actuals_filtered) == 0:
                    results.append({
                        "Model": name, "MSE": np.nan, "MAE": np.nan,
                        "Directional Accuracy (%)": np.nan
                    })
                    continue

                errors = actuals_filtered - forecast_filtered

                # Directional accuracy: compare sign of actual change vs. predicted change
                # For time series, this often means comparing sign of (actual_t - actual_{t-1}) vs (forecast_t - actual_{t-1})
                # Or simply sign of forecast vs sign of actual if data is centered around zero.
                # Given the current implementation, it's sign of forecast vs sign of actual.

                # Ensure no division by zero or log of zero for sign comparison
                if np.all(actuals_filtered == 0) or np.all(forecast_filtered == 0):
                    directional_accuracy = np.nan
                else:
                    directional_accuracy = np.mean(
                        np.sign(forecast_filtered) == np.sign(actuals_filtered)
                    ) * 100

                results.append({
                    "Model": name,
                    "MSE": np.mean(errors**2),
                    "MAE": np.mean(np.abs(errors)),
                    "Directional Accuracy (%)": directional_accuracy
                })
            return pd.DataFrame(results).set_index("Model")

    # === Outputs ===
    def get_predictions(self):
        """
        Returns the actual values and the generated forecasts.
        """
        if self.use_svr:
            return self.actual_values, self.forecasts_svr
        else:
            return {
                'actual': self.actual_values,
                'baseline': self.forecasts_baseline,
                'ar1': self.forecasts_ar1,
                'ar_selected': self.forecasts_ar_selected
            }

    def plot(self):
        """
        Plots the actual values against the forecasted values.
        """
        if len(self.actual_values) == 0:
            print("No actuals or forecasts to plot.")
            return

        plt.figure(figsize=(12, 6))
        if self.use_svr:
            plt.plot(self.actual_values, label='Actual', linewidth=2, color='blue')
            plt.plot(self.forecasts_svr, '--', label='SVR Forecast', color='red')
        else:
            plt.plot(self.actual_values, label='Actual', linewidth=2, color='blue')
            plt.plot(self.forecasts_baseline, '--', label='Baseline', color='orange')
            plt.plot(self.forecasts_ar1, '-.', label='AR(1)', color='green')
            plt.plot(self.forecasts_ar_selected, ':', label=f"AR({self.lag_criterion})", color='purple')
        plt.title("Forecast vs Actual")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()


class EnhancedRollingSVRForecaster:
    """
    Enhanced rolling window SVR forecaster with data preparation, validation,
    hyperparameter tuning, and comprehensive evaluation.
    """
    def __init__(self, window_size: int = 24, kernel: str = 'rbf',
                 C: float = 1.0, epsilon: float = 0.1, gamma: str = 'scale',
                 verbose: bool = False):
        """
        :param window_size: Historical observations per prediction (R)
        :param kernel: SVM kernel type
        :param C: Regularization parameter
        :param epsilon: Epsilon-tube width
        :param gamma: Kernel coefficient
        :param verbose: Enable debug outputs
        """
        self.window_size = window_size
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.verbose = verbose
        self.scaler = StandardScaler()
        self.model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        self.X_train = None
        self.y_train = None
        self.train_indices = None
        self.test_indices = None

        # Validate parameters
        if window_size < 1:
            raise ValueError("window_size must be ≥ 1")
        if C <= 0:
            raise ValueError("C must be > 0")

    def _create_rolling_windows(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rolling windows for SVR input (X) and target (y).
        X will be (N-window_size) x window_size, y will be (N-window_size).
        """
        if len(data) < self.window_size + 1: # Need at least window_size + 1 points for one X,y pair
            raise ValueError(f"Data length ({len(data)}) must be at least window_size + 1 ({self.window_size + 1}) to create rolling windows.")

        n_windows = len(data) - self.window_size
        X = np.zeros((n_windows, self.window_size))
        y = np.zeros(n_windows)

        for i in range(n_windows):
            X[i] = data[i : i + self.window_size]
            y[i] = data[i + self.window_size]

        return X, y

    def prepare_data(self, data: np.ndarray, train_size: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for SVR by creating rolling windows and splitting into train/test sets.
        Scales data using StandardScaler.

        Parameters:
        - data: The input time series data (1D numpy array).
        - train_size: Proportion of data to use for training (0.0 to 1.0).

        Returns:
        - Tuple of (X_train, X_test, y_train, y_test).
        """
        # Input checks
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        if data.ndim != 1:
            raise ValueError("Data must be 1D array")
        if not (0 < train_size < 1):
            raise ValueError("train_size must be between 0 and 1 (exclusive).")

        X, y = self._create_rolling_windows(data)
        split_idx = int(len(X) * train_size)

        if split_idx == 0 or split_idx == len(X):
            raise ValueError("Training or testing set would be empty. Adjust train_size or data length.")

        # Temporal split
        self.X_train, self.y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]

        # Store indices for visualization
        self.train_indices = np.arange(self.window_size, self.window_size + split_idx)
        self.test_indices = np.arange(self.window_size + split_idx, len(data))

        # Scaling with leakage protection: fit only on training data
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        X_test = self.scaler.transform(X_test)

        if self.verbose:
            print(f"Data prepared. X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

        return self.X_train, X_test, self.y_train, y_test

    def fit(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
        """
        Trains the SVR model.

        Parameters:
        - X: Training features. If None, uses self.X_train.
        - y: Training targets. If None, uses self.y_train.
        """
        X_fit = X if X is not None else self.X_train
        y_fit = y if y is not None else self.y_train

        if X_fit is None or y_fit is None:
            raise ValueError("No training data available. Run prepare_data() first or provide X, y.")
        if len(X_fit) == 0:
            raise ValueError("Training data is empty.")

        self.model.fit(X_fit, y_fit)
        if self.verbose:
            print(f"Model trained on {len(X_fit)} samples.")
        return self

    def rolling_predict(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates rolling predictions for the entire time series.

        Parameters:
        - data: The full time series data (1D numpy array).

        Returns:
        - Tuple of (predictions, actuals).
        """
        X, y = self._create_rolling_windows(data)
        X_scaled = self.scaler.transform(X) # Use the scaler fitted on training data
        predictions = self.model.predict(X_scaled)

        if self.verbose:
            print("\nDebug Predictions (first 5):")
            for i in range(min(5, len(predictions))):
                print(f"Window {i+1}: {X[i]} -> Pred: {predictions[i]:.3f} | Actual: {y[i]:.3f}")
            print("...")
            print("Debug Predictions (last 5):")
            for i in range(max(0, len(predictions)-5), len(predictions)):
                print(f"Window {i+1}: {X[i]} -> Pred: {predictions[i]:.3f} | Actual: {y[i]:.3f}")

        return predictions, y

    def evaluate_rolling(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculates comprehensive evaluation metrics for rolling predictions.

        Parameters:
        - data: The full time series data (1D numpy array).

        Returns:
        - Dictionary of metrics (MSE, RMSE, MAE, R², MAPE).
        """
        predictions, actuals = self.rolling_predict(data)

        if len(actuals) == 0:
            return {'MSE': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'R²': np.nan, 'MAPE': np.nan}

        metrics = {
            'MSE': mean_squared_error(actuals, predictions),
            'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
            'MAE': mean_absolute_error(actuals, predictions),
            'R²': r2_score(actuals, predictions)
        }

        try:
            metrics['MAPE'] = mean_absolute_percentage_error(actuals, predictions)
        except ValueError:
            if self.verbose:
                print("MAPE skipped due to zero values in actuals.")
            metrics['MAPE'] = np.nan

        return metrics

    def visualize(self, data: np.ndarray, predictions: np.ndarray):
        """
        Visualizes the actual time series, predictions, and highlights train/test regions.

        Parameters:
        - data: The full time series data (1D numpy array).
        - predictions: The generated predictions.
        """
        if len(data) == 0 or len(predictions) == 0:
            print("No data or predictions to visualize.")
            return

        plt.figure(figsize=(14, 7))

        # Plot actuals
        plt.plot(data, label='Actual', color='#1f77b4', alpha=0.7)

        # Plot predictions aligned with their true positions
        # Predictions start at index `window_size`
        pred_indices = np.arange(self.window_size, self.window_size + len(predictions))
        plt.plot(pred_indices, predictions, label='Predictions',
                linestyle='--', color='#ff7f0e', linewidth=2)

        # Highlight train-test split
        if self.train_indices is not None and self.test_indices is not None:
            # The split point is the last index of the training predictions
            split_point_index = self.train_indices[-1] if len(self.train_indices) > 0 else self.window_size

            plt.axvline(split_point_index, color='red', linestyle=':',
                       label='Train/Test Split')

            # Fill between for training region
            if len(self.train_indices) > 0:
                plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]], # Use current y-limits
                                x1=self.train_indices[0], x2=self.train_indices[-1],
                                color='green', alpha=0.1, label='Training Region')
            # Fill between for test region
            if len(self.test_indices) > 0:
                plt.fill_betweenx(y=[plt.ylim()[0], plt.ylim()[1]],
                                x1=self.test_indices[0], x2=self.test_indices[-1],
                                color='red', alpha=0.1, label='Test Region')

        plt.title(f"Rolling Window Forecast (Window Size={self.window_size})")
        plt.xlabel("Time Index")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()

    def tune_hyperparameters(self, param_grid: Optional[Dict] = None,
                            cv: int = 5) -> Dict:
        """
        Performs hyperparameter tuning for the SVR model using GridSearchCV.

        Parameters:
        - param_grid: Dictionary of hyperparameters to search. If None, uses a default grid.
        - cv: Number of cross-validation folds.

        Returns:
        - Dictionary of best hyperparameters found.
        """
        if self.X_train is None or self.y_train is None:
            raise ValueError("Run prepare_data() first to set up training data.")

        param_grid = param_grid or {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1, 1],
            'epsilon': [0.001, 0.01, 0.1]
        }

        grid = GridSearchCV(SVR(kernel=self.kernel), param_grid,
                           scoring='neg_mean_squared_error', # Use negative MSE for maximization
                           cv=cv, n_jobs=-1, verbose=1) # verbose=1 to see progress
        grid.fit(self.X_train, self.y_train)

        self.model = grid.best_estimator_ # Update the model with the best estimator
        if self.verbose:
            print(f"Best parameters: {grid.best_params_}")
            print(f"Best RMSE: {np.sqrt(-grid.best_score_):.4f}")

        return grid.best_params_


class AdaptiveLearningForecaster:
    """
    Unified implementation of adaptive learning forecasting based on Kyriazi, Thomakos, and Guerard (2019).
    Supports both continuous and binary (-1/+1) data.
    """
    def __init__(
        self,
        models: List[callable],
        mode: str = "continuous",
        initial_weights: Optional[List[float]] = None,
        learning_rate: float = 0.1,
        window_size: int = 20,
        threshold: float = 0.0
    ):
        if mode not in ["continuous", "binary"]:
            raise ValueError("Mode must be 'continuous' or 'binary'")
        if not models:
            raise ValueError("At least one model must be provided.")
        if not (0 <= learning_rate <= 1):
            raise ValueError("Learning rate must be between 0 and 1.")
        if window_size <= 0:
            raise ValueError("Window size must be a positive integer.")

        self.models = models
        self.n_models = len(models)
        self.mode = mode

        if initial_weights is None:
            self.weights = np.ones(self.n_models) / self.n_models
        else:
            self.weights = np.array(initial_weights)
            if np.sum(self.weights) == 0:
                raise ValueError("Initial weights cannot sum to zero.")
            self.weights = self.weights / np.sum(self.weights) # Normalize initial weights

        self.learning_rate = learning_rate
        self.window_size = window_size
        self.threshold = threshold if mode == "binary" else None

        self.forecast_history = [] # Stores combined forecasts (raw, before binary conversion)
        self.output_history = []    # Stores final output (continuous or binary)
        self.error_history = []     # Stores errors for weight update
        self.actual_history = []    # Stores actual values

    def forecast(self, data: np.ndarray) -> float:
        """
        Generates a combined forecast from the ensemble of models.

        Parameters:
        - data: The historical data to use for forecasting.

        Returns:
        - The combined forecast (continuous or binary).
        """
        if len(data) == 0:
            # If no data, return a default or raise error depending on desired behavior
            # For now, return 0 or 1/-1 for binary
            return 0.0 if self.mode == "continuous" else 1.0

        model_forecasts = []
        for model in self.models:
            try:
                # Pass a copy to models to prevent in-place modification if any
                model_forecasts.append(model(data.copy()))
            except Exception as e:
                # print(f"Warning: Individual model forecast failed: {e}. Using 0.0 for this model.")
                model_forecasts.append(0.0) # Use a neutral value if a model fails

        model_forecasts = np.array(model_forecasts)

        # Handle cases where all model forecasts might be invalid (e.g., all NaNs or inf)
        if np.all(~np.isfinite(model_forecasts)):
            combined_forecast = 0.0 # Default if all models fail
        else:
            # Only use valid forecasts for weighted sum
            valid_forecasts_mask = np.isfinite(model_forecasts)
            if np.any(valid_forecasts_mask):
                combined_forecast = np.dot(self.weights[valid_forecasts_mask], model_forecasts[valid_forecasts_mask]) / np.sum(self.weights[valid_forecasts_mask])
            else:
                combined_forecast = 0.0 # Fallback if no valid forecasts

        if self.mode == "continuous":
            forecast = combined_forecast
        else:
            forecast = 1 if combined_forecast >= self.threshold else -1

        self.forecast_history.append(combined_forecast)
        self.output_history.append(forecast)
        return forecast

    def update(self, actual: float) -> None:
        """
        Updates the model weights based on the latest actual value and forecast error.

        Parameters:
        - actual: The actual observed value.
        """
        if not self.output_history:
            raise ValueError("Must make a forecast before updating.")
        if self.mode == "binary" and actual not in [-1, 1]:
            raise ValueError("Actual value must be -1 or +1 in binary mode.")

        self.actual_history.append(actual)

        latest_output = self.output_history[-1]
        error = 0.0
        if self.mode == "continuous":
            error = (actual - latest_output) ** 2
        else:
            error = 1 if latest_output != actual else 0 # 0 for correct, 1 for incorrect

        self.error_history.append(error)

        # Maintain window size for history
        if len(self.error_history) > self.window_size:
            self.error_history.pop(0)
            self.forecast_history.pop(0)
            self.output_history.pop(0)
            self.actual_history.pop(0)

        self._update_weights()

    def _update_weights(self) -> None:
        """
        Updates the weights of individual models based on their recent performance.
        """
        if len(self.error_history) < 2: # Need at least two errors to calculate relative performance meaningfully
            return

        model_errors = []
        # Calculate error for each individual model over the window_size
        for model_idx, model in enumerate(self.models):
            errors_for_this_model = []
            # Re-forecast with each model over the historical window to get individual errors
            for i in range(len(self.actual_history)):
                # Data slice for individual model forecast: data up to actual_history[i-1]
                # This is tricky because `model(data)` expects `data` to be the history *up to* the point of forecast.
                # We need to simulate the forecast that *would have been made* at each historical step.

                # For simplicity and to align with the original design intent of `update`,
                # we'll assume `model(data_slice)` can work with `data_slice` being the historical actuals
                # leading up to the point where the forecast was made.

                # Let's use the actual_history to re-evaluate model performance
                if i > 0: # Need at least one previous point for a forecast
                    data_slice_for_model = np.array(self.actual_history[:i])
                    if len(data_slice_for_model) == 0:
                        continue

                    try:
                        individual_pred_raw = model(data_slice_for_model)
                        actual_val = self.actual_history[i]

                        if self.mode == "continuous":
                            error = (actual_val - individual_pred_raw) ** 2
                        else:
                            individual_pred_binary = 1 if individual_pred_raw >= self.threshold else -1
                            error = 1 if individual_pred_binary != actual_val else 0
                        errors_for_this_model.append(error)
                    except Exception as e:
                        # print(f"Error re-evaluating model {model_idx} at history step {i}: {e}")
                        errors_for_this_model.append(np.inf) # Mark as bad performance

            error_metric = np.mean(errors_for_this_model) if errors_for_this_model else np.inf
            model_errors.append(error_metric)

        model_errors = np.array(model_errors)

        # Handle cases where all models have infinite errors (e.g., all failed)
        if np.all(np.isinf(model_errors)):
            # If all models failed, do not update weights or keep them as is
            return

        # Normalize errors to avoid division by zero or very small numbers
        # Add a small epsilon to the denominator to prevent division by zero
        mean_finite_errors = np.mean(model_errors[np.isfinite(model_errors)])
        if mean_finite_errors == 0:
            mean_finite_errors = 1e-10 # Prevent division by zero if all finite errors are zero

        relative_performance = 1 / (1 + model_errors / (mean_finite_errors + 1e-10))

        # Adaptive learning rate based on overall error magnitude
        # Ensure error_history is not empty before calculating mean
        current_mean_error = np.mean(self.error_history) if self.error_history else 0.0
        adaptive_lr = self.learning_rate * (1 - np.exp(-current_mean_error))

        # Update weights
        new_weights = self.weights + adaptive_lr * (relative_performance - np.mean(relative_performance))

        # Normalize and clip weights to ensure they sum to 1 and are within [0, 1]
        sum_new_weights = np.sum(new_weights)
        if sum_new_weights <= 0: # Avoid division by zero or negative sum
            self.weights = np.ones(self.n_models) / self.n_models # Reset to uniform if sum is problematic
        else:
            self.weights = new_weights / sum_new_weights
            self.weights = np.clip(self.weights, 0, 1) # Ensure weights are between 0 and 1

    def get_weights(self) -> np.ndarray:
        """
        Returns the current weights of the individual models.
        """
        return self.weights.copy()


# Define robust forecasting models for AdaptiveLearningForecaster
def moving_average_model(data: np.ndarray) -> float:
    """Moving average over last 5 points, robust to short data."""
    window = 5
    if len(data) == 0:
        return 0.0
    return np.mean(data[-window:]) if len(data) >= window else np.mean(data)

def momentum_model(data: np.ndarray) -> float:
    """Predict based on last difference, robust to short data."""
    if len(data) < 2:
        return data[-1] if len(data) > 0 else 0.0
    return data[-1] + (data[-1] - data[-2])

def majority_vote_model(data: np.ndarray) -> float:
    """Predict based on majority sign of last 5 differences."""
    if len(data) < 2:
        return 1.0 # Default to +1 if not enough data for difference
    diffs = np.diff(data[-6:]) if len(data) >= 6 else np.diff(data)
    if len(diffs) == 0:
        return 1.0 # No differences to compute sign

    # Count positive, negative, and zero differences
    pos_count = np.sum(diffs > 0)
    neg_count = np.sum(diffs < 0)

    if pos_count > neg_count:
        return 1.0
    elif neg_count > pos_count:
        return -1.0
    else:
        return 0.0 # Tie, or all zeros, return 0 or a neutral value


class ARDLX:
    """
    Autoregressive Distributed Lag with eXogenous variables (ARDLX) model.
    Supports OLS and Elastic Net regression.
    """
    def __init__(self, p: int, model_type: str = 'ols', alpha: float = 1.0, l1_ratio: float = 0.5):
        """
        :param p: Number of autoregressive lags for y.
        :param model_type: 'ols' for Ordinary Least Squares or 'elastic_net' for Elastic Net.
        :param alpha: Regularization strength for ElasticNet (ignored if model_type is 'ols').
        :param l1_ratio: L1 ratio for ElasticNet (ignored if model_type is 'ols').
        """
        if p < 0:
            raise ValueError("p (number of autoregressive lags) must be non-negative.")
        if model_type not in ['ols', 'elastic_net']:
            raise ValueError("model_type must be 'ols' or 'elastic_net'.")

        self.p = p
        self.model_type = model_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = None
        self.feature_names = None

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        Fits the ARDLX model.

        Parameters:
        - y: Dependent variable (pandas Series).
        - X: Exogenous variables (pandas DataFrame).
        """
        if not isinstance(y, pd.Series) or not isinstance(X, pd.DataFrame):
            raise TypeError("y must be a pandas Series and X must be a pandas DataFrame.")
        if y.empty or X.empty:
            raise ValueError("Input data (y or X) cannot be empty.")

        y_lags = pd.DataFrame()
        for i in range(1, self.p + 1):
            y_lags[f'y_lag_{i}'] = y.shift(i)

        # Combine lagged y and X, then drop rows with NaNs
        combined_X = pd.concat([y_lags, X], axis=1)

        # Align y with the combined_X after dropping NaNs
        aligned_data = pd.concat([combined_X, y.rename('y_target')], axis=1).dropna()

        if aligned_data.empty:
            raise ValueError("After creating lags and dropping NaNs, no valid data points remain for fitting.")

        y_aligned = aligned_data['y_target']
        X_aligned = aligned_data.drop(columns=['y_target'])

        if self.model_type == 'ols':
            self.model = LinearRegression()
        elif self.model_type == 'elastic_net':
            self.model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, fit_intercept=True, random_state=42)

        self.model.fit(X_aligned, y_aligned)
        self.feature_names = X_aligned.columns.tolist()

    def predict(self, X_new: pd.DataFrame):
        """
        Predicts using the fitted ARDLX model.

        Parameters:
        - X_new: New data for prediction (pandas DataFrame), must have the same columns as training X,
                 including lagged y terms if applicable.
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        if self.feature_names is None:
            raise ValueError("Feature names not set. Model might not have been fitted correctly.")

        # Ensure X_new has the same columns as the training features
        # This is crucial for prediction consistency
        missing_cols = set(self.feature_names) - set(X_new.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in X_new for prediction: {missing_cols}. "
                             "Ensure X_new contains all features used during training, including lagged y terms.")

        # Reorder columns of X_new to match the order used during fitting
        X_new_ordered = X_new[self.feature_names]

        return self.model.predict(X_new_ordered)


class ESN:
    """
    Echo State Network (ESN) for time series forecasting.
    """
    def __init__(self, reservoir_size: int = 50, leaking_rate: float = 0.3,
                 spectral_radius: float = 0.9, input_scaling: float = 1.0,
                 ridge_param: float = 1e-5, random_state: Optional[int] = None):
        """
        :param reservoir_size: Number of neurons in the reservoir.
        :param leaking_rate: Leaking rate (alpha).
        :param spectral_radius: Spectral radius of the reservoir weight matrix.
        :param input_scaling: Scaling factor for input weights.
        :param ridge_param: Ridge regularization parameter for output weights.
        :param random_state: Seed for random number generation for reproducibility.
        """
        if reservoir_size <= 0:
            raise ValueError("reservoir_size must be a positive integer.")
        if not (0 < leaking_rate <= 1):
            raise ValueError("leaking_rate must be between 0 (exclusive) and 1 (inclusive).")
        if spectral_radius <= 0:
            raise ValueError("spectral_radius must be positive.")
        if input_scaling <= 0:
            raise ValueError("input_scaling must be positive.")
        if ridge_param < 0:
            raise ValueError("ridge_param must be non-negative.")

        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.ridge_param = ridge_param
        self.random_state = random_state

        self.W_in = None
        self.W_res = None
        self.W_out = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _initialize_weights(self, input_dim: int):
        """Initializes input and reservoir weights."""
        self.W_in = (np.random.rand(self.reservoir_size, input_dim) * 2 - 1) * self.input_scaling
        self.W_res = np.random.rand(self.reservoir_size, self.reservoir_size) * 2 - 1

        # Normalize spectral radius
        try:
            radius = np.max(np.abs(np.linalg.eigvals(self.W_res)))
            if radius > 0:
                self.W_res *= self.spectral_radius / radius
            else: # Handle case where radius is 0 (e.g., all zeros matrix)
                self.W_res = np.zeros_like(self.W_res) # Or re-initialize, depending on desired behavior
        except np.linalg.LinAlgError:
            # Handle singular matrix case for eigvals
            print("Warning: Could not compute eigenvalues for reservoir matrix. Skipping spectral radius normalization.")
            pass # Keep W_res as is

    def fit(self, y: pd.Series, X: pd.DataFrame):
        """
        Fits the ESN model.

        Parameters:
        - y: Dependent variable (pandas Series).
        - X: Input features (pandas DataFrame).
        """
        if y.empty or X.empty:
            raise ValueError("Input data (y or X) cannot be empty.")
        if len(y) != len(X):
            raise ValueError("Length of y and X must be the same.")

        input_data = X.values
        n_samples, input_dim = input_data.shape

        if n_samples < 1:
            raise ValueError("Not enough samples to fit ESN.")

        self._initialize_weights(input_dim)

        # Collect reservoir states
        reservoir_states = np.zeros((n_samples, self.reservoir_size))
        current_state = np.zeros(self.reservoir_size) # Initial reservoir state

        for t in range(n_samples):
            input_t = input_data[t, :].reshape(-1, 1) # Ensure input is column vector

            # Compute preactivation
            preactivation = np.dot(self.W_in, input_t).flatten() + np.dot(self.W_res, current_state)

            # Update current state using leaking rate and tanh activation
            current_state = (1 - self.leaking_rate) * current_state + self.leaking_rate * np.tanh(preactivation)
            reservoir_states[t, :] = current_state

        # Add a bias term to the reservoir states for the output layer
        X_ridge = np.hstack([reservoir_states, np.ones((n_samples, 1))])
        y_target = y.values.reshape(-1, 1) # Ensure y is a column vector

        # Solve for output weights using Ridge regression (pseudo-inverse)
        # (X_ridge.T @ X_ridge + lambda * I) @ W_out = X_ridge.T @ y_target
        # W_out = inv(X_ridge.T @ X_ridge + lambda * I) @ X_ridge.T @ y_target
        try:
            self.W_out = pinv(X_ridge.T @ X_ridge + self.ridge_param * np.eye(self.reservoir_size + 1)) @ X_ridge.T @ y_target
        except np.linalg.LinAlgError as e:
            raise RuntimeError(f"Failed to compute pseudo-inverse during ESN fit: {e}. Try adjusting ridge_param.")

    def predict(self, X_new: pd.DataFrame):
        """
        Predicts using the fitted ESN model.

        Parameters:
        - X_new: New input features (pandas DataFrame).

        Returns:
        - Predicted values (numpy array).
        """
        if self.W_out is None or self.W_in is None or self.W_res is None:
            raise ValueError("ESN model has not been fitted yet. Call fit() first.")

        input_data = X_new.values
        n_samples, input_dim = input_data.shape

        # Re-initialize current_state for prediction, or use last state from training if continuing
        # For new data, assume cold start or provide a way to pass last state.
        # For simplicity, we'll assume a cold start for each prediction batch.
        current_state = np.zeros(self.reservoir_size)

        reservoir_states = np.zeros((n_samples, self.reservoir_size))
        for t in range(n_samples):
            input_t = input_data[t, :].reshape(-1, 1)
            preactivation = np.dot(self.W_in, input_t).flatten() + np.dot(self.W_res, current_state)
            current_state = (1 - self.leaking_rate) * current_state + self.leaking_rate * np.tanh(preactivation)
            reservoir_states[t, :] = current_state

        X_ridge = np.hstack([reservoir_states, np.ones((n_samples, 1))])
        return (X_ridge @ self.W_out).flatten()


class Forecast:
    """
    A class to encapsulate ARDLX and ESN forecasting models, including feature construction.
    """
    def __init__(self, y: pd.Series, z: pd.Series, p: int, q: int, delta: int):
        """
        :param y: Dependent variable (pandas Series).
        :param z: Exogenous variable (pandas Series).
        :param p: Number of autoregressive lags for y in ARDLX.
        :param q: Number of lags for z in ARDLX.
        :param delta: Lag for the binary dummy variable (d).
        """
        if not isinstance(y, pd.Series) or not isinstance(z, pd.Series):
            raise TypeError("y and z must be pandas Series.")
        if p < 0 or q < 0 or delta < 0:
            raise ValueError("p, q, and delta must be non-negative integers.")

        self.y = y
        self.z = z
        self.p = p
        self.q = q
        self.delta = delta
        self.X = None # Will store constructed features

        # Initialize models
        self.ardlx_ols = ARDLX(p, model_type='ols')
        self.ardlx_en = ARDLX(p, model_type='elastic_net')
        self.esn = ESN()

    def construct_features(self):
        """
        Constructs features for ARDLX and ESN models.
        Features include lagged z and a binary dummy 'd' based on lagged y.
        """
        # Create binary dummy variable 'd'
        # d is 1 if y shifted by delta is negative, 0 otherwise
        d = (self.y.shift(self.delta) < 0).astype(int)
        d.name = 'd' # Name the series for concatenation

        # Create lagged z variables
        z_lags = pd.DataFrame()
        for i in range(1, self.q + 1):
            z_lags[f'z_lag_{i}'] = self.z.shift(i)

        # Combine all features
        # Ensure indices align when concatenating
        combined_features = pd.concat([z_lags, d], axis=1)

        # Align y with the combined features and drop NaNs
        # The target variable y should also be aligned with the features
        full_data_aligned = pd.concat([self.y.rename('y_target'), combined_features], axis=1).dropna()

        if full_data_aligned.empty:
            raise ValueError("After constructing features and dropping NaNs, no valid data points remain.")

        self.y = full_data_aligned['y_target']
        self.X = full_data_aligned.drop(columns=['y_target'])

    def fit_models(self):
        """
        Fits the ARDLX (OLS and Elastic Net) and ESN models using the constructed features.
        """
        if self.X is None or self.y is None:
            raise ValueError("Features and target not constructed. Call construct_features() first.")
        if self.X.empty or self.y.empty:
            raise ValueError("Empty features or target after construction. Cannot fit models.")

        # ARDLX models
        self.ardlx_ols.fit(self.y, self.X)
        self.ardlx_en.fit(self.y, self.X)

        # ESN model requires lagged y in its input features
        # For ESN, the input X should include lagged y terms as well, similar to ARDLX's combined_X
        y_lags_for_esn = pd.DataFrame()
        for i in range(1, self.p + 1):
            y_lags_for_esn[f'y_lag_{i}'] = self.y.shift(i)

        # Combine lagged y with the existing exogenous features
        esn_X_input = pd.concat([y_lags_for_esn, self.X], axis=1)

        # Align y and esn_X_input and drop NaNs
        esn_full_data_aligned = pd.concat([self.y.rename('y_target'), esn_X_input], axis=1).dropna()

        if esn_full_data_aligned.empty:
            raise ValueError("Empty data for ESN after aligning features and target.")

        esn_y_aligned = esn_full_data_aligned['y_target']
        esn_X_aligned = esn_full_data_aligned.drop(columns=['y_target'])

        self.esn.fit(esn_y_aligned, esn_X_aligned)

    def predict(self, X_test: pd.DataFrame):
        """
        Predicts using the fitted models.

        Parameters:
        - X_test: New data for prediction (pandas DataFrame), must have the same structure as self.X.

        Returns:
        - Dictionary of predictions from 'ols', 'elastic_net', and 'esn'.
        """
        if self.ardlx_ols.model is None or self.ardlx_en.model is None or self.esn.W_out is None:
            raise ValueError("Models have not been fitted yet. Call fit_models() first.")

        # Ensure X_test has the correct columns for ARDLX
        # ARDLX expects lagged y and exogenous variables.
        # For a single-step-ahead forecast, X_test should contain the values that would be X_t for predicting y_{t+1}
        # This means X_test needs y_lag_1, ..., y_lag_p (which are y_t, ..., y_{t-p+1}) and z_lag_1, ..., z_lag_q (which are z_t, ..., z_{t-q+1})
        # and d_t (which is based on y_{t-delta}).

        # The current `X_test = forecaster.X.iloc[-1:].shift(-1).fillna(0)` in RollingWindowBacktest
        # is problematic for generating the correct features for the next step.
        # It's better to explicitly construct the next X_test based on the last observed actuals.

        # For demonstration purposes, assuming X_test is correctly structured for the next prediction.
        # The `predict` method of ARDLX and ESN will handle column matching based on their `feature_names` or internal structure.

        predictions = {
            'ols': self.ardlx_ols.predict(X_test),
            'elastic_net': self.ardlx_en.predict(X_test),
            'esn': self.esn.predict(X_test)
        }
        return predictions


class EvaluationMetrics:
    """
    Static class for computing common regression evaluation metrics.
    """
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).

        Parameters:
        - y_true: True values.
        - y_pred: Predicted values.

        Returns:
        - Dictionary of computed metrics.
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length.")
        if len(y_true) == 0:
            return {'MAE': np.nan, 'MSE': np.nan, 'RMSE': np.nan}

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        return {'MAE': mae, 'MSE': mse, 'RMSE': rmse}


class RollingWindowBacktest:
    """
    Performs rolling window backtesting for ARDLX and ESN models.
    """
    def __init__(self, window_size: int, horizon: int = 1):
        """
        :param window_size: Size of the training window.
        :param horizon: Forecast horizon (how many steps ahead to predict).
        """
        if window_size <= 0:
            raise ValueError("window_size must be a positive integer.")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer.")

        self.window_size = window_size
        self.horizon = horizon

    def run_backtest(self, y: pd.Series, z: pd.Series, p: int, q: int, delta: int):
        """
        Runs the rolling window backtest.

        Parameters:
        - y: Dependent variable (pandas Series).
        - z: Exogenous variable (pandas Series).
        - p: Number of autoregressive lags for y in ARDLX.
        - q: Number of lags for z in ARDLX.
        - delta: Lag for the binary dummy variable (d).

        Returns:
        - Tuple of (metrics, predictions, true_values).
        """
        if not isinstance(y, pd.Series) or not isinstance(z, pd.Series):
            raise TypeError("y and z must be pandas Series.")
        if len(y) != len(z):
            raise ValueError("y and z must have the same length.")
        if len(y) < self.window_size + self.horizon:
            raise ValueError(f"Data length ({len(y)}) too short for specified window_size ({self.window_size}) and horizon ({self.horizon}). "
                             "Need at least window_size + horizon data points.")

        predictions = {'ols': [], 'elastic_net': [], 'esn': []}
        true_values = []

        # The loop should go up to `len(y) - self.horizon` to ensure there's a target value
        for i in range(self.window_size, len(y) - self.horizon + 1):
            # Define training window
            train_y = y.iloc[i - self.window_size : i]
            train_z = z.iloc[i - self.window_size : i]

            # Initialize and fit the forecaster on the training window
            forecaster = Forecast(train_y, train_z, p, q, delta)

            try:
                forecaster.construct_features()
                forecaster.fit_models()
            except ValueError as e:
                print(f"Skipping iteration {i} due to data/model fitting issue: {e}")
                # Append NaNs for this iteration if models cannot be fitted
                for model_type in predictions:
                    predictions[model_type].append(np.nan)
                true_values.append(y.iloc[i + self.horizon - 1]) # Still record actual
                continue
            except RuntimeError as e: # Catch ESN specific errors
                print(f"Skipping iteration {i} due to ESN model fitting issue: {e}")
                for model_type in predictions:
                    predictions[model_type].append(np.nan)
                true_values.append(y.iloc[i + self.horizon - 1])
                continue

            # Prepare data for prediction (next step's features)
            # This is the most complex part: constructing X_test for horizon-step ahead forecast.
            # For horizon=1, X_test should be based on the last point of the training window.
            # For horizon > 1, this needs a recursive approach or a direct multi-step forecast from models.
            # Assuming horizon=1 for simplicity as per original example's `shift(-1)`

            # To predict y[i + horizon - 1], we need features corresponding to time i + horizon - 1.
            # This means y_lag_k should be y[i + horizon - 1 - k]
            # z_lag_k should be z[i + horizon - 1 - k]
            # d should be based on y[i + horizon - 1 - delta]

            # For a rolling window, we use the *actual* data up to `i + horizon - 1 - 1` to form the features for `i + horizon - 1`.
            # This is "pseudo-real-time" as it uses future actuals to form features.
            # A true real-time forecast would only use information available at time `i-1` to predict `i + horizon - 1`.

            # Let's simplify X_test construction for this example to avoid look-ahead bias for now.
            # The original code's `forecaster.X.iloc[-1:].shift(-1).fillna(0)` is problematic.
            # A better way is to construct the features for the *next* time step based on the last available data.

            # Construct features for the next step (i.e., for predicting y.iloc[i])
            # This requires y[i-1], y[i-2]... and z[i-1], z[i-2]... and d based on y[i-1-delta]
            # We need to create a dummy DataFrame for the next prediction based on available data

            # Let's assume we are predicting y[i + self.horizon - 1]
            # The features for this prediction need to be based on data up to index `i + self.horizon - 2`

            # Create a mock DataFrame for the next prediction's features
            # This needs to reflect the structure of forecaster.X

            # Get the actual values that would be used to form the features for the prediction point
            # For predicting y[i + self.horizon - 1], the features are based on y and z values
            # up to index i + self.horizon - 2.

            # This is a critical point: how to generate X_test for the next horizon step.
            # If the models are truly one-step-ahead, then for multi-step, we need iterative prediction.
            # Given the existing `predict` method, it expects a pre-formed X_test.
            # The simplest interpretation for `X_test = forecaster.X.iloc[-1:].shift(-1).fillna(0)`
            # is trying to get the features for the very next step after the training data ends.

            # Correct approach for rolling window:
            # Train on data up to `i-1`. Predict `y[i]`.
            # The features `X_test` for predicting `y[i]` would involve `y[i-1], y[i-2], ...` and `z[i-1], z[i-2], ...`
            # These are available at time `i-1`.

            # Let's reformulate: `train_y` and `train_z` are up to `i-1`.
            # We want to predict `y[i]`.
            # So, the features for `y[i]` need `y[i-1], y[i-2]...` and `z[i-1], z[i-2]...` and `d` based on `y[i-1-delta]`.

            # Construct features for the *next* point (at index `i`) using the full `y` and `z` series
            # This is still somewhat "cheating" if horizon > 1 and we use future `y` or `z`
            # For horizon=1, it's fine.

            current_idx = i + self.horizon - 1
            if current_idx >= len(y):
                continue # No actual value to compare against

            # Create a single row DataFrame for the features at `current_idx`
            # This requires values from `y` and `z` up to `current_idx - 1`

            # Lagged y terms for ARDLX and ESN
            next_y_lags = {}
            for k in range(1, p + 1):
                next_y_lags[f'y_lag_{k}'] = [y.iloc[current_idx - k]] if (current_idx - k) >= 0 else [np.nan]

            # Lagged z terms
            next_z_lags = {}
            for k in range(1, q + 1):
                next_z_lags[f'z_lag_{k}'] = [z.iloc[current_idx - k]] if (current_idx - k) >= 0 else [np.nan]

            # Binary dummy 'd'
            next_d_val = [1 if y.iloc[current_idx - delta] < 0 else 0] if (current_idx - delta) >= 0 else [np.nan]

            # Combine into a single DataFrame for X_test
            X_test_dict = {**next_y_lags, **next_z_lags, 'd': next_d_val}
            X_test_row = pd.DataFrame(X_test_dict, index=[0]) # Single row DataFrame

            # Ensure column order matches training data for ARDLX
            # This requires the feature names from the fitted forecaster
            if forecaster.ardlx_ols.feature_names:
                X_test_row_ordered_ardlx = X_test_row[forecaster.ardlx_ols.feature_names]
            else:
                X_test_row_ordered_ardlx = X_test_row # Fallback, might cause issues

            # For ESN, the input needs to match its training input structure
            # This means it needs lagged y as well, which are already in `next_y_lags`
            esn_feature_names = forecaster.esn.W_in.shape[1] if forecaster.esn.W_in is not None else None
            # This requires more careful handling of ESN's X_test as it's a numpy array.
            # For simplicity, we'll assume the structure is consistent.

            # Perform predictions
            preds = forecaster.predict(X_test_row_ordered_ardlx) # Use ARDLX's feature set for all

            for model_type in predictions:
                predictions[model_type].append(preds[model_type][0])
            true_values.append(y.iloc[current_idx])

        # Filter out NaNs from predictions and true_values before computing metrics
        # This is crucial because some iterations might have failed to produce valid forecasts
        valid_indices_mask = ~np.isnan(predictions['ols']) & ~np.isnan(predictions['elastic_net']) & ~np.isnan(predictions['esn'])

        filtered_predictions = {
            model_type: np.array(preds)[valid_indices_mask] for model_type, preds in predictions.items()
        }
        filtered_true_values = np.array(true_values)[valid_indices_mask]

        if len(filtered_true_values) == 0:
            print("No valid predictions generated for evaluation.")
            return {}, {}, {} # Return empty results

        metrics = {model: EvaluationMetrics.compute_metrics(filtered_true_values, filtered_predictions[model])
                   for model in filtered_predictions}

        return metrics, filtered_predictions, filtered_true_values


# --- Example Usage Section ---

def run_all_examples():
    """
    Runs examples for all classes: Plotting, Descriptive Statistics, and Forecasting.
    """
    print("--- Running Unified Data Analysis and Forecasting Examples ---")

    # --- 1. Plotting Examples ---
    print("\n=== Plotting Examples ===")
    # Create sample data for TimeSeriesPlotter
    time_data = pd.to_datetime(pd.date_range(start='2020-01-01', periods=100, freq='M'))
    series_data1 = np.random.rand(100) * 100 + np.sin(np.arange(100)/10) * 20
    series_data2 = np.random.rand(100) * 50 + 50 + np.cos(np.arange(100)/5) * 15
    series_data3 = np.random.rand(100) * 30 + 70

    df_plot = pd.DataFrame({
        'time': time_data,
        'series1': series_data1,
        'series2': series_data2,
        'series3': series_data3,
    })

    plotter = TimeSeriesPlotter(df_plot, title="Sample Time Series Data", xlabel="Date", ylabel="Value")

    print("\nPlotting Single Series:")
    plotter.plot_single_series('series1', secondary_series='series2', secondary_ylabel='Secondary Value')

    print("\nPlotting Multiple Series:")
    plotter.plot_multiple_series(series_names=['series1', 'series3'])

    print("\nPlotting with Shaded Recession Periods:")
    recession_periods = [(pd.to_datetime('2020-06-01'), pd.to_datetime('2020-09-01')),
                         (pd.to_datetime('2021-03-01'), pd.to_datetime('2021-07-01'))]
    recession_colors = ['gray', 'red']
    plotter.plot_with_shading('series1', recession_periods, recession_colors)

    print("\nPlotting Rolling Statistics:")
    plotter.plot_rolling_statistics('series1', window_size=12)

    print("\nPlotting Rolling Correlation:")
    plotter.plot_rolling_correlation(window_size=12)

    # Example for TimeSeriesWithHistogram
    print("\nPlotting Time Series with Histogram and Density:")
    df_hist = pd.DataFrame({
        'Value': np.random.normal(loc=50, scale=10, size=500),
        'Date': pd.date_range(start='2023-01-01', periods=500)
    }).set_index('Date')
    hist_plotter = TimeSeriesWithHistogram(df_hist)
    hist_plotter.plot(series_name='Value', bins=25, kde=True,
                      title='Value Series with Distribution',
                      xlabel='Date', ylabel='Observation', background_color='#f0f0f0')


    # --- 2. Descriptive Statistics & PCA Examples ---
    print("\n=== Descriptive Statistics & PCA Examples ===")
    data_analyzer = {
        'F1': np.random.randn(100),
        'F2': np.random.randn(100) * 2,
        'F3': np.random.rand(100) * 10,
        'Group': np.random.choice(['A', 'B', 'C'], size=100),
        'Category': np.random.choice(['X', 'Y'], size=100)
    }
    df_analyzer = pd.DataFrame(data_analyzer)

    analyzer = DataAnalyzer(df_analyzer)

    print("\nGenerating Descriptive Statistics:")
    analyzer.descriptive_statistics(save_as='csv')

    print("\nGenerating Correlation Matrix:")
    analyzer.correlation_matrix(save_as='png')

    print("\nPerforming Principal Component Analysis:")
    analyzer.principal_component_analysis(n_components=2, save_as='pdf')

    print("\nPerforming MANOVA:")
    try:
        analyzer.manova(dependent_vars=['F1', 'F2'], independent_var='Group', save_as='txt')
    except ValueError as e:
        print(f"MANOVA example error: {e}")

    # TimeSeriesBootstrap Example
    print("\nPerforming Time Series Distributional Fit Test (Bootstrap):")
    bootstrap_data = np.random.normal(loc=0, scale=1, size=200) # Sample normal data
    ts_bootstrap = TimeSeriesBootstrap(bootstrap_data, num_samples=50, block_size=10)
    ts_bootstrap.print_results_table()


    # --- 3. Forecasting Models Examples ---
    print("\n=== Forecasting Models Examples ===")

    # Example data for forecasting
    np.random.seed(42)
    forecast_data = np.cumsum(np.random.normal(0, 0.5, 150)) + np.sin(np.linspace(0, 20, 150)) * 5
    forecast_data_series = pd.Series(forecast_data)

    print("\n--- MDL_AR_Model and AR1_Model Backtesting ---")
    # Backtesting with expanding sample
    print("\nBacktesting with Expanding Sample:")
    backtesting_expanding = Backtesting(forecast_data, sample_type='expanding')
    backtesting_expanding.run_backtest(forecast_steps=1)
    backtesting_expanding.plot_forecasts_and_errors()

    # Backtesting with rolling sample
    print("\nBacktesting with Rolling Sample (window_size=30):")
    backtesting_rolling = Backtesting(forecast_data, sample_type='rolling', window_size=30)
    backtesting_rolling.run_backtest(forecast_steps=1)
    backtesting_rolling.plot_forecasts_and_errors()

    print("\n--- Real-Time Forecasting ---")
    real_time_forecasting = RealTimeForecasting(forecast_data)
    real_time_forecasting.compute_forecasts(forecast_steps=1)

    print("\n--- UnifiedForecaster (AR and SVR) ---")
    # AR Forecast with UnifiedForecaster
    print("\nUnifiedForecaster - AR Models:")
    uf_ar = UnifiedForecaster(forecast_data, R=20, lag_criterion='BIC', verbose=True)
    uf_ar.plot()
    print("Evaluation Metrics for AR Models:")
    print(uf_ar.evaluate())

    # SVR Forecast with UnifiedForecaster
    print("\nUnifiedForecaster - SVR Model:")
    svr_params = {'kernel': 'rbf', 'C': 10, 'epsilon': 0.1, 'gamma': 'scale'}
    uf_svr = UnifiedForecaster(forecast_data, R=20, use_svr=True, svr_params=svr_params, verbose=True)
    uf_svr.plot()
    print("Evaluation Metrics for SVR Model:")
    print(uf_svr.evaluate())

    print("\n--- EnhancedRollingSVRForecaster (Detailed SVR) ---")
    # Generate synthetic data with seasonality and noise for SVR
    t_svr = np.linspace(0, 8*np.pi, 500)
    data_svr = np.sin(t_svr) + 0.5*np.cos(2*t_svr) + np.random.normal(0, 0.2, len(t_svr))

    forecaster_svr = EnhancedRollingSVRForecaster(
        window_size=30,
        kernel='rbf',
        C=10,
        verbose=True
    )

    # Data preparation
    try:
        X_train_svr, X_test_svr, y_train_svr, y_test_svr = forecaster_svr.prepare_data(data_svr, train_size=0.8)
    except ValueError as e:
        print(f"Enhanced SVR Data error: {e}")
        X_train_svr, X_test_svr, y_train_svr, y_test_svr = None, None, None, None # Set to None to prevent further errors

    if X_train_svr is not None:
        # Hyperparameter tuning
        print("\nSVR Hyperparameter Tuning:")
        best_params_svr = forecaster_svr.tune_hyperparameters()
        print(f"Best SVR parameters: {best_params_svr}")

        # Training
        forecaster_svr.fit()

        # Full series predictions
        predictions_svr, actuals_svr = forecaster_svr.rolling_predict(data_svr)

        # Evaluation
        metrics_svr = forecaster_svr.evaluate_rolling(data_svr)
        print("\nEnhanced SVR Performance Metrics (on full series):")
        for k, v in metrics_svr.items():
            print(f"{k+':':<8} {v:.4f}")

        # Visualization
        forecaster_svr.visualize(data_svr, predictions_svr)
    else:
        print("Skipping EnhancedRollingSVRForecaster due to data preparation issues.")


    print("\n--- ARDLX and ESN Backtesting ---")
    # Example data for ARDLX and ESN
    np.random.seed(42)
    y_ardlx = pd.Series(np.cumsum(np.random.randn(200))) # Dependent variable
    z_ardlx = pd.Series(np.random.randn(200)) # Exogenous variable

    # Define parameters for ARDLX and ESN
    p_ardlx = 2 # Lags for y
    q_ardlx = 1 # Lags for z
    delta_ardlx = 1 # Lag for dummy 'd'

    # Rolling window backtest
    window_size_ardlx = 50
    horizon_ardlx = 1 # Predict 1 step ahead

    backtester_ardlx = RollingWindowBacktest(window_size=window_size_ardlx, horizon=horizon_ardlx)
    print(f"\nRolling Window Backtest for ARDLX and ESN (Window: {window_size_ardlx}, Horizon: {horizon_ardlx}):")
    try:
        metrics_ardlx, predictions_ardlx, true_values_ardlx = backtester_ardlx.run_backtest(y_ardlx, z_ardlx, p_ardlx, q_ardlx, delta_ardlx)

        if metrics_ardlx:
            print("\nEvaluation Metrics:")
            for model, model_metrics in metrics_ardlx.items():
                print(f"  {model.capitalize()} Model:")
                for metric_name, value in model_metrics.items():
                    print(f"    {metric_name}: {value:.4f}")

            # Plotting the backtest results
            plt.figure(figsize=(14, 7))
            plt.plot(true_values_ardlx, label='Actual', color='blue', linewidth=2)
            plt.plot(predictions_ardlx['ols'], '--', label='ARDLX (OLS) Forecast', color='orange')
            plt.plot(predictions_ardlx['elastic_net'], '-.', label='ARDLX (Elastic Net) Forecast', color='green')
            plt.plot(predictions_ardlx['esn'], ':', label='ESN Forecast', color='purple')
            plt.title("Rolling Window Backtest: Actual vs. Forecasts")
            plt.xlabel("Time Index")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.show()
        else:
            print("No metrics or predictions generated for ARDLX/ESN backtest.")

    except ValueError as e:
        print(f"ARDLX/ESN Backtest error: {e}")
    except RuntimeError as e:
        print(f"ARDLX/ESN Backtest runtime error: {e}")

    print("\n--- Adaptive Learning Forecaster ---")
    # Define robust forecasting models for AdaptiveLearningForecaster
    # (Already defined globally above for clarity)

    # Run the example function
    run_forecasting_example() # This function is part of the original adaptive_learning_forecaster.py

    print("\n--- End of Unified Data Analysis and Forecasting Examples ---")


# Example usage function for AdaptiveLearningForecaster (from original file)
def run_forecasting_example():
    # Set random seed for reproducibility
    np.random.seed(42)

    # --- Continuous Mode Example ---
    print("\n=== Adaptive Learning Forecaster: Continuous Mode Example ===")
    # Generate synthetic data: noisy sine wave
    t = np.linspace(0, 100, 200)
    continuous_data = np.sin(0.1 * t) + np.random.normal(0, 0.2, 200)

    # Initialize forecaster
    forecaster_cont = AdaptiveLearningForecaster(
        models=[moving_average_model, momentum_model],
        mode="continuous",
        learning_rate=0.1,
        window_size=10
    )

    # Run forecasting
    forecasts_cont = []
    weights_history_cont = []
    start_idx = 20 # Start forecasting after some initial data
    for i in range(start_idx, len(continuous_data)):
        # Ensure enough data is passed to the forecast method
        if len(continuous_data[:i]) == 0:
            forecasts_cont.append(np.nan) # Cannot forecast with empty data
            weights_history_cont.append(forecaster_cont.get_weights().copy())
            continue

        forecast = forecaster_cont.forecast(continuous_data[:i])
        forecasts_cont.append(forecast)

        # Ensure there's an actual value to update with
        if i < len(continuous_data):
            forecaster_cont.update(continuous_data[i])
        weights_history_cont.append(forecaster_cont.get_weights().copy())

    # Filter out NaNs for MSE calculation
    valid_forecasts_cont = np.array(forecasts_cont)
    valid_actuals_cont = continuous_data[start_idx:]

    # Align valid forecasts and actuals
    min_len = min(len(valid_forecasts_cont), len(valid_actuals_cont))
    valid_forecasts_cont = valid_forecasts_cont[:min_len]
    valid_actuals_cont = valid_actuals_cont[:min_len]

    errors = [(f - a) ** 2 for f, a in zip(valid_forecasts_cont, valid_actuals_cont) if np.isfinite(f) and np.isfinite(a)]
    mse_cont = np.mean(errors) if errors else np.nan
    print(f"Continuous Mode MSE: {mse_cont:.4f}")

    # Plot results
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t[start_idx:start_idx+len(valid_actuals_cont)], valid_actuals_cont, label="Actual", color="blue")
    plt.plot(t[start_idx:start_idx+len(valid_forecasts_cont)], valid_forecasts_cont, label="Forecast", color="orange", linestyle="--")
    plt.title("Continuous Forecasting: Noisy Sine Wave")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    plt.subplot(2, 1, 2)
    weights_history_cont = np.array(weights_history_cont)
    if weights_history_cont.shape[0] > 0:
        plt.plot(t[start_idx:start_idx+weights_history_cont.shape[0]], weights_history_cont[:, 0], label="Moving Average Weight")
        plt.plot(t[start_idx:start_idx+weights_history_cont.shape[0]], weights_history_cont[:, 1], label="Momentum Weight")
        plt.title("Model Weights Over Time")
        plt.xlabel("Time")
        plt.ylabel("Weight")
        plt.legend()

    plt.tight_layout()
    plt.show()

    # --- Binary Mode Example ---
    print("\n=== Adaptive Learning Forecaster: Binary Mode Example ===")
    random_walk = np.cumsum(np.random.normal(0, 1, 200))
    binary_data = np.sign(np.diff(random_walk))
    binary_data = np.concatenate(([1], binary_data)) # Prepend a value to match length

    forecaster_bin = AdaptiveLearningForecaster(
        models=[majority_vote_model, momentum_model],
        mode="binary",
        learning_rate=0.1,
        window_size=10,
        threshold=0.0
    )

    forecasts_bin = []
    weights_history_bin = []
    start_idx = 20
    for i in range(start_idx, len(binary_data)):
        if len(binary_data[:i]) == 0:
            forecasts_bin.append(np.nan)
            weights_history_bin.append(forecaster_bin.get_weights().copy())
            continue

        forecast = forecaster_bin.forecast(binary_data[:i])
        forecasts_bin.append(forecast)

        if i < len(binary_data):
            forecaster_bin.update(binary_data[i])
        weights_history_bin.append(forecaster_bin.get_weights().copy())

    valid_forecasts_bin = np.array(forecasts_bin)
    valid_actuals_bin = binary_data[start_idx:]

    min_len_bin = min(len(valid_forecasts_bin), len(valid_actuals_bin))
    valid_forecasts_bin = valid_forecasts_bin[:min_len_bin]
    valid_actuals_bin = valid_actuals_bin[:min_len_bin]

    accuracy_bin = np.mean([1 if f == a else 0 for f, a in zip(valid_forecasts_bin, valid_actuals_bin) if np.isfinite(f) and np.isfinite(a)])
    print(f"Binary Mode Accuracy: {accuracy_bin:.4f}")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.step(range(start_idx, start_idx+len(valid_actuals_bin)), valid_actuals_bin, label="Actual", color="blue")
    plt.step(range(start_idx, start_idx+len(valid_forecasts_bin)), valid_forecasts_bin, label="Forecast", color="orange", linestyle="--")
    plt.title("Binary Forecasting: Sign of Random Walk Changes")
    plt.xlabel("Time Step")
    plt.ylabel("Value (-1 or +1)")
    plt.legend()

    plt.subplot(2, 1, 2)
    weights_history_bin = np.array(weights_history_bin)
    if weights_history_bin.shape[0] > 0:
        plt.plot(range(start_idx, start_idx+weights_history_bin.shape[0]), weights_history_bin[:, 0], label="Majority Vote Weight")
        plt.plot(range(start_idx, start_idx+weights_history_bin.shape[0]), weights_history_bin[:, 1], label="Momentum Weight")
        plt.title("Model Weights Over Time")
        plt.xlabel("Time Step")
        plt.ylabel("Weight")
        plt.legend()

    plt.tight_layout()
    plt.show()


# Run all examples when the script is executed
if __name__ == "__main__":
    run_all_examples()
