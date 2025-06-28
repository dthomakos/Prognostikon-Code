#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-inferential-speculator/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
# -------------------------------------------------------------------------------------

from Cox_Trading_System import CoxProbabilisticTrader_V2
from Cox_Trading_System import WalkForwardOptimizer
import numpy as np
from itertools import product
import yfinance as yf
from scipy.stats import norm
from collections import defaultdict
import matplotlib.pyplot as plt

# Download data
data = yf.download("SPY", start="2022-01-01", end="2025-06-01")
prices = data['Close'].values

# Define parameter grid for optimization
param_grid = {
    'lookback_window': [7, 14, 21],
    'entropy_threshold': [0.1, 0.15, 0.2],
    'prior_belief': [0.3, 0.5],
    'gamma': [0.5, 0.75, 1.0],
    'momentum_threshold': [-0.02, -0.01, 0.0],
    'sharpe_threshold': [0.4, 0.5, 0.6],
    'utility_threshold': [-0.1, 0.0, 0.1],
    'trade_threshold': [0.4, 0.5, 0.6, 0.7]
}

# Initialize WFO
wfo = WalkForwardOptimizer(
    trader_class=CoxProbabilisticTrader_V2,
    param_grid=param_grid,
    lookback_window=21,
    train_window=504,
    test_window=177
)

# Run optimization
results = wfo.optimize(prices)

# Print summary
for res in results:
    print(f"Train: {res['train_start']} to {res['train_end']}, Test: {res['test_start']} to {res['test_end']}")
    print(f"Best Params: {res['best_params']}")
    print(f"Test Cumulative Return: {res['test_performance']['cumulative_return']:.2%}")
    print(f"Test Sharpe Ratio: {res['test_performance']['sharpe_ratio']:.2f}")
    print("-" * 50)

# from Cox_Trading_System_V1 import CoxProbabilisticTrader_V1, momentum_func, volatility_func, returns_func, trend_func
# from WFO_Procedure import WalkForwardOptimizer
# import yfinance as yf
#
# # Historical price data
# data = yf.download("SPY", start="2018-01-01", end="2024-06-01")
# prices = data['Close'].values
#
# # Static feature definitions (assumed constant across WFO)
# proposition_system = {
#     'momentum': (0.0, 0.01),
#     'volatility': (0.015, 0.005),
#     'returns': (0.0, 0.01),
#     'trend': (0.0, 0.01)
# }
#
# feature_functions = {
#     'momentum': momentum_func,
#     'volatility': volatility_func,
#     'returns': returns_func,
#     'trend': trend_func
# }
#
# # Parameters to optimize (must match V1 signature)
# param_grid = {
#     'lookback_window': [63],
#     'entropy_threshold': [0.1, 0.15],
#     'prior_belief': [0.3],
#     'state_k': [0.5, 0.65],
#     'trade_threshold': [0.25, 0.35]
# }
#
# # Custom wrapper for WFO that injects the static arguments
# class V1Wrapper:
#     def __init__(self, **kwargs):
#         self.trader = CoxProbabilisticTrader_V1(
#             proposition_system=proposition_system,
#             feature_functions=feature_functions,
#             **kwargs
#         )
#
#     def backtest(self, prices):
#         return self.trader.backtest(prices)
#
# # Initialize WFO
# wfo = WalkForwardOptimizer(
#     trader_class=V1Wrapper,
#     param_grid=param_grid,
#     lookback_window=63,
#     train_window=252,
#     test_window=63
# )
#
# # Run optimization
# results = wfo.optimize(prices)
#
# # Print results
# for res in results:
#     print(f"Train {res['train_start']} to {res['train_end']}, Test {res['test_start']} to {res['test_end']}")
#     print(f"Best Params: {res['best_params']}")
#     print(f"Test Cumulative Return: {res['test_performance']['cumulative_return']:.2%}")
#     print(f"Test Sharpe Ratio: {res['test_performance']['sharpe_ratio']:.2f}")
#     print("-" * 50)
