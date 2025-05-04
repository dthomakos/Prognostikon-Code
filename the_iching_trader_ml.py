#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/thomakos/the-i-ching-trader/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
# -------------------------------------------------------------------------------------


# ----------------------------------------------------------------------------------
# Define the dictionary of hexagram meaning; note that action is also "deleveraging"
# ----------------------------------------------------------------------------------
HEXAGRAM_MEANINGS = {
    1: {'name': 'Creative Heaven', 'action': 1.0, 'confidence': 0.95, 'regime': 'bull', 'element': 'metal',
        'comment': "Strong upward momentum - maximum long exposure"},
    2: {'name': 'Receptive Earth', 'action': -1.0, 'confidence': 0.9, 'regime': 'bear', 'element': 'earth',
        'comment': "Market consolidation - full risk-off positioning"},
    3: {'name': 'Difficulty at Beginning', 'action': 0.2, 'confidence': 0.6, 'regime': 'transition', 'element': 'water',
        'comment': "Initial volatility - small exploratory positions"},
    4: {'name': 'Youthful Folly', 'action': 0.0, 'confidence': 0.4, 'regime': 'choppy', 'element': 'mountain',
        'comment': "Unclear trends - stay in cash and observe"},
    5: {'name': 'Waiting', 'action': 0.3, 'confidence': 0.65, 'regime': 'accumulation', 'element': 'water',
        'comment': "Fundamental improvement - gradual accumulation"},
    6: {'name': 'Conflict', 'action': -0.7, 'confidence': 0.75, 'regime': 'correction', 'element': 'heaven',
        'comment': "Heightened volatility - hedge positions"},
    7: {'name': 'Army', 'action': 0.5, 'confidence': 0.8, 'regime': 'breakout', 'element': 'earth',
        'comment': "Institutional buying - follow smart money flow"},
    8: {'name': 'Holding Together', 'action': 0.4, 'confidence': 0.7, 'regime': 'rally', 'element': 'water',
        'comment': "Sector rotation opportunities - diversify holdings"},
    9: {'name': 'Small Taming', 'action': 0.25, 'confidence': 0.6, 'regime': 'range', 'element': 'wind',
        'comment': "Range-bound market - option strategies preferred"},
    10: {'name': 'Treading', 'action': -0.3, 'confidence': 0.65, 'regime': 'overbought', 'element': 'lake',
         'comment': "Technical resistance - take partial profits"},
    11: {'name': 'Peace', 'action': 0.8, 'confidence': 0.85, 'regime': 'expansion', 'element': 'earth',
         'comment': "Harmonious growth - maximize long exposure"},
    12: {'name': 'Standstill', 'action': -0.8, 'confidence': 0.8, 'regime': 'contraction', 'element': 'heaven',
         'comment': "Market stagnation - defensive positioning"},
    13: {'name': 'Fellowship', 'action': 0.6, 'confidence': 0.75, 'regime': 'momentum', 'element': 'fire',
         'comment': "Strong consensus moves - trend following"},
    14: {'name': 'Great Possession', 'action': 0.9, 'confidence': 0.9, 'regime': 'bull', 'element': 'fire',
         'comment': "Wealth accumulation phase - aggressive longs"},
    15: {'name': 'Modesty', 'action': 0.1, 'confidence': 0.55, 'regime': 'correction', 'element': 'mountain',
         'comment': "Market humility - reduce leverage"},
    16: {'name': 'Enthusiasm', 'action': 0.7, 'confidence': 0.8, 'regime': 'breakout', 'element': 'thunder',
         'comment': "Ebullient markets - momentum chasing"},
    17: {'name': 'Following', 'action': 0.65, 'confidence': 0.78, 'regime': 'trend', 'element': 'lake',
         'comment': "Trend continuation - add to winners"},
    18: {'name': 'Work on the Decayed', 'action': -0.6, 'confidence': 0.7, 'regime': 'repair', 'element': 'wind',
         'comment': "Structural weaknesses - short weak sectors"},
    19: {'name': 'Approach', 'action': 0.4, 'confidence': 0.65, 'regime': 'recovery', 'element': 'earth',
         'comment': "Early cycle phase - value investing"},
    20: {'name': 'Contemplation', 'action': 0.0, 'confidence': 0.5, 'regime': 'neutral', 'element': 'wind',
         'comment': "Market indecision - maintain positions"},
    21: {'name': 'Biting Through', 'action': 0.55, 'confidence': 0.72, 'regime': 'breakout', 'element': 'fire',
         'comment': "Breaking resistance - directional plays"},
    22: {'name': 'Grace', 'action': 0.3, 'confidence': 0.6, 'regime': 'aesthetic', 'element': 'mountain',
         'comment': "Style rotation - thematic investing"},
    23: {'name': 'Splitting Apart', 'action': -0.85, 'confidence': 0.88, 'regime': 'collapse', 'element': 'mountain',
         'comment': "Market breakdown - capital preservation"},
    24: {'name': 'Return', 'action': 0.75, 'confidence': 0.82, 'regime': 'reversal', 'element': 'earth',
         'comment': "Mean reversion - counter-trend plays"},
    25: {'name': 'Innocence', 'action': 0.15, 'confidence': 0.58, 'regime': 'neutral', 'element': 'heaven',
         'comment': "Uncertain fundamentals - small bets"},
    26: {'name': 'Great Taming', 'action': 0.5, 'confidence': 0.7, 'regime': 'stability', 'element': 'mountain',
         'comment': "Volatility contraction - long convexity"},
    27: {'name': 'Nourishment', 'action': 0.7, 'confidence': 0.85, 'regime': 'growth', 'element': 'thunder',
         'comment': "Fundamental strength - growth stocks"},
    28: {'name': 'Great Excess', 'action': -0.9, 'confidence': 0.9, 'regime': 'bubble', 'element': 'lake',
         'comment': "Market overextension - prepare reversal"},
    29: {'name': 'Abysmal Water', 'action': -0.6, 'confidence': 0.75, 'regime': 'crisis', 'element': 'water',
         'comment': "Liquidity risks - increase cash"},
    30: {'name': 'Clinging Fire', 'action': 0.8, 'confidence': 0.85, 'regime': 'momentum', 'element': 'fire',
         'comment': "Strong trends - ride the wave"},
    31: {'name': 'Influence', 'action': 0.45, 'confidence': 0.68, 'regime': 'sentiment', 'element': 'lake',
         'comment': "Social proof trading - follow flows"},
    32: {'name': 'Duration', 'action': 0.6, 'confidence': 0.75, 'regime': 'trend', 'element': 'thunder',
         'comment': "Established trends - stay invested"},
    33: {'name': 'Retreat', 'action': -0.7, 'confidence': 0.8, 'regime': 'correction', 'element': 'mountain',
         'comment': "Smart money exit - reduce exposure"},
    34: {'name': 'Great Power', 'action': 0.85, 'confidence': 0.88, 'regime': 'strength', 'element': 'thunder',
         'comment': "Market force - maximize leverage"},
    35: {'name': 'Progress', 'action': 0.65, 'confidence': 0.78, 'regime': 'expansion', 'element': 'fire',
         'comment': "Steady growth - compound positions"},
    36: {'name': 'Darkening Light', 'action': -0.75, 'confidence': 0.82, 'regime': 'decline', 'element': 'earth',
         'comment': "Hidden risks - defensive rotation"},
    37: {'name': 'Family', 'action': 0.2, 'confidence': 0.55, 'regime': 'stable', 'element': 'wind',
         'comment': "Sector correlations - pairs trading"},
    38: {'name': 'Opposition', 'action': -0.5, 'confidence': 0.65, 'regime': 'divergence', 'element': 'fire',
         'comment': "Market fractures - relative value"},
    39: {'name': 'Obstruction', 'action': -0.4, 'confidence': 0.6, 'regime': 'resistance', 'element': 'water',
         'comment': "Technical barriers - range trading"},
    40: {'name': 'Deliverance', 'action': 0.75, 'confidence': 0.83, 'regime': 'recovery', 'element': 'thunder',
         'comment': "Stress relief - buy the dip"},
    41: {'name': 'Decrease', 'action': -0.6, 'confidence': 0.72, 'regime': 'contraction', 'element': 'mountain',
         'comment': "Reduced liquidity - lower exposure"},
    42: {'name': 'Increase', 'action': 0.8, 'confidence': 0.85, 'regime': 'expansion', 'element': 'wind',
         'comment': "Favorable conditions - scale in"},
    43: {'name': 'Breakthrough', 'action': 0.9, 'confidence': 0.9, 'regime': 'breakout', 'element': 'lake',
         'comment': "Key level breach - momentum entry"},
    44: {'name': 'Coming to Meet', 'action': 0.25, 'confidence': 0.58, 'regime': 'confluence', 'element': 'wind',
         'comment': "Multiple timeframes align - confirm trade"},
    45: {'name': 'Gathering', 'action': 0.5, 'confidence': 0.7, 'regime': 'consolidation', 'element': 'earth',
         'comment': "Capital concentration - follow big players"},
    46: {'name': 'Pushing Upward', 'action': 0.7, 'confidence': 0.8, 'regime': 'bull', 'element': 'wood',
         'comment': "Sustainable uptrend - pyramiding"},
    47: {'name': 'Oppression', 'action': -0.85, 'confidence': 0.87, 'regime': 'distress', 'element': 'lake',
         'comment': "Liquidity crunch - risk off"},
    48: {'name': 'The Well', 'action': 0.3, 'confidence': 0.62, 'regime': 'value', 'element': 'water',
         'comment': "Deep value opportunities - contrarian plays"},
    49: {'name': 'Revolution', 'action': 0.6, 'confidence': 0.75, 'regime': 'change', 'element': 'fire',
         'comment': "Paradigm shift - disruptive themes"},
    50: {'name': 'The Cauldron', 'action': 0.4, 'confidence': 0.65, 'regime': 'transformation', 'element': 'fire',
         'comment': "Fundamental change - special situations"},
    51: {'name': 'Arousing', 'action': 0.9, 'confidence': 0.92, 'regime': 'volatility', 'element': 'thunder',
         'comment': "Event-driven moves - gamma trading"},
    52: {'name': 'Keeping Still', 'action': -0.3, 'confidence': 0.6, 'regime': 'stagnation', 'element': 'mountain',
         'comment': "Low volatility - theta strategies"},
    53: {'name': 'Gradual Progress', 'action': 0.55, 'confidence': 0.73, 'regime': 'grind', 'element': 'wind',
         'comment': "Steady uptrend - swing trading"},
    54: {'name': 'Marrying Maiden', 'action': 0.1, 'confidence': 0.52, 'regime': 'speculation', 'element': 'thunder',
         'comment': "Meme stock behavior - small speculation"},
    55: {'name': 'Abundance', 'action': 0.85, 'confidence': 0.89, 'regime': 'bubble', 'element': 'fire',
         'comment': "Irrational exuberance - cautious participation"},
    56: {'name': 'Wanderer', 'action': 0.0, 'confidence': 0.45, 'regime': 'uncertain', 'element': 'mountain',
         'comment': "No clear edge - stay flat"},
    57: {'name': 'Penetrating', 'action': 0.65, 'confidence': 0.77, 'regime': 'momentum', 'element': 'wind',
         'comment': "Breakdown/breakout - directional bias"},
    58: {'name': 'Joyous', 'action': 0.35, 'confidence': 0.63, 'regime': 'optimism', 'element': 'lake',
         'comment': "Positive sentiment - growth bias"},
    59: {'name': 'Dispersion', 'action': -0.55, 'confidence': 0.68, 'regime': 'breakdown', 'element': 'wind',
         'comment': "Loss of momentum - reduce risk"},
    60: {'name': 'Limitation', 'action': -0.2, 'confidence': 0.55, 'regime': 'constraint', 'element': 'water',
         'comment': "Capital constraints - position sizing"},
    61: {'name': 'Inner Truth', 'action': 0.45, 'confidence': 0.67, 'regime': 'clarity', 'element': 'wind',
         'comment': "Strong convictions - concentrated bets"},
    62: {'name': 'Small Exceed', 'action': 0.15, 'confidence': 0.53, 'regime': 'fragile', 'element': 'thunder',
         'comment': "Fragile gains - tight stops"},
    63: {'name': 'After Completion', 'action': -0.4, 'confidence': 0.65, 'regime': 'mature', 'element': 'fire',
         'comment': "Cycle peak - profit taking"},
    64: {'name': 'Before Completion', 'action': 0.5, 'confidence': 0.72, 'regime': 'turnaround', 'element': 'fire',
         'comment': "Market bottoming - early accumulation"}
}

# Import the needed packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# -------------------------
# List all helper functions
# -------------------------
def rolling_max_drawdown(price_series, window):
    roll_max = price_series.rolling(window, min_periods=1).max()
    drawdown = price_series / roll_max - 1
    max_dd = drawdown.rolling(window, min_periods=1).min()
    return max_dd.abs()
#
def calculate_attributes(price, window=20, risk_aversion=1, delay=1):
    returns = price.pct_change().dropna()
    ret_rolling = returns.rolling(window, min_periods=window)
    momentum = ret_rolling.apply(lambda x: np.prod(x+1)-1, raw=True).shift(periods=delay)
    volatility = ret_rolling.std().shift(periods=delay)
    skewness = ret_rolling.apply(skew).shift(periods=delay)
    kurt = ret_rolling.apply(kurtosis).shift(periods=delay)
    utility = (ret_rolling.mean() - risk_aversion*(volatility ** 2)).shift(periods=delay)
    max_dd = rolling_max_drawdown(price, window).shift(periods=delay)
    attrs = pd.DataFrame({
        'momentum': momentum,
        'volatility': volatility,
        'skewness': skewness,
        'kurtosis': kurt,
        'utility': utility,
        'max_drawdown': max_dd
    })
    return attrs.dropna()
#
def attribute_to_binary(attr_df):
    binary_df = attr_df.copy()
    for col in attr_df.columns:
        median_val = attr_df[col].median()
        if col == 'max_drawdown':
            binary_df[col] = (attr_df[col] < median_val).astype(int)
        else:
            binary_df[col] = (attr_df[col] > median_val).astype(int)
    return binary_df
#
def binary_to_hexagram(binary_series):
    bits = binary_series[['momentum', 'volatility', 'skewness', 'kurtosis', 'utility', 'max_drawdown']].values
    hexagram = 0
    for i, bit in enumerate(bits):
        hexagram += bit << i
    # The I Ching hexagrams are 1-indexed (1-64), so add 1
    return hexagram + 1

# --- Augmented Signal Generation using HEXAGRAM_MEANINGS ---
def generate_signals(price, window=20, risk_aversion=1, delay=1):
    attrs = calculate_attributes(price, window, risk_aversion, delay)
    binary_attrs = attribute_to_binary(attrs)
    hexagrams = binary_attrs.apply(binary_to_hexagram, axis=1)
    signals_df = pd.DataFrame(index=hexagrams.index)
    signals_df['hexagram'] = hexagrams
    # Attach all hexagram meanings data
    signals_df['action'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('action', 0.0))
    signals_df['confidence'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('confidence', 0.5))
    signals_df['regime'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('regime', 'unknown'))
    signals_df['element'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('element', 'unknown'))
    signals_df['comment'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('comment', ''))
    signals_df['hexagram_name'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('name', f'Hexagram {h}'))
    # Add realized next day return (target for ML) - no need to shift, attributes are already shifted
    signals_df['return_next_day'] = price.pct_change()
    return signals_df

# --- Backtest: Original Hexagram vs ML Signal ---
def backtest(price, signals_df, add_2delay=0, leverage=True, f_factor=252, signal_switch=1, label='Strategy', stype='H'):
    # Set strategy type
    if stype == 'H':
        s_label = 'action'
    else:
        s_label = 'ml_action'
    # Note that we are not shifting the returns, the attributes were already
    # calculated with a delay; but we allow a second delay for the signals
    returns = price.pct_change()
    if leverage:
        signals = np.sign(signals_df[s_label])
    else:
        signals = signals_df[s_label]
    # Below is the second delay
    strat_returns = signal_switch * signals.shift(periods=add_2delay) * returns
    all = pd.concat([strat_returns, returns], axis=1).dropna()
    strat_returns = all.iloc[:,0]
    returns = all.iloc[:,1]
    # Compute the performance measures
    cum_rets = (1 + strat_returns).cumprod() - 1
    sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(f_factor) if strat_returns.std() != 0 else np.nan
    cum_returns = (1 + strat_returns).cumprod()
    roll_max = cum_returns.cummax()
    drawdown = (cum_returns - roll_max) / roll_max
    max_dd = drawdown.min()
    # Repeat the measures for the benchmark
    cumr_bench = (1 + returns).cumprod()-1
    sharpe_bench = returns.mean() / returns.std() * np.sqrt(f_factor) if returns.std() != 0 else np.nan
    cumb_returns = (1 + returns).cumprod()
    roll_max_bench = cumb_returns.cummax()
    drawdown_bench = (cumb_returns - roll_max) / roll_max
    max_dd_bench = drawdown_bench.min()
    performance = {
        'Strategy is:': label,
        'Total Return Benchmark': cumr_bench.iloc[-1],
        'Annualized Sharpe Benchmark': sharpe_bench,
        'Max Drawdown Benchmark': max_dd_bench,
        'Total Return': cum_rets.iloc[-1],
        'Annualized Sharpe': sharpe,
        'Max Drawdown': max_dd
    }
    return strat_returns, cum_rets, cumr_bench, performance

# --- Run the system ---

# Download some data
ticker = 'SPY'
start_date = '2022-01-01'
end_date = '2025-05-01'
freq = '1d'
if freq == '1d':
    set_f_factor = 252
elif freq == '1wk':
    set_f_factor = 52
elif freq == '1mo':
    set_f_factor = 12
data = yf.download(ticker, start=start_date, end=end_date, interval=freq)
# Extract closing prices
price = data['Close'].dropna()
# Convert to series
price = pd.Series(price.values.flatten(), index=price.index, name=ticker)

# Set the parametrizations
set_window = 21
set_risk_aversion = 1
# You can set the delay and 2nd delay as: (1, 0) or (1, 1) or (0, 1) or similar combination
# but never as (0, 0)!!!
set_delay = 0
set_delay_ml = 1 # this has to equal or greater to 1 one for the ML training
set_add_2delay = 1
set_leverage = True
set_signal_switch = +1
# Set probability thresholds for ML classification
prob_up = 0.55
prob_dn = 0.45
# Set training split
train_pct = 0.5

# Generate the initial signals
signals_df = generate_signals(price, set_window, set_risk_aversion, set_delay)
# One more time for the ML training
signals_df_ml = generate_signals(price, set_window, set_risk_aversion, set_delay_ml)

# --- Machine Learning Pipeline ---
# One-hot encode hexagram, regime, element
categorical_cols = ['hexagram', 'regime', 'element']
X_cats = pd.get_dummies(signals_df_ml[categorical_cols], drop_first=True)
X_num = signals_df_ml[['action', 'confidence']].copy()
X = pd.concat([X_cats, X_num], axis=1)

# Binary classification: will next day return be positive?
y = (signals_df_ml['return_next_day'] > 0).astype(int)

# Train/test split (time-ordered for realistic backtest) - there is cheating going on
# here because the training of the ML part is not rolling per-se, the code is illustrative
split_idx = int(len(X) * train_pct)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
dates_train, dates_test = signals_df_ml.index[:split_idx], signals_df_ml.index[split_idx:]

# Train Random Forest, illustrative values
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ML-based signal: probability of up move, note that we should test on the unobserved data
# and we stick with the benchmark on the previous periods
ml_prob = clf.predict_proba(X_test)[:, 1]
signals_df_ml['ml_signal'] = prob_up + 0.01 # stay long in the training period
signals_df_ml['ml_signal'].iloc[split_idx:] = ml_prob
signals_df_ml['ml_action'] = np.where(signals_df_ml['ml_signal'] > prob_up, 1, np.where(signals_df_ml['ml_signal'] < prob_dn, -1, 0))

# Original dictionary-based action
strat_returns, cum_rets, cumr_bench, performance = backtest(price, signals_df, set_add_2delay, set_leverage, set_f_factor, set_signal_switch, label='Hexagram Dictionary', stype='H')

# ML-based action
ml_strat_returns, ml_cum_rets, ml_cumr_bench, ml_performance = backtest(price, signals_df_ml, set_add_2delay, set_leverage, set_f_factor, set_signal_switch, label='ML Hexagrams', stype='ML')

# --- Plot ---
plt.figure(figsize=(12,6))
plt.plot(cum_rets, label='Hexagram Dictionary Cumulative Return')
plt.plot(ml_cum_rets, label='ML Signal Cumulative Return')
plt.plot(cumr_bench, label='Buy & Hold')
plt.legend()
plt.title('I Ching Hexagram vs ML Signal vs Buy & Hold')
plt.show()

# --- Print performance ---
print(performance)
print(ml_performance)

# --- ML Performance Metrics ---
y_pred = clf.predict(X_test)
print("\nML Out-of-sample accuracy: {:.3f}".format(accuracy_score(y_test, y_pred)))
print("ML Out-of-sample ROC-AUC: {:.3f}".format(roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])))

# --- Show sample signals with meanings and ML probabilities ---
print("\nSample signals with hexagram meanings and ML probabilities:")
print(signals_df_ml[['hexagram', 'action', 'ml_signal', 'ml_action', 'return_next_day']].tail(10))
