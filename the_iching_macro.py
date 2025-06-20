#
# Python code replicating results on this post:
#
# https://prognostikon.cce.uoa.gr/fkyriazi/the-i-ching-macro-monitor/
#
# at my blog Prognostikon
#
# (c) Dimitrios D. Thomakos, dimitrios.thomakos@gmail.com, https://github.com/dthomakos
# -------------------------------------------------------------------------------------


# Import packages
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader.data as web
import datetime

# --- Economic Hexagram Meanings Dictionary ---
HEXAGRAM_MEANINGS = {
    1: {'name': 'Creative Heaven', 'action': 1.0, 'confidence': 0.95, 'regime': 'growth', 'element': 'metal',
        'comment': 'National renewal; strong growth, robust reserves, confidence in economic leadership.'},
    2: {'name': 'Receptive Earth', 'action': -1.0, 'confidence': 0.9, 'regime': 'stability', 'element': 'earth',
        'comment': 'Foundations, capacity for support; economic stability, readiness for new cycles.'},
    3: {'name': 'Difficulty at the Beginning', 'action': 0.2, 'confidence': 0.6, 'regime': 'recovery', 'element': 'water',
        'comment': 'Emerging sectors, start-up economies, or post-crisis recovery; uncertainty and need for nurturing.'},
    4: {'name': 'Youthful Folly', 'action': 0.0, 'confidence': 0.4, 'regime': 'learning', 'element': 'mountain',
        'comment': 'Inexperience in policy, learning phase, risk of naive decisions; need for guidance and education.'},
    5: {'name': 'Waiting', 'action': 0.3, 'confidence': 0.65, 'regime': 'patience', 'element': 'water',
        'comment': 'Patience in policy, reserves held in readiness, waiting for favorable conditions.'},
    6: {'name': 'Conflict', 'action': -0.7, 'confidence': 0.75, 'regime': 'dispute', 'element': 'heaven',
        'comment': 'Trade disputes, legal/economic conflicts, need for negotiation and arbitration.'},
    7: {'name': 'The Army', 'action': 0.5, 'confidence': 0.8, 'regime': 'mobilization', 'element': 'earth',
        'comment': 'Mobilization of resources, coordinated policy action, state-led investment.'},
    8: {'name': 'Holding Together', 'action': 0.4, 'confidence': 0.7, 'regime': 'alliance', 'element': 'water',
        'comment': 'Economic alliances, regional integration, or stimulus through collective effort.'},
    9: {'name': 'Small Taming', 'action': 0.25, 'confidence': 0.6, 'regime': 'reform', 'element': 'wind',
        'comment': 'Incremental reforms, fine-tuning policy, gradual improvements.'},
    10: {'name': 'Treading', 'action': -0.3, 'confidence': 0.65, 'regime': 'caution', 'element': 'lake',
        'comment': 'Cautious progress, risk management, walking a fine line in policy.'},
    11: {'name': 'Peace', 'action': 0.8, 'confidence': 0.85, 'regime': 'harmony', 'element': 'earth',
        'comment': 'Economic harmony, strong reserves, prosperity, confidence in international markets.'},
    12: {'name': 'Standstill', 'action': -0.8, 'confidence': 0.8, 'regime': 'stagnation', 'element': 'heaven',
        'comment': 'Stagnation, policy gridlock, or external shocks causing economic pause.'},
    13: {'name': 'Fellowship', 'action': 0.6, 'confidence': 0.75, 'regime': 'cooperation', 'element': 'fire',
        'comment': 'International cooperation, trade agreements, global partnerships.'},
    14: {'name': 'Great Possession', 'action': 0.9, 'confidence': 0.9, 'regime': 'abundance', 'element': 'fire',
        'comment': 'Abundance of resources, peak reserves, economic clout.'},
    15: {'name': 'Modesty', 'action': 0.1, 'confidence': 0.55, 'regime': 'prudence', 'element': 'mountain',
        'comment': 'Prudent fiscal policy, conservative spending, sustainable growth.'},
    16: {'name': 'Enthusiasm', 'action': 0.7, 'confidence': 0.8, 'regime': 'optimism', 'element': 'thunder',
        'comment': 'Economic optimism, consumer confidence, expansionary sentiment.'},
    17: {'name': 'Following', 'action': 0.65, 'confidence': 0.78, 'regime': 'adaptation', 'element': 'lake',
        'comment': 'Adapting to global trends, following market leaders, policy alignment.'},
    18: {'name': 'Work on the Decayed', 'action': -0.6, 'confidence': 0.7, 'regime': 'reform', 'element': 'wind',
        'comment': 'Addressing structural weaknesses, reforming outdated sectors, anti-corruption.'},
    19: {'name': 'Approach', 'action': 0.4, 'confidence': 0.65, 'regime': 'investment', 'element': 'earth',
        'comment': 'Policy outreach, investment in infrastructure, preparation for growth.'},
    20: {'name': 'Contemplation', 'action': 0.0, 'confidence': 0.5, 'regime': 'assessment', 'element': 'wind',
        'comment': 'Economic assessment, strategic review, policy evaluation.'},
    21: {'name': 'Biting Through', 'action': 0.55, 'confidence': 0.72, 'regime': 'enforcement', 'element': 'fire',
        'comment': 'Enforcement of regulations, decisive intervention, anti-monopoly action.'},
    22: {'name': 'Grace', 'action': 0.3, 'confidence': 0.6, 'regime': 'soft power', 'element': 'mountain',
        'comment': 'Economic “soft power,” branding, investment in culture and image.'},
    23: {'name': 'Splitting Apart', 'action': -0.85, 'confidence': 0.88, 'regime': 'crisis', 'element': 'mountain',
        'comment': 'Erosion of reserves, fiscal crisis, risk of instability.'},
    24: {'name': 'Return', 'action': 0.75, 'confidence': 0.82, 'regime': 'recovery', 'element': 'earth',
        'comment': 'Recovery from recession, return of growth, cyclical upturn.'},
    25: {'name': 'Innocence', 'action': 0.15, 'confidence': 0.58, 'regime': 'innovation', 'element': 'heaven',
        'comment': 'Unencumbered innovation, simple markets, or risk of unregulated speculation.'},
    26: {'name': 'Great Taming', 'action': 0.5, 'confidence': 0.7, 'regime': 'buffering', 'element': 'mountain',
        'comment': 'Accumulation of reserves, restraint, building buffers for future use.'},
    27: {'name': 'Nourishment', 'action': 0.7, 'confidence': 0.85, 'regime': 'human capital', 'element': 'thunder',
        'comment': 'Investment in human capital, education, social safety nets.'},
    28: {'name': 'Great Excess', 'action': -0.9, 'confidence': 0.9, 'regime': 'bubble', 'element': 'lake',
        'comment': 'Overextension, bubbles, unsustainable debt or asset prices.'},
    29: {'name': 'The Abysmal', 'action': -0.6, 'confidence': 0.75, 'regime': 'crisis', 'element': 'water',
        'comment': 'Crisis, recession, liquidity trap, or external shocks.'},
    30: {'name': 'The Clinging', 'action': 0.8, 'confidence': 0.85, 'regime': 'transparency', 'element': 'fire',
        'comment': 'Transparency, information economy, clarity in policy.'},
    31: {'name': 'Influence', 'action': 0.45, 'confidence': 0.68, 'regime': 'soft diplomacy', 'element': 'lake',
        'comment': 'Soft diplomacy, cultural exports, influence through economic means.'},
    32: {'name': 'Duration', 'action': 0.6, 'confidence': 0.75, 'regime': 'sustainability', 'element': 'thunder',
        'comment': 'Long-term planning, sustainability, enduring policy frameworks.'},
    33: {'name': 'Retreat', 'action': -0.7, 'confidence': 0.8, 'regime': 'austerity', 'element': 'mountain',
        'comment': 'Policy withdrawal, austerity, reduction in commitments.'},
    34: {'name': 'Great Power', 'action': 0.85, 'confidence': 0.88, 'regime': 'strength', 'element': 'thunder',
        'comment': 'Economic might, potential for transformation, need for wise use of power.'},
    35: {'name': 'Progress', 'action': 0.65, 'confidence': 0.78, 'regime': 'development', 'element': 'fire',
        'comment': 'Rapid development, technological advancement, upward mobility.'},
    36: {'name': 'Darkening of the Light', 'action': -0.75, 'confidence': 0.82, 'regime': 'opacity', 'element': 'earth',
        'comment': 'Loss of transparency, corruption, or policy obfuscation.'},
    37: {'name': 'The Family', 'action': 0.2, 'confidence': 0.55, 'regime': 'domestic focus', 'element': 'wind',
        'comment': 'Domestic policy focus, household sector, social cohesion.'},
    38: {'name': 'Opposition', 'action': -0.5, 'confidence': 0.65, 'regime': 'divergence', 'element': 'fire',
        'comment': 'Policy divergence, regional disparities, internal conflict.'},
    39: {'name': 'Obstruction', 'action': -0.4, 'confidence': 0.6, 'regime': 'barriers', 'element': 'water',
        'comment': 'Barriers to growth, trade restrictions, supply chain issues.'},
    40: {'name': 'Deliverance', 'action': 0.75, 'confidence': 0.83, 'regime': 'relief', 'element': 'thunder',
        'comment': 'Resolution of crisis, debt relief, bailout or stimulus.'},
    41: {'name': 'Decrease', 'action': -0.6, 'confidence': 0.72, 'regime': 'contraction', 'element': 'mountain',
        'comment': 'Austerity, resource depletion, reduction in reserves.'},
    42: {'name': 'Increase', 'action': 0.8, 'confidence': 0.85, 'regime': 'expansion', 'element': 'wind',
        'comment': 'Growth, resource windfall, expansion of reserves.'},
    43: {'name': 'Breakthrough', 'action': 0.9, 'confidence': 0.9, 'regime': 'reform', 'element': 'lake',
        'comment': 'Major reform, policy shift, breakthrough innovation.'},
    44: {'name': 'Coming to Meet', 'action': 0.25, 'confidence': 0.58, 'regime': 'investment', 'element': 'wind',
        'comment': 'Foreign investment, new partnerships, external opportunities.'},
    45: {'name': 'Gathering Together', 'action': 0.5, 'confidence': 0.7, 'regime': 'capital accumulation', 'element': 'earth',
        'comment': 'Capital accumulation, coalition-building, pooling resources.'},
    46: {'name': 'Pushing Upward', 'action': 0.7, 'confidence': 0.8, 'regime': 'mobility', 'element': 'wood',
        'comment': 'Social mobility, upward economic movement, inclusive growth.'},
    47: {'name': 'Oppression', 'action': -0.85, 'confidence': 0.87, 'regime': 'hardship', 'element': 'lake',
        'comment': 'Economic hardship, debt burden, unemployment.'},
    48: {'name': 'The Well', 'action': 0.3, 'confidence': 0.62, 'regime': 'renewal', 'element': 'water',
        'comment': 'Sustainable resources, effective reserve management, economic renewal.'},
    49: {'name': 'Revolution', 'action': 0.6, 'confidence': 0.75, 'regime': 'transformation', 'element': 'fire',
        'comment': 'Structural reform, regime change, economic transformation.'},
    50: {'name': 'The Cauldron', 'action': 0.4, 'confidence': 0.65, 'regime': 'innovation', 'element': 'fire',
        'comment': 'Transformation through investment, innovation, or policy “alchemy”.'},
    51: {'name': 'The Arousing', 'action': 0.9, 'confidence': 0.92, 'regime': 'shock', 'element': 'thunder',
        'comment': 'Economic shock, sudden change, need for rapid response.'},
    52: {'name': 'Keeping Still', 'action': -0.3, 'confidence': 0.6, 'regime': 'stabilization', 'element': 'mountain',
        'comment': 'Policy pause, stabilization, holding pattern.'},
    53: {'name': 'Gradual Progress', 'action': 0.55, 'confidence': 0.73, 'regime': 'steady growth', 'element': 'wind',
        'comment': 'Steady development, incremental improvement, compounding gains.'},
    54: {'name': 'The Marrying Maiden', 'action': 0.1, 'confidence': 0.52, 'regime': 'emerging markets', 'element': 'thunder',
        'comment': 'New entrants, emerging markets, risk of dependency.'},
    55: {'name': 'Abundance', 'action': 0.85, 'confidence': 0.89, 'regime': 'boom', 'element': 'fire',
        'comment': 'Economic boom, peak prosperity, overflowing reserves.'},
    56: {'name': 'The Wanderer', 'action': 0.0, 'confidence': 0.45, 'regime': 'instability', 'element': 'mountain',
        'comment': 'Capital flight, economic migration, instability.'},
    57: {'name': 'The Gentle', 'action': 0.65, 'confidence': 0.77, 'regime': 'soft reform', 'element': 'wind',
        'comment': 'Gradual influence, soft reforms, subtle policy shifts.'},
    58: {'name': 'The Joyous', 'action': 0.35, 'confidence': 0.63, 'regime': 'confidence', 'element': 'lake',
        'comment': 'Consumer confidence, prosperity, positive sentiment.'},
    59: {'name': 'Dispersion', 'action': -0.55, 'confidence': 0.68, 'regime': 'disintegration', 'element': 'wind',
        'comment': 'Capital outflow, dissolution of alliances, loss of cohesion.'},
    60: {'name': 'Limitation', 'action': -0.2, 'confidence': 0.55, 'regime': 'constraint', 'element': 'water',
        'comment': 'Policy constraints, regulatory limits, budget caps.'},
    61: {'name': 'Inner Truth', 'action': 0.45, 'confidence': 0.67, 'regime': 'credibility', 'element': 'wind',
        'comment': 'Trust in institutions, economic credibility, transparency.'},
    62: {'name': 'Small Exceeding', 'action': 0.15, 'confidence': 0.53, 'regime': 'caution', 'element': 'thunder',
        'comment': 'Minor gains, cautious optimism, limited progress.'},
    63: {'name': 'After Completion', 'action': -0.4, 'confidence': 0.65, 'regime': 'maturity', 'element': 'fire',
        'comment': 'Maturity, stable prosperity, risk of complacency.'},
    64: {'name': 'Before Completion', 'action': 0.5, 'confidence': 0.72, 'regime': 'transition', 'element': 'fire',
        'comment': 'Unfinished reforms, transition, potential for renewal.'}
}

# --- Download relevant monthly US macroeconomic data from FRED using your preferred method ---
series_ids = {
    'CPI': 'CPIAUCSL',
    'Energy_Price': 'MCOILWTICO',
    'Interest_Rate': 'FEDFUNDS',
    'Industrial_Production': 'INDPRO',
    'Unemployment_Rate': 'UNRATE',
    'Recessions': 'USREC'
}

start_date = '1980-01-01'
end_date = datetime.datetime.today()
data = pd.DataFrame()
for name, series_id in series_ids.items():
    print(f"Downloading {name} ({series_id})...")
    series = web.FredReader(series_id, start_date, end_date).read()
    series = series.resample('ME').ffill()  # Ensure monthly frequency
    data = pd.concat([data, series], axis=1)
data = data.dropna()
data.columns = series_ids.keys()
# Remove the recession indicator
recessions_data = data['Recessions']
data = data.drop(columns='Recessions')

# Find transitions (0→1 = recession start, 1→0 = recession end)
rec_data = recessions_data.diff()
starts = rec_data.index[rec_data == 1]   # Start dates
ends = rec_data.index[rec_data == -1]    # End dates

# Handle edge cases (if current recession is ongoing)
if rec_data.iloc[-1] == 1:
    ends = ends.append(pd.DatetimeIndex([end]))  # Extend last end date to today

# Pair start/end dates
recessions = [(start, end) for start, end in zip(starts, ends)]

# --- Rolling feature calculations ---
def calculate_attributes(df, window=12, delay=1):
    """
    Calculate rolling statistical attributes for each column in the DataFrame.
    """
    returns = df.pct_change().dropna()
    ret_rolling = returns.rolling(window, min_periods=window)
    features = {}
    for col in df.columns:
        momentum = (1 + ret_rolling[col].apply(np.prod, raw=True) - 1).shift(periods=delay)
        volatility = ret_rolling[col].std().shift(periods=delay)
        skewness = ret_rolling[col].apply(skew).shift(periods=delay)
        kurt = ret_rolling[col].apply(kurtosis).shift(periods=delay)
        features[f'{col}_momentum'] = momentum
        features[f'{col}_volatility'] = volatility
        features[f'{col}_skewness'] = skewness
        features[f'{col}_kurtosis'] = kurt
    features_df = pd.DataFrame(features)
    return features_df.dropna()

def attribute_to_binary(attr_df):
    """
    Convert selected attributes to binary values according to specified rules.
    """
    binary_df = pd.DataFrame(index=attr_df.index)
    below_median_features = [
        'CPI_skewness',
        'CPI_volatility',
        'Energy_Price_momentum',
        'Interest_Rate_volatility',
        'Unemployment_Rate_volatility'
    ]
    above_median_features = [
        'Industrial_Production_momentum'
    ]
    for col in below_median_features:
        median_val = attr_df[col].median()
        binary_df[col] = (attr_df[col] < median_val).astype(int)
    for col in above_median_features:
        median_val = attr_df[col].median()
        binary_df[col] = (attr_df[col] > median_val).astype(int)
    return binary_df

def binary_to_hexagram(binary_series):
    """
    Map the six selected binary features to a hexagram number (1 to 64).
    """
    features_order = [
        'CPI_skewness',
        'CPI_volatility',
        'Energy_Price_momentum',
        'Interest_Rate_volatility',
        'Industrial_Production_momentum',
        'Unemployment_Rate_volatility'
    ]
    hexagram = 0
    for i, feature in enumerate(features_order):
        bit = binary_series[feature]
        hexagram += bit << i
    return hexagram + 1  # 1-indexed hexagram number

# --- Pipeline execution ---
attributes = calculate_attributes(data, window=60, delay=1)
selected_features = [
    'CPI_skewness',
    'CPI_volatility',
    'Energy_Price_momentum',
    'Interest_Rate_volatility',
    'Industrial_Production_momentum',
    'Unemployment_Rate_volatility'
]
attributes_selected = attributes[selected_features]
binary_attrs = attribute_to_binary(attributes_selected)
hexagrams = binary_attrs.apply(binary_to_hexagram, axis=1)

# Attach hexagram meanings
hexagram_info = pd.DataFrame(index=hexagrams.index)
hexagram_info['hexagram'] = hexagrams
hexagram_info['name'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('name', f'Hexagram {h}'))
hexagram_info['action'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('action', 0.0))
hexagram_info['confidence'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('confidence', 0.5))
hexagram_info['regime'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('regime', 'unknown'))
hexagram_info['element'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('element', 'unknown'))
hexagram_info['comment'] = hexagrams.map(lambda h: HEXAGRAM_MEANINGS.get(h, {}).get('comment', ''))

print("Sample Hexagram Assignments and Economic Meanings:")
print(hexagram_info.tail(10))
hexagram_info.to_csv('temp.csv')

# --- Histogram & KDE: Only show hexagrams that appear in the data ---
unique_hexagrams = np.sort(hexagram_info['hexagram'].unique())
plt.figure(figsize=(14, 6))
sns.histplot(
    hexagram_info['hexagram'],
    bins=np.arange(unique_hexagrams.min() - 0.5, unique_hexagrams.max() + 1.5, 1),
    kde=False,
    color='skyblue',
    edgecolor='black',
    stat='count'
)
sns.kdeplot(
    hexagram_info['hexagram'],
    bw_adjust=1.5,
    color='darkblue',
    linewidth=2,
    fill=False,
    clip=(unique_hexagrams.min(), unique_hexagrams.max())
)
plt.xticks(unique_hexagrams)
plt.xlim(unique_hexagrams.min() - 0.5, unique_hexagrams.max() + 0.5)
plt.xlabel('Hexagram Number')
plt.ylabel('Frequency')
plt.title('Observed Hexagram Frequency Distribution with Kernel Density')
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()

# --- Time series plot: Hexagram regime, 12-month moving average, and grid ---
plt.figure(figsize=(16, 6))
hexagram_info['hexagram'].plot(label='Hexagram Regime', color='mediumblue', linewidth=1)
hexagram_info['hexagram'].rolling(window=12, min_periods=1, center=True).mean().plot(
    label='12-Month Moving Average', color='orange', linewidth=2
)
#
# Shade recession periods
for (s, e) in recessions:
    plt.axvspan(s, e, color='gray', alpha=0.3)
#
plt.title('Economic Regimes via Hexagrams Over Time')
plt.ylabel('Hexagram Number')
plt.xlabel('Date')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.show()
