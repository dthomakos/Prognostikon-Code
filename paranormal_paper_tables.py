"""
paranormal_paper_tables.py
--------------------------
Downloads all five empirical series, fits the Paranormal distribution
plus six benchmark models, and writes LaTeX-formatted tables ready to
\\input{} into paranormal_paper_final.tex.

Run order (Section 6 of the paper):
    python paranormal_paper_tables.py

Outputs:
    artifacts/tables/table_params_macro.tex
    artifacts/tables/table_params_finance.tex
    artifacts/tables/table_aic.tex
    artifacts/data/  (cached CSVs for reproducibility)

Bugs fixed vs previous version
-------------------------------
1. yfinance >= 0.2 writes MultiIndex column headers (e.g. ('Close','SPY')).
   The old loader silently returned NaN-only series from cached CSVs.
   Fixed: _extract_close() flattens MultiIndex and coerces to numeric.
2. np.isfinite() guard added to every return series after computation.
3. Global finite filter on the datasets dict before any fitting.
4. build_aic_table() crashed with ValueError: min() arg is empty sequence
   when ALL models failed for a series (e.g. SPY/TNA had all NaN data).
   Fixed: guard with `if finite_vals` before calling min().
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import warnings
import pathlib

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

ROOT      = pathlib.Path('artifacts')
TABLE_DIR = ROOT / 'tables'
DATA_DIR  = ROOT / 'data'
for d in [TABLE_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

from paranormal_improved import ParanormalFitter

# =============================================================================
# 1.  DATA DOWNLOAD
# =============================================================================

def download_fred(series_id, start='1970-01-01', end='2025-12-31'):
    cache = DATA_DIR / f'{series_id}.csv'
    if cache.exists():
        print(f'  [cache]    {series_id}')
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        return df[series_id]
    print(f'  [FRED]     {series_id}')
    try:
        from pandas_datareader import data as pdr
        df = pdr.DataReader(series_id, 'fred', start=start, end=end)
        df.to_csv(cache)
        return df[series_id]
    except Exception as e:
        print(f'    Warning: {e}')
        return None


def _extract_close(df, ticker=''):
    """
    Robustly extract the close-price column from a yfinance DataFrame.

    yfinance >= 0.2 uses MultiIndex columns such as ('Close', 'SPY').
    When saved to CSV and reloaded, these become strings like "Close SPY".
    This function handles all known variants.
    """
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [' '.join(str(c) for c in col).strip()
                      for col in df.columns]

    # Priority candidates
    candidates = [
        'Adj Close', 'Close',
        f'Adj Close {ticker}', f'Close {ticker}',
        'adj close', 'close',
    ]
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors='coerce')

    # Fall back to first numeric column
    num_cols = df.select_dtypes(include=[float, int]).columns.tolist()
    if num_cols:
        return pd.to_numeric(df[num_cols[0]], errors='coerce')

    # Absolute last resort
    return pd.to_numeric(df.iloc[:, 0], errors='coerce')


def download_yfinance(ticker, start='2010-01-01', end='2025-12-31'):
    safe  = ticker.replace('-', '_')
    cache = DATA_DIR / f'{safe}.csv'

    if cache.exists():
        print(f'  [cache]    {ticker}')
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        s  = _extract_close(df, ticker)
        s  = s.dropna()
        # If cached file is corrupt (all NaN), re-download
        if len(s) < 100:
            print(f'    Cached file appears corrupt ({len(s)} valid rows).'
                  f' Re-downloading...')
            cache.unlink(missing_ok=True)
            return download_yfinance(ticker, start=start, end=end)
        return s

    print(f'  [yfinance] {ticker}')
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)
        df.to_csv(cache)
        s  = _extract_close(df, ticker)
        return s.dropna()
    except Exception as e:
        print(f'    Warning: {e}')
        return None


print('=' * 70)
print('STEP 1: Download / load data')
print('=' * 70)

# CPI inflation: year-over-year log-change x 100
inflation = None
cpi_raw = download_fred('CPIAUCSL') #, start='1970-01-01', end='2025-12-31')
if cpi_raw is not None:
    cpi_raw   = cpi_raw.dropna()
    inflation = (np.log(cpi_raw) - np.log(cpi_raw.shift(12))).dropna() * 100
    #inflation = inflation['2000-01-01':]
    inflation = inflation[np.isfinite(inflation)]
    print(f'  CPI inflation : {len(inflation)} monthly obs')
    inflation.to_csv(DATA_DIR / 'inflation.csv')

# GDP growth: annualised QoQ log-change x 400
gdp_growth = None
gdp_raw = download_fred('GDPC1') #, start='1970-10-01', end='2025-12-31')
if gdp_raw is not None:
    gdp_raw    = gdp_raw.dropna()
    gdp_growth = (np.log(gdp_raw) - np.log(gdp_raw.shift(1))).dropna() * 400
    #gdp_growth = gdp_growth['2000-01-01':]
    gdp_growth = gdp_growth[np.isfinite(gdp_growth)]
    print(f'  GDP growth    : {len(gdp_growth)} quarterly obs')
    gdp_growth.to_csv(DATA_DIR / 'gdp_growth.csv')

# SPY daily log-returns
spy_ret = None
spy_price = download_yfinance('SPY') #, start='2010-01-01')
if spy_price is not None:
    spy_price = pd.to_numeric(spy_price, errors='coerce').dropna()
    spy_ret   = (np.log(spy_price) - np.log(spy_price.shift(1))).dropna() * 100
    spy_ret   = spy_ret[(spy_ret != 0.0) & np.isfinite(spy_ret)]
    print(f'  SPY returns   : {len(spy_ret)} daily obs')
    spy_ret.to_csv(DATA_DIR / 'spy_returns.csv')

# TNA daily log-returns
tna_ret = None
tna_price = download_yfinance('TNA')#, start='2010-11-01')
if tna_price is not None:
    tna_price = pd.to_numeric(tna_price, errors='coerce').dropna()
    tna_ret   = (np.log(tna_price) - np.log(tna_price.shift(1))).dropna() * 100
    tna_ret   = tna_ret[(tna_ret != 0.0) & np.isfinite(tna_ret)]
    print(f'  TNA returns   : {len(tna_ret)} daily obs')
    tna_ret.to_csv(DATA_DIR / 'tna_returns.csv')

# BTC-USD daily log-returns
btc_ret = None
btc_price = download_yfinance('BTC-USD') #, start='2010-07-01')
if btc_price is not None:
    btc_price = pd.to_numeric(btc_price, errors='coerce').dropna()
    btc_ret   = (np.log(btc_price) - np.log(btc_price.shift(1))).dropna() * 100
    btc_ret   = btc_ret[(btc_ret.abs() < 60) & np.isfinite(btc_ret)]
    print(f'  BTC returns   : {len(btc_ret)} daily obs')
    btc_ret.to_csv(DATA_DIR / 'btc_returns.csv')


# Build datasets dict with a final NaN/Inf safety net
datasets = {}
for label, arr in [
    ('CPI Inflation', inflation),
    ('GDP Growth',    gdp_growth),
    ('SPY',           spy_ret),
    ('TNA',           tna_ret),
    ('BTC-USD',       btc_ret),
]:
    if arr is not None:
        a = np.asarray(arr, dtype=float).flatten()
        a = a[np.isfinite(a)]
        if len(a) >= 30:
            datasets[label] = a
        else:
            print(f'  WARNING: {label} has only {len(a)} finite obs — skipped')


# =============================================================================
# 2.  BENCHMARK FITTERS
# =============================================================================

def _ic(ll, k, n):
    return {'ll': ll, 'k': k,
            'aic': 2*k - 2*ll,
            'bic': k*np.log(n) - 2*ll}


def fit_normal(x):
    ll = np.sum(stats.norm.logpdf(x, x.mean(), x.std(ddof=1)))
    return _ic(ll, 2, len(x))


def fit_student_t(x):
    p = stats.t.fit(x)
    return _ic(np.sum(stats.t.logpdf(x, *p)), 3, len(x))


def fit_skewnorm(x):
    p = stats.skewnorm.fit(x)
    return _ic(np.sum(stats.skewnorm.logpdf(x, *p)), 3, len(x))


def fit_skewt(x):
    try:
        p  = stats.jf_skew_t.fit(x)
        ll = np.sum(stats.jf_skew_t.logpdf(x, *p))
        k  = 4
    except AttributeError:
        p  = stats.t.fit(x)
        ll = np.sum(stats.t.logpdf(x, *p))
        k  = 3
    return _ic(ll, k, len(x))


def fit_gauss_mix(x):
    from sklearn.mixture import GaussianMixture
    x_col = x.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=SEED, max_iter=500)
    gm.fit(x_col)
    ll = gm.score(x_col) * len(x)
    return _ic(ll, 5, len(x))


def fit_nig(x):
    try:
        p  = stats.norminvgauss.fit(x)
        ll = np.sum(stats.norminvgauss.logpdf(x, *p))
        k  = 4
    except Exception:
        p  = stats.t.fit(x)
        ll = np.sum(stats.t.logpdf(x, *p))
        k  = 3
    return _ic(ll, k, len(x))


BENCHMARKS = [
    ('Normal',      fit_normal),
    ('Student-t',   fit_student_t),
    ('Skew-Normal', fit_skewnorm),
    ('Skew-t',      fit_skewt),
    ('Gauss.Mix.',  fit_gauss_mix),
    ('NIG',         fit_nig),
]
MODEL_NAMES = [b[0] for b in BENCHMARKS] + ['Paranormal']


# =============================================================================
# 3.  FIT ALL MODELS
# =============================================================================

print()
print('=' * 70)
print('STEP 2: Fit all models')
print('=' * 70)

results = {}

for dname, data in datasets.items():
    # Final safety check
    assert np.all(np.isfinite(data)), f'{dname}: still has non-finite values!'
    print(f'\n--- {dname}  (n={len(data)}) ---')
    res = {}

    for mname, fitter in BENCHMARKS:
        try:
            r = fitter(data)
            res[mname] = r
            print(f'  {mname:<14} LL={r["ll"]:>10.2f}'
                  f'  AIC={r["aic"]:>10.2f}  BIC={r["bic"]:>10.2f}')
        except Exception as e:
            res[mname] = {'ll': np.nan, 'aic': np.nan, 'bic': np.nan, 'k': 0}
            print(f'  {mname:<14} ERROR: {e}')

    try:
        fp = ParanormalFitter.fit(data, maxiter=100, verbose=False)
        res['Paranormal'] = {
            'll':     fp['log_likelihood'],
            'aic':    fp['AIC'],
            'bic':    fp['BIC'],
            'k':      9,
            'params': fp['parameters'],
        }
        print(f'  {"Paranormal":<14} LL={fp["log_likelihood"]:>10.2f}'
              f'  AIC={fp["AIC"]:>10.2f}  BIC={fp["BIC"]:>10.2f}')
    except Exception as e:
        res['Paranormal'] = {'ll': np.nan, 'aic': np.nan,
                             'bic': np.nan, 'k': 9}
        print(f'  {"Paranormal":<14} ERROR: {e}')

    results[dname] = res


# =============================================================================
# 4.  LATEX TABLE HELPERS
# =============================================================================

def fv(x, d=3):
    """Format a float; return dashes if NaN/None."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '---'
    return f'{x:.{d}f}'


def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(content)
    print(f'  Written: {path}')


# =============================================================================
# 4a.  PARAMETER TABLES
# =============================================================================

PARAM_ROWS = [
    ('mu1',    r'$\hat{\mu}_1$'),
    ('sigma1', r'$\hat{\sigma}_1$'),
    ('alpha1', r'$\hat{\alpha}_1$'),
    ('nu1',    r'$\hat{\nu}_1$'),
    ('mu2',    r'$\hat{\mu}_2$'),
    ('sigma2', r'$\hat{\sigma}_2$'),
    ('alpha2', r'$\hat{\alpha}_2$'),
    ('nu2',    r'$\hat{\nu}_2$'),
    ('w1',     r'$\hat{w}_1$'),
]


def build_params_table(series_names, caption, label):
    nc   = len(series_names)
    cols = 'l' + 'c' * nc
    hdr  = ' & '.join(series_names)

    rows = []
    for key, tex_name in PARAM_ROWS:
        cells = []
        for sname in series_names:
            p = (results.get(sname, {})
                        .get('Paranormal', {})
                        .get('params', {}))
            cells.append(fv(p.get(key, np.nan)))
        rows.append(tex_name + ' & ' + ' & '.join(cells) + r' \\')

    stat_rows = []
    for slabel, skey, dec in [
            ('Log-lik.',  'll',  1),
            ('AIC',       'aic', 1),
            ('BIC',       'bic', 1),
    ]:
        cells = []
        for sname in series_names:
            v = (results.get(sname, {})
                        .get('Paranormal', {})
                        .get(skey, np.nan))
            cells.append(fv(v, dec))
        stat_rows.append(slabel + ' & ' + ' & '.join(cells) + r' \\')

    note = (r'\multicolumn{' + str(nc + 1) + r'}{l}{\footnotesize'
            r' Run \texttt{paranormal\_paper\_tables.py} to populate.}')

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{' + cols + '}',
        r'\toprule',
        r'Parameter & ' + hdr + r' \\',
        r'\midrule',
    ]
    lines += rows
    lines += [r'\midrule']
    lines += stat_rows
    lines += [
        r'\bottomrule',
        note + r' \\',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# =============================================================================
# 4b.  AIC COMPARISON TABLE
# =============================================================================

def build_aic_table():
    series_names = list(datasets.keys())
    nc   = len(series_names)
    cols = 'l' + 'c' * nc
    hdr  = ' & '.join(series_names)

    # Best AIC per column — guard against all-NaN
    best_aic = {}
    for sname in series_names:
        vals = [results.get(sname, {}).get(m, {}).get('aic', np.inf)
                for m in MODEL_NAMES]
        finite_vals = [v for v in vals if np.isfinite(v)]
        best_aic[sname] = min(finite_vals) if finite_vals else np.inf

    rows = []
    for mname in MODEL_NAMES:
        cells = []
        for sname in series_names:
            v    = results.get(sname, {}).get(mname, {}).get('aic', np.nan)
            cell = fv(v, 1)
            if np.isfinite(v) and abs(v - best_aic[sname]) < 0.15:
                cell = r'\textbf{' + cell + '}'
            cells.append(cell)

        name_cell = (r'\textbf{Paranormal}' if mname == 'Paranormal'
                     else mname)
        rows.append(name_cell + ' & ' + ' & '.join(cells) + r' \\')

    note = (r'\multicolumn{' + str(nc + 1) + r'}{l}{\footnotesize'
            r' Bold = best model per series (lowest AIC). '
            r'Run \texttt{paranormal\_paper\_tables.py} to populate.}')

    lines = [
        r'\begin{table}[ht]',
        r'\centering',
        r'\caption{Model comparison by AIC (lower $=$ better). '
        r'Parameter counts: Normal~2, Student-$t$~3, Skew-Normal~3, '
        r'Skew-$t$~4, Gaussian Mix.~5, NIG~4, Paranormal~9.}',
        r'\label{tab:aic}',
        r'\begin{tabular}{' + cols + '}',
        r'\toprule',
        r'Model & ' + hdr + r' \\',
        r'\midrule',
    ]
    lines += rows
    lines += [
        r'\bottomrule',
        note + r' \\',
        r'\end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


# =============================================================================
# 5.  WRITE TABLES
# =============================================================================

print()
print('=' * 70)
print('STEP 3: Write LaTeX tables')
print('=' * 70)

macro_series   = [s for s in ['CPI Inflation', 'GDP Growth']
                  if s in datasets]
finance_series = [s for s in ['SPY', 'TNA', 'BTC-USD']
                  if s in datasets]

if macro_series:
    tex = build_params_table(
        macro_series,
        'Paranormal MLE: macroeconomic series.',
        'tab:params_macro',
    )
    write_file(TABLE_DIR / 'table_params_macro.tex', tex)

if finance_series:
    tex = build_params_table(
        finance_series,
        'Paranormal MLE: financial series.',
        'tab:params_finance',
    )
    write_file(TABLE_DIR / 'table_params_finance.tex', tex)

tex = build_aic_table()
write_file(TABLE_DIR / 'table_aic.tex', tex)

# CSV summary
import pandas as _pd
rows_csv = []
for sname in datasets:
    for mname in MODEL_NAMES:
        r = results.get(sname, {}).get(mname, {})
        rows_csv.append({
            'series': sname, 'model': mname,
            'll':  r.get('ll',  np.nan),
            'aic': r.get('aic', np.nan),
            'bic': r.get('bic', np.nan),
        })
_pd.DataFrame(rows_csv).to_csv(DATA_DIR / 'model_comparison.csv', index=False)
print(f'  Written: {DATA_DIR}/model_comparison.csv')

print()
print('=' * 70)
print('DONE')
print('=' * 70)
print()
print('Use in paranormal_paper_final.tex:')
print('  \\\\input{artifacts/tables/table_params_macro.tex}')
print('  \\\\input{artifacts/tables/table_params_finance.tex}')
print('  \\\\input{artifacts/tables/table_aic.tex}')
