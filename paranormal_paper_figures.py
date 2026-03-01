"""
paranormal_paper_figures.py
----------------------------
Generates all figures for paranormal_paper_final.tex.

Figures produced:
    fig1_density_macro.png    – CPI inflation + GDP growth fitted PDFs
    fig2_density_finance.png  – SPY / TNA / BTC-USD fitted PDFs
    fig3_qqplot.png           – Q-Q plots (all 5 series)
    fig4_tails.png            – Tail comparison vs Normal / Student-t
    fig5_regime.png           – Regime classification time-series

Run: python paranormal_paper_figures.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator
import scipy.stats as stats
import pathlib, os, warnings

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)

FIG_DIR  = pathlib.Path('artifacts/figures')
DATA_DIR = pathlib.Path('artifacts/data')
FIG_DIR.mkdir(parents=True, exist_ok=True)

from paranormal_improved import ParanormalDistribution, ParanormalFitter

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})
C1, C2, CN = '#1f77b4', '#d62728', '#2ca02c'   # blue, red, green

# ─────────────────────────────────────────────────────────────────────────────
# Helper: load cached series
# ─────────────────────────────────────────────────────────────────────────────

def load_series(fname, col=0):
    path = DATA_DIR / fname
    if not path.exists():
        print(f'  WARNING: {path} not found. Run paranormal_paper_tables.py first.')
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, col].dropna()


def fit_cached(data, maxiter=100):
    return ParanormalFitter.fit(np.asarray(data), maxiter=maxiter, verbose=False)


# ─────────────────────────────────────────────────────────────────────────────
# Load & fit
# ─────────────────────────────────────────────────────────────────────────────

print('Loading and fitting series...')

series = {}
fits   = {}

for key, fname in [
    ('CPI', 'inflation.csv'),
    ('GDP', 'gdp_growth.csv'),
    ('SPY', 'spy_returns.csv'),
    ('TNA', 'tna_returns.csv'),
    ('BTC', 'btc_returns.csv'),
]:
    s = load_series(fname)
    if s is not None:
        series[key] = s
        fits[key]   = fit_cached(s.values)
        print(f'  {key}: n={len(s)}, LL={fits[key]["log_likelihood"]:.1f}')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Density overlay – macro series
# ─────────────────────────────────────────────────────────────────────────────

def plot_density_panel(ax, data, fit_result, title, xlabel, bins=50):
    """Plot histogram + fitted PDF + Normal + Student-t."""
    dist = fit_result['distribution']
    x = np.linspace(data.min(), data.max(), 600)

    ax.hist(data, bins=bins, density=True, color='silver',
            edgecolor='grey', linewidth=0.4, label='Empirical', zorder=1)

    # Paranormal PDF
    ax.plot(x, dist.pdf(x), color=C1, lw=2.5, label='Paranormal', zorder=4)

    # Component PDFs (weighted)
    pdf1 = dist.w1 * dist._skew_t_pdf(x, dist.mu1, dist.sigma1,
                                       dist.alpha1, dist.nu1)
    pdf2 = dist.w2 * dist._skew_t_pdf(x, dist.mu2, dist.sigma2,
                                       dist.alpha2, dist.nu2)
    ax.plot(x, pdf1, color=C1, lw=1.2, ls='--', alpha=0.7, label='Component 1')
    ax.plot(x, pdf2, color=C2, lw=1.2, ls='--', alpha=0.7, label='Component 2')

    # Normal benchmark
    ax.plot(x, stats.norm.pdf(x, data.mean(), data.std()),
            color=CN, lw=1.5, ls=':', label='Normal', zorder=3)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Density')
    ax.legend(loc='upper right', framealpha=0.8)
    ax.xaxis.set_minor_locator(AutoMinorLocator())


print('\nCreating Figure 1: macro density...')
if 'CPI' in series and 'GDP' in series:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plot_density_panel(axes[0], series['CPI'].values, fits['CPI'],
                       'US CPI Inflation (YoY %)', 'Inflation Rate (%)')
    plot_density_panel(axes[1], series['GDP'].values, fits['GDP'],
                       'US Real GDP Growth (Annualised QoQ %)',
                       'Growth Rate (%)', bins=30)
    fig.suptitle('Figure 1: Paranormal Distribution – Macroeconomic Series',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_density_macro.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print('  Saved: fig1_density_macro.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2: Density overlay – financial series
# ─────────────────────────────────────────────────────────────────────────────

print('Creating Figure 2: financial density...')
fin_keys   = [k for k in ['SPY', 'TNA', 'BTC'] if k in series]
fin_titles = {'SPY': 'SPY Daily Log-Returns (%)',
              'TNA': 'TNA Daily Log-Returns (%)',
              'BTC': 'BTC-USD Daily Log-Returns (%)'}

if fin_keys:
    fig, axes = plt.subplots(1, len(fin_keys),
                             figsize=(6*len(fin_keys), 5))
    if len(fin_keys) == 1:
        axes = [axes]
    for ax, key in zip(axes, fin_keys):
        plot_density_panel(ax, series[key].values, fits[key],
                           fin_titles[key], 'Return (%)', bins=80)
    fig.suptitle('Figure 2: Paranormal Distribution – Financial Series',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_density_finance.png', dpi=300,
                bbox_inches='tight')
    plt.close(fig)
    print('  Saved: fig2_density_finance.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3: Q-Q plots
# ─────────────────────────────────────────────────────────────────────────────

def qq_panel(ax, data, dist, title):
    """Q-Q plot: empirical quantiles vs fitted Paranormal quantiles."""
    n = len(data)
    probs = np.linspace(0.01, 0.99, min(n, 500))
    # Approximate theoretical quantiles via large sample from fitted dist
    np.random.seed(SEED)
    big_sample = np.sort(dist.rvs(size=50_000, random_state=SEED))
    theo_q = np.quantile(big_sample, probs)
    emp_q  = np.quantile(data, probs)
    ax.scatter(theo_q, emp_q, s=8, alpha=0.6, color=C1, zorder=3)
    lo = min(theo_q.min(), emp_q.min())
    hi = max(theo_q.max(), emp_q.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='y = x')
    ax.set_xlabel('Theoretical Quantile')
    ax.set_ylabel('Empirical Quantile')
    ax.set_title(title, fontweight='bold')
    ax.legend(fontsize=8)


print('Creating Figure 3: Q-Q plots...')
all_keys = [k for k in ['CPI', 'GDP', 'SPY', 'TNA', 'BTC'] if k in series]
if all_keys:
    ncols = min(3, len(all_keys))
    nrows = (len(all_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5*ncols, 4.8*nrows), squeeze=False)
    titles = {'CPI': 'CPI Inflation', 'GDP': 'GDP Growth',
              'SPY': 'SPY', 'TNA': 'TNA', 'BTC': 'BTC-USD'}
    for idx, key in enumerate(all_keys):
        r, c = divmod(idx, ncols)
        qq_panel(axes[r][c], series[key].values,
                 fits[key]['distribution'], titles[key])
    # hide empty panels
    for idx in range(len(all_keys), nrows*ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
    fig.suptitle('Figure 3: Q-Q Plots vs. Paranormal Distribution',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_qqplot.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: fig3_qqplot.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4: Tail comparison (log scale)
# ─────────────────────────────────────────────────────────────────────────────

def tail_panel(ax, data, fit_result, title):
    """Right-tail comparison: empirical ECDF vs Paranormal vs Normal vs t."""
    dist = fit_result['distribution']
    n = len(data)

    # Empirical exceedance (right tail)
    sorted_data = np.sort(data)
    ecdf = 1 - np.arange(1, n+1) / n
    tail_mask = ecdf < 0.10   # show only top 10%
    if tail_mask.sum() < 10:
        tail_mask = ecdf < 0.3

    x_tail = sorted_data[tail_mask]
    e_tail  = ecdf[tail_mask]

    # Model exceedance
    np.random.seed(SEED)
    big = np.sort(dist.rvs(size=100_000, random_state=SEED))
    theo_ecdf = 1 - np.arange(1, len(big)+1) / len(big)

    # Plot
    ax.semilogy(x_tail, e_tail, 'o', ms=3, color='silver',
                label='Empirical', zorder=2)
    x_q = np.linspace(x_tail.min(), x_tail.max(), 300)
    # Paranormal (sample-based)
    pn_exceed = np.array([(big > xv).mean() for xv in x_q])
    ax.semilogy(x_q, pn_exceed, color=C1, lw=2.0, label='Paranormal', zorder=4)
    # Normal
    ax.semilogy(x_q, 1-stats.norm.cdf(x_q, data.mean(), data.std()),
                color=CN, lw=1.5, ls=':', label='Normal', zorder=3)
    # Student-t
    tp = stats.t.fit(data)
    ax.semilogy(x_q, 1-stats.t.cdf(x_q, *tp),
                color=C2, lw=1.5, ls='--', label='Student-t', zorder=3)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('P(X > x)  [log scale]')
    ax.legend(fontsize=8)


print('Creating Figure 4: tail comparison...')
tail_keys = [k for k in ['SPY', 'TNA', 'BTC'] if k in series]
if tail_keys:
    fig, axes = plt.subplots(1, len(tail_keys),
                             figsize=(6*len(tail_keys), 5))
    if len(tail_keys) == 1:
        axes = [axes]
    titles = {'SPY': 'SPY Right Tail',
              'TNA': 'TNA Right Tail',
              'BTC': 'BTC-USD Right Tail'}
    for ax, key in zip(axes, tail_keys):
        tail_panel(ax, series[key].values, fits[key], titles[key])
    fig.suptitle('Figure 4: Right-Tail Comparison (Log Scale)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_tails.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: fig4_tails.png')


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5: Regime classification time-series
# ─────────────────────────────────────────────────────────────────────────────

def compute_posterior(data_vals, dist):
    """Return P(component 1 | x_i) for each observation."""
    p1 = dist.w1 * dist._skew_t_pdf(data_vals, dist.mu1, dist.sigma1,
                                     dist.alpha1, dist.nu1)
    p2 = dist.w2 * dist._skew_t_pdf(data_vals, dist.mu2, dist.sigma2,
                                     dist.alpha2, dist.nu2)
    total = p1 + p2
    total[total == 0] = np.finfo(float).tiny
    return p1 / total


def regime_panel(ax_ts, ax_prob, series_obj, fit_result, title):
    """Top: time-series coloured by regime. Bottom: posterior P(regime 1)."""
    data_vals = series_obj.values
    dist = fit_result['distribution']
    posterior1 = compute_posterior(data_vals, dist)

    dates = series_obj.index
    regime1 = posterior1 > 0.5

    # Time-series scatter coloured by regime
    ax_ts.scatter(dates[regime1],  data_vals[regime1],
                  c=C2, s=6, alpha=0.7, label=f'Regime 1 (lower)')
    ax_ts.scatter(dates[~regime1], data_vals[~regime1],
                  c=C1, s=6, alpha=0.7, label=f'Regime 2 (upper)')
    ax_ts.axhline(0, color='black', lw=0.8, ls='--')
    ax_ts.set_ylabel('Value')
    ax_ts.set_title(title, fontweight='bold')
    ax_ts.legend(loc='upper right', fontsize=8, markerscale=2)
    ax_ts.set_xticklabels([])

    # Posterior probability
    ax_prob.fill_between(dates, posterior1, alpha=0.4, color=C2)
    ax_prob.plot(dates, posterior1, color=C2, lw=0.8)
    ax_prob.axhline(0.5, color='black', lw=1.0, ls='--')
    ax_prob.set_ylim([0, 1])
    ax_prob.set_ylabel('P(Regime 1)')
    ax_prob.set_xlabel('Date')


print('Creating Figure 5: regime classification...')
regime_keys = [k for k in ['CPI', 'SPY', 'BTC'] if k in series]
if regime_keys:
    nk = len(regime_keys)
    fig = plt.figure(figsize=(6.5*nk, 7))
    outer = gridspec.GridSpec(1, nk, figure=fig, hspace=0.05)
    rtitles = {'CPI': 'CPI Inflation', 'SPY': 'SPY Returns', 'BTC': 'BTC-USD'}
    for col, key in enumerate(regime_keys):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[col], height_ratios=[3, 1.4], hspace=0.08)
        ax_ts   = fig.add_subplot(inner[0])
        ax_prob = fig.add_subplot(inner[1])
        regime_panel(ax_ts, ax_prob, series[key], fits[key], rtitles[key])
    fig.suptitle('Figure 5: Regime Classification via Posterior Probabilities',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_regime.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('  Saved: fig5_regime.png')


print()
print('='*60)
print('All figures saved to artifacts/figures/')
print('='*60)
print()
print('LaTeX usage in paranormal_paper_final.tex:')
for i, fn in enumerate([
    'fig1_density_macro.png',
    'fig2_density_finance.png',
    'fig3_qqplot.png',
    'fig4_tails.png',
    'fig5_regime.png',
], start=1):
    print(f'  \\includegraphics[width=\\linewidth]{{artifacts/figures/{fn}}}')
