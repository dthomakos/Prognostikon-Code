"""
Paranormal Distribution: Visualization Examples
Demonstrates all plotting capabilities with real data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Import our modules
from paranormal_improved import ParanormalDistribution, ParanormalFitter
from paranormal_visualization import (plot_fit_diagnostics, plot_comparison,
                                      plot_parameter_interpretation)

print("="*80)
print("PARANORMAL DISTRIBUTION - VISUALIZATION EXAMPLES")
print("="*80)

# ============================================================================
# EXAMPLE 1: Simulated Data Visualization
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 1: SIMULATED BIMODAL DATA")
print("="*80)

# Generate synthetic data with known parameters
np.random.seed(42)
print("\nGenerating bimodal, skewed, fat-tailed data...")

# Component 1: Negative regime
n1 = 600
mu1, sigma1, alpha1, nu1 = -2.0, 1.5, -2.0, 3.5
u1 = np.random.chisquare(nu1, n1) / nu1
z1 = np.random.normal(0, 1, n1)
delta1 = alpha1 / np.sqrt(1 + alpha1**2)
comp1 = mu1 + sigma1 * (delta1 * np.abs(z1) + np.sqrt(1 - delta1**2) * z1) / np.sqrt(u1)

# Component 2: Positive regime
n2 = 900
mu2, sigma2, alpha2, nu2 = 2.5, 1.0, 1.5, 4.0
u2 = np.random.chisquare(nu2, n2) / nu2
z2 = np.random.normal(0, 1, n2)
delta2 = alpha2 / np.sqrt(1 + alpha2**2)
comp2 = mu2 + sigma2 * (delta2 * np.abs(z2) + np.sqrt(1 - delta2**2) * z2) / np.sqrt(u2)

# Combine
data_sim = np.concatenate([comp1, comp2])
np.random.shuffle(data_sim)

print(f"✓ Generated {len(data_sim)} observations")
print(f"  Component 1: n={n1}, μ={mu1}, σ={sigma1}, α={alpha1}, ν={nu1}")
print(f"  Component 2: n={n2}, μ={mu2}, σ={sigma2}, α={alpha2}, ν={nu2}")

# Fit distribution
print("\nFitting Paranormal distribution...")
fit_sim = ParanormalFitter.fit(data_sim, maxiter=60, verbose=False)
dist_sim = fit_sim['distribution']

print(f"✓ Fitted successfully (AIC: {fit_sim['AIC']:.2f})")

# Create diagnostic plots
print("\nCreating diagnostic plots...")
fig1 = plot_fit_diagnostics(data_sim, dist_sim,
                            title="Simulated Data: Paranormal Distribution Fit",
                            save_path="fig1_diagnostics_simulated.png")
plt.close()
print("✓ Saved: fig1_diagnostics_simulated.png")

# Create comparison plots
print("Creating model comparison plots...")
fig2 = plot_comparison(data_sim, dist_sim,
                      title="Model Comparison: Simulated Data",
                      save_path="fig2_comparison_simulated.png")
plt.close()
print("✓ Saved: fig2_comparison_simulated.png")


# ============================================================================
# EXAMPLE 2: S&P 500 Returns Visualization
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 2: S&P 500 DAILY RETURNS")
print("="*80)

try:
    print("\nDownloading S&P 500 data...")
    sp500 = yf.download('^GSPC', start='2020-01-01', end='2025-12-28', progress=False)
    sp500_returns = sp500['Close'].pct_change().dropna() * 100

    print(f"✓ Downloaded {len(sp500_returns)} daily returns")

    # Fit distribution
    print("Fitting Paranormal distribution...")
    fit_sp500 = ParanormalFitter.fit(sp500_returns.values, maxiter=80, verbose=False)
    dist_sp500 = fit_sp500['distribution']

    print(f"✓ Fitted successfully (AIC: {fit_sp500['AIC']:.2f})")

    # Create diagnostic plots
    print("Creating diagnostic plots...")
    fig3 = plot_fit_diagnostics(sp500_returns.values, dist_sp500,
                                title="S&P 500 Returns (2020-2025): Paranormal Distribution",
                                save_path="fig3_diagnostics_sp500.png")
    plt.close()
    print("✓ Saved: fig3_diagnostics_sp500.png")

    # Create comparison plots
    print("Creating model comparison plots...")
    fig4 = plot_comparison(sp500_returns.values, dist_sp500,
                          title="S&P 500 Returns: Model Comparison",
                          save_path="fig4_comparison_sp500.png")
    plt.close()
    print("✓ Saved: fig4_comparison_sp500.png")

except Exception as e:
    print(f"Error with S&P 500 data: {e}")


# ============================================================================
# EXAMPLE 3: Bitcoin Returns Visualization
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 3: BITCOIN DAILY RETURNS")
print("="*80)

try:
    print("\nDownloading Bitcoin data...")
    btc = yf.download('BTC-USD', start='2020-01-01', end='2025-12-28', progress=False)
    btc_returns = btc['Close'].pct_change().dropna() * 100

    print(f"✓ Downloaded {len(btc_returns)} daily returns")

    # Fit distribution
    print("Fitting Paranormal distribution...")
    fit_btc = ParanormalFitter.fit(btc_returns.values, maxiter=80, verbose=False)
    dist_btc = fit_btc['distribution']

    print(f"✓ Fitted successfully (AIC: {fit_btc['AIC']:.2f})")

    # Create diagnostic plots
    print("Creating diagnostic plots...")
    fig5 = plot_fit_diagnostics(btc_returns.values, dist_btc,
                                title="Bitcoin Returns (2020-2025): Paranormal Distribution",
                                save_path="fig5_diagnostics_bitcoin.png")
    plt.close()
    print("✓ Saved: fig5_diagnostics_bitcoin.png")

    # Create comparison plots
    print("Creating model comparison plots...")
    fig6 = plot_comparison(btc_returns.values, dist_btc,
                          title="Bitcoin Returns: Model Comparison",
                          save_path="fig6_comparison_bitcoin.png")
    plt.close()
    print("✓ Saved: fig6_comparison_bitcoin.png")

except Exception as e:
    print(f"Error with Bitcoin data: {e}")


# ============================================================================
# EXAMPLE 4: Parameter Interpretation Visualization
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 4: PARAMETER INTERPRETATION")
print("="*80)

print("\nCreating parameter interpretation plots...")

# Use the fitted S&P 500 distribution if available, otherwise simulated
if 'dist_sp500' in locals():
    fig7 = plot_parameter_interpretation(dist_sp500,
                                        title="Parameter Effects: S&P 500 Distribution")
    plt.savefig("fig7_parameter_interpretation.png", dpi=300, bbox_inches='tight')
    plt.close()
else:
    fig7 = plot_parameter_interpretation(dist_sim,
                                        title="Parameter Effects: Simulated Distribution")
    plt.savefig("fig7_parameter_interpretation.png", dpi=300, bbox_inches='tight')
    plt.close()

print("✓ Saved: fig7_parameter_interpretation.png")


# ============================================================================
# EXAMPLE 5: Custom Advanced Visualization - Regime Classification
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 5: REGIME CLASSIFICATION PLOT")
print("="*80)

print("\nCreating regime classification visualization...")

# Use S&P 500 data if available
if 'sp500_returns' in locals() and 'dist_sp500' in locals():
    data_viz = sp500_returns.values
    dist_viz = dist_sp500
    dates_viz = sp500_returns.index
    title_viz = "S&P 500 Returns: Regime Classification"
    filename_viz = "fig8_regime_classification_sp500.png"
else:
    data_viz = data_sim
    dist_viz = dist_sim
    dates_viz = None
    title_viz = "Simulated Data: Regime Classification"
    filename_viz = "fig8_regime_classification_simulated.png"

# Calculate posterior probabilities
pdf1 = dist_viz._skew_t_pdf(data_viz, dist_viz.mu1, dist_viz.sigma1,
                            dist_viz.alpha1, dist_viz.nu1)
pdf2 = dist_viz._skew_t_pdf(data_viz, dist_viz.mu2, dist_viz.sigma2,
                            dist_viz.alpha2, dist_viz.nu2)

posterior_regime1 = (dist_viz.w1 * pdf1) / (dist_viz.w1 * pdf1 + dist_viz.w2 * pdf2)

# Create classification plot
fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle(title_viz, fontsize=16, fontweight='bold')

# Top panel: Time series with regime coloring
ax1 = axes[0]
if dates_viz is not None:
    # Color by regime
    colors = ['red' if p < 0.5 else 'green' for p in posterior_regime1]
    ax1.scatter(dates_viz, data_viz, c=colors, alpha=0.5, s=10)
    ax1.set_xlabel('Date', fontsize=12)
else:
    colors = ['red' if p < 0.5 else 'green' for p in posterior_regime1]
    ax1.scatter(range(len(data_viz)), data_viz, c=colors, alpha=0.5, s=10)
    ax1.set_xlabel('Observation Index', fontsize=12)

ax1.set_ylabel('Returns (%)', fontsize=12)
ax1.set_title('Returns Colored by Regime (Red = Regime 1, Green = Regime 2)',
             fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)

# Bottom panel: Regime probability over time
ax2 = axes[1]
if dates_viz is not None:
    ax2.plot(dates_viz, posterior_regime1.flatten(), color='blue', linewidth=1, alpha=0.7)
    ax2.fill_between(dates_viz, 0, posterior_regime1.flatten(), alpha=0.3, color='blue')
    ax2.set_xlabel('Date', fontsize=12)
else:
    ax2.plot(range(len(data_viz)), posterior_regime1.flatten(), color='blue', linewidth=1, alpha=0.7)
    ax2.fill_between(range(len(data_viz)), 0, posterior_regime1.flatten(), alpha=0.3, color='blue')
    ax2.set_xlabel('Observation Index', fontsize=12)

ax2.set_ylabel('P(Regime 1)', fontsize=12)
ax2.set_title('Posterior Probability of Regime 1', fontsize=12, fontweight='bold')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Classification threshold')
ax2.set_ylim([0, 1])
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig(filename_viz, dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ Saved: {filename_viz}")


# ============================================================================
# EXAMPLE 6: Multi-Asset Comparison Visualization
# ============================================================================

print("\n" + "="*80)
print("EXAMPLE 6: MULTI-ASSET COMPARISON")
print("="*80)

print("\nDownloading multiple assets and fitting distributions...")

assets = {
    '^GSPC': 'S&P 500',
    'BTC-USD': 'Bitcoin',
    '^VIX': 'VIX'
}

fitted_dists = {}
for ticker, name in assets.items():
    try:
        data = yf.download(ticker, start='2020-01-01', end='2025-12-28', progress=False)
        if ticker == '^VIX':
            returns = data['Close'].diff().dropna()
        else:
            returns = data['Close'].pct_change().dropna() * 100

        fit_result = ParanormalFitter.fit(returns.values, maxiter=60, verbose=False)
        fitted_dists[name] = {
            'data': returns.values,
            'dist': fit_result['distribution'],
            'params': fit_result['parameters']
        }
        print(f"✓ Fitted {name}")
    except Exception as e:
        print(f"✗ Could not process {name}: {e}")

if fitted_dists:
    # Create comparison figure
    fig, axes = plt.subplots(len(fitted_dists), 2, figsize=(14, 5*len(fitted_dists)))
    if len(fitted_dists) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Multi-Asset Paranormal Distribution Comparison',
                 fontsize=16, fontweight='bold')

    for idx, (asset_name, asset_data) in enumerate(fitted_dists.items()):
        data = asset_data['data']
        dist = asset_data['dist']
        params = asset_data['params']

        # Left panel: Histogram with PDF
        ax_left = axes[idx, 0]
        ax_left.hist(data, bins=50, density=True, alpha=0.6, color='steelblue',
                    edgecolor='black')

        x_range = np.linspace(data.min(), data.max(), 500)
        pdf_vals = dist.pdf(x_range)
        ax_left.plot(x_range, pdf_vals, 'r-', linewidth=2.5, label='Paranormal PDF')

        ax_left.set_xlabel('Value', fontsize=11)
        ax_left.set_ylabel('Density', fontsize=11)
        ax_left.set_title(f'{asset_name}: PDF Fit', fontsize=12, fontweight='bold')
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)

        # Right panel: Parameter summary
        ax_right = axes[idx, 1]
        ax_right.axis('off')

        param_text = f"""
        {asset_name} - Paranormal Parameters

        Component 1:
          μ₁ = {params['mu1']:7.3f}
          σ₁ = {params['sigma1']:7.3f}
          α₁ = {params['alpha1']:7.3f}
          ν₁ = {params['nu1']:7.3f}
          w₁ = {params['w1']:7.3f}

        Component 2:
          μ₂ = {params['mu2']:7.3f}
          σ₂ = {params['sigma2']:7.3f}
          α₂ = {params['alpha2']:7.3f}
          ν₂ = {params['nu2']:7.3f}
          w₂ = {params['w2']:7.3f}

        Data Statistics:
          Mean = {data.mean():7.3f}
          Std  = {data.std():7.3f}
          Skew = {stats.skew(data)[0]:7.3f}
          Kurt = {stats.kurtosis(data)[0]:7.3f}
        """

        ax_right.text(0.1, 0.5, param_text, fontsize=10, family='monospace',
                     verticalalignment='center')

    plt.tight_layout()
    plt.savefig("fig9_multi_asset_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("\n✓ Saved: fig9_multi_asset_comparison.png")


# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION EXAMPLES COMPLETE")
print("="*80)

print("\nGenerated Figures:")
print("  1. fig1_diagnostics_simulated.png      - 4-panel diagnostics (simulated)")
print("  2. fig2_comparison_simulated.png        - Model comparison (simulated)")
print("  3. fig3_diagnostics_sp500.png           - 4-panel diagnostics (S&P 500)")
print("  4. fig4_comparison_sp500.png            - Model comparison (S&P 500)")
print("  5. fig5_diagnostics_bitcoin.png         - 4-panel diagnostics (Bitcoin)")
print("  6. fig6_comparison_bitcoin.png          - Model comparison (Bitcoin)")
print("  7. fig7_parameter_interpretation.png    - Parameter effects")
print("  8. fig8_regime_classification_*.png     - Regime identification")
print("  9. fig9_multi_asset_comparison.png      - Cross-asset summary")

print("\nAll figures saved as high-resolution PNG files (300 DPI)")
print("Ready for publication or presentation!")
print("="*80)
