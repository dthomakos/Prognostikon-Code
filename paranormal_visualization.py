"""
Paranormal Distribution: Visualization and Diagnostics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats

def plot_fit_diagnostics(data, fitted_dist, title="Paranormal Distribution Fit", 
                         bins=50, save_path=None):
    """
    Comprehensive diagnostic plots for fitted Paranormal distribution

    Parameters
    ----------
    data : array-like
        Observed data
    fitted_dist : ParanormalDistribution
        Fitted distribution object
    title : str
        Plot title
    bins : int
        Number of histogram bins
    save_path : str, optional
        Path to save figure
    """
    data = np.asarray(data).flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Panel 1: Histogram with fitted PDF
    ax1 = axes[0, 0]
    ax1.hist(data, bins=bins, density=True, alpha=0.6, color='steelblue', 
             edgecolor='black', label='Data')

    # Plot fitted distribution
    x_range = np.linspace(data.min(), data.max(), 500)
    pdf_vals = fitted_dist.pdf(x_range)
    ax1.plot(x_range, pdf_vals, 'r-', linewidth=2.5, label='Paranormal PDF')

    # Plot individual components
    pdf1 = fitted_dist.w1 * fitted_dist._skew_t_pdf(x_range, fitted_dist.mu1, 
                                                     fitted_dist.sigma1, 
                                                     fitted_dist.alpha1, 
                                                     fitted_dist.nu1)
    pdf2 = fitted_dist.w2 * fitted_dist._skew_t_pdf(x_range, fitted_dist.mu2, 
                                                     fitted_dist.sigma2, 
                                                     fitted_dist.alpha2, 
                                                     fitted_dist.nu2)
    ax1.plot(x_range, pdf1, 'g--', linewidth=1.5, alpha=0.7, label=f'Component 1 (w={fitted_dist.w1:.2f})')
    ax1.plot(x_range, pdf2, 'b--', linewidth=1.5, alpha=0.7, label=f'Component 2 (w={fitted_dist.w2:.2f})')

    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('Histogram with Fitted PDF', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Q-Q Plot
    ax2 = axes[0, 1]

    # Generate theoretical quantiles from fitted distribution
    n_points = min(100, len(data))
    p_vals = np.linspace(0.01, 0.99, n_points)

    # Empirical quantiles
    empirical_q = np.quantile(data, p_vals)

    # Theoretical quantiles (use sampling approximation)
    np.random.seed(42)
    theoretical_sample = fitted_dist.rvs(size=10000)
    theoretical_q = np.quantile(theoretical_sample, p_vals)

    ax2.scatter(theoretical_q, empirical_q, alpha=0.6, s=30, color='steelblue')

    # Reference line
    min_val = min(theoretical_q.min(), empirical_q.min())
    max_val = max(theoretical_q.max(), empirical_q.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect fit')

    ax2.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax2.set_ylabel('Sample Quantiles', fontsize=11)
    ax2.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Residuals
    ax3 = axes[1, 0]

    # Standardized residuals (approximate)
    sorted_data = np.sort(data)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Get theoretical CDF values (sample a subset for speed)
    sample_indices = np.linspace(0, len(sorted_data)-1, min(200, len(sorted_data)), dtype=int)
    sample_data = sorted_data[sample_indices]
    sample_emp_cdf = empirical_cdf[sample_indices]

    theoretical_cdf = []
    for x in sample_data:
        # Approximate CDF using sample
        theoretical_cdf.append(np.mean(theoretical_sample <= x))
    theoretical_cdf = np.array(theoretical_cdf)

    residuals = sample_emp_cdf - theoretical_cdf

    ax3.scatter(sample_data, residuals, alpha=0.6, s=30, color='steelblue')
    ax3.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Value', fontsize=11)
    ax3.set_ylabel('Empirical CDF - Theoretical CDF', fontsize=11)
    ax3.set_title('CDF Residuals', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Component assignment (posterior probabilities)
    ax4 = axes[1, 1]

    # Calculate posterior probability of belonging to component 1
    pdf1_vals = fitted_dist._skew_t_pdf(data, fitted_dist.mu1, fitted_dist.sigma1,
                                        fitted_dist.alpha1, fitted_dist.nu1)
    pdf2_vals = fitted_dist._skew_t_pdf(data, fitted_dist.mu2, fitted_dist.sigma2,
                                        fitted_dist.alpha2, fitted_dist.nu2)

    posterior_comp1 = (fitted_dist.w1 * pdf1_vals) / (fitted_dist.w1 * pdf1_vals + 
                                                       fitted_dist.w2 * pdf2_vals)

    # Scatter plot colored by component membership
    colors = ['green' if p > 0.5 else 'blue' for p in posterior_comp1]
    ax4.scatter(data, posterior_comp1, c=colors, alpha=0.5, s=20)
    ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Classification boundary')
    ax4.set_xlabel('Value', fontsize=11)
    ax4.set_ylabel('P(Component 1 | Data)', fontsize=11)
    ax4.set_title('Posterior Component Membership', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_comparison(data, fitted_paranormal, title="Model Comparison", save_path=None):
    """
    Compare Paranormal fit against Normal and Student-t
    """
    data = np.asarray(data).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Fit competitors
    mu_norm, std_norm = sp_stats.norm.fit(data)
    params_t = sp_stats.t.fit(data)

    x_range = np.linspace(data.min(), data.max(), 500)

    # Panel 1: PDF comparison
    ax1 = axes[0]
    ax1.hist(data, bins=50, density=True, alpha=0.4, color='gray', 
             edgecolor='black', label='Data')

    pdf_paranormal = fitted_paranormal.pdf(x_range)
    pdf_normal = sp_stats.norm.pdf(x_range, mu_norm, std_norm)
    pdf_t = sp_stats.t.pdf(x_range, *params_t)

    ax1.plot(x_range, pdf_paranormal, 'r-', linewidth=2.5, label='Paranormal')
    ax1.plot(x_range, pdf_normal, 'b--', linewidth=2, label='Normal')
    ax1.plot(x_range, pdf_t, 'g-.', linewidth=2, label='Student-t')

    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('PDF Comparison', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Tail comparison (log scale)
    ax2 = axes[1]

    # Focus on tail region
    tail_cutoff = np.percentile(data, 95)
    tail_mask = x_range >= tail_cutoff

    ax2.semilogy(x_range[tail_mask], pdf_paranormal[tail_mask], 'r-', 
                 linewidth=2.5, label='Paranormal')
    ax2.semilogy(x_range[tail_mask], pdf_normal[tail_mask], 'b--', 
                 linewidth=2, label='Normal')
    ax2.semilogy(x_range[tail_mask], pdf_t[tail_mask], 'g-.', 
                 linewidth=2, label='Student-t')

    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Density (log scale)', fontsize=11)
    ax2.set_title('Right Tail Comparison (95th percentile+)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    return fig


def plot_parameter_interpretation(fitted_dist, title="Parameter Interpretation"):
    """
    Visualize what each parameter controls
    """
    params = {
        'mu1': fitted_dist.mu1, 'sigma1': fitted_dist.sigma1,
        'alpha1': fitted_dist.alpha1, 'nu1': fitted_dist.nu1,
        'mu2': fitted_dist.mu2, 'sigma2': fitted_dist.sigma2,
        'alpha2': fitted_dist.alpha2, 'nu2': fitted_dist.nu2,
        'w1': fitted_dist.w1
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    x_range = np.linspace(-10, 10, 500)

    # Panel 1: Location (μ)
    ax1 = axes[0, 0]
    for mu in [-2, 0, 2]:
        dist = fitted_dist._skew_t_pdf(x_range, mu, 1, 0, 5)
        ax1.plot(x_range, dist, linewidth=2, label=f'μ = {mu}')
    ax1.set_title('Location Parameter (μ)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Scale (σ)
    ax2 = axes[0, 1]
    for sigma in [0.5, 1, 2]:
        dist = fitted_dist._skew_t_pdf(x_range, 0, sigma, 0, 5)
        ax2.plot(x_range, dist, linewidth=2, label=f'σ = {sigma}')
    ax2.set_title('Scale Parameter (σ)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Panel 3: Skewness (α)
    ax3 = axes[1, 0]
    for alpha in [-3, 0, 3]:
        dist = fitted_dist._skew_t_pdf(x_range, 0, 1, alpha, 5)
        ax3.plot(x_range, dist, linewidth=2, label=f'α = {alpha}')
    ax3.set_title('Skewness Parameter (α)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Panel 4: Tail heaviness (ν)
    ax4 = axes[1, 1]
    for nu in [3, 5, 10, 30]:
        dist = fitted_dist._skew_t_pdf(x_range, 0, 1, 0, nu)
        ax4.plot(x_range, dist, linewidth=2, label=f'ν = {nu}')
    ax4.set_title('Degrees of Freedom (ν) - Lower = Heavier Tails', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('x')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
