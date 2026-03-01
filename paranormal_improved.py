"""
The Paranormal Distribution - Production Implementation
A bimodal, asymmetric, fat-tailed probability distribution

Enhanced version with diagnostics, visualization utilities, and real-world applications
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad
from scipy.special import gamma as gamma_func
import warnings
warnings.filterwarnings('ignore')

class ParanormalDistribution:
    """
    The Paranormal Distribution: Bimodal, Asymmetric, Fat-Tailed

    f(x; Θ) = w₁·ST(x; μ₁, σ₁, α₁, ν₁) + w₂·ST(x; μ₂, σ₂, α₂, ν₂)

    Parameters
    ----------
    mu1, mu2 : float
        Location parameters (mode centers)
    sigma1, sigma2 : float > 0
        Scale parameters (dispersion)
    alpha1, alpha2 : float
        Skewness parameters
    nu1, nu2 : float > 2
        Degrees of freedom (tail heaviness, lower = heavier)
    w1 : float in (0,1)
        Mixing weight for component 1
    """

    def __init__(self, mu1, sigma1, alpha1, nu1, mu2, sigma2, alpha2, nu2, w1):
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.alpha1 = alpha1
        self.nu1 = nu1
        self.mu2 = mu2
        self.sigma2 = sigma2
        self.alpha2 = alpha2
        self.nu2 = nu2
        self.w1 = w1
        self.w2 = 1 - w1

    def _skew_t_pdf(self, x, mu, sigma, alpha, nu):
        """Skew-t PDF (Azzalini formulation)"""
        z = (x - mu) / sigma
        t_pdf = stats.t.pdf(z, df=nu)
        t_cdf = stats.t.cdf(alpha * z * np.sqrt((nu + 1) / (nu + z**2)), df=nu + 1)
        return (2 / sigma) * t_pdf * t_cdf

    def pdf(self, x):
        """Probability density function"""
        x = np.atleast_1d(x)
        pdf1 = self._skew_t_pdf(x, self.mu1, self.sigma1, self.alpha1, self.nu1)
        pdf2 = self._skew_t_pdf(x, self.mu2, self.sigma2, self.alpha2, self.nu2)
        return self.w1 * pdf1 + self.w2 * pdf2

    def logpdf(self, x):
        """Log probability density function"""
        return np.log(np.maximum(self.pdf(x), 1e-300))

    def cdf(self, x):
        """Cumulative distribution function (numerical)"""
        x_scalar = np.isscalar(x)
        x = np.atleast_1d(x)

        cdf_vals = np.zeros_like(x, dtype=float)
        for i, xi in enumerate(x):
            result, _ = quad(lambda t: self.pdf(t), -50, xi, limit=100)
            cdf_vals[i] = result

        return cdf_vals[0] if x_scalar else cdf_vals

    def ppf(self, q, tol=1e-6):
        """Percent point function (inverse CDF) via bisection"""
        from scipy.optimize import brentq

        q_scalar = np.isscalar(q)
        q = np.atleast_1d(q)

        ppf_vals = np.zeros_like(q, dtype=float)
        for i, qi in enumerate(q):
            if qi <= 0:
                ppf_vals[i] = -np.inf
            elif qi >= 1:
                ppf_vals[i] = np.inf
            else:
                # Find x such that CDF(x) = qi
                # Start with reasonable bounds
                lower, upper = -100, 100
                ppf_vals[i] = brentq(lambda x: self.cdf(x) - qi, lower, upper, xtol=tol)

        return ppf_vals[0] if q_scalar else ppf_vals

    def rvs(self, size=1, random_state=None):
        """Generate random variates"""
        if random_state is not None:
            np.random.seed(random_state)

        n_from_comp1 = np.random.binomial(size, self.w1)
        n_from_comp2 = size - n_from_comp1

        samples = []

        if n_from_comp1 > 0:
            samples1 = self._skew_t_rvs(self.mu1, self.sigma1, self.alpha1, 
                                        self.nu1, n_from_comp1)
            samples.append(samples1)

        if n_from_comp2 > 0:
            samples2 = self._skew_t_rvs(self.mu2, self.sigma2, self.alpha2, 
                                        self.nu2, n_from_comp2)
            samples.append(samples2)

        all_samples = np.concatenate(samples) if len(samples) > 0 else np.array([])
        np.random.shuffle(all_samples)
        return all_samples

    def _skew_t_rvs(self, mu, sigma, alpha, nu, size):
        """Generate from skew-t via stochastic representation"""
        u = np.random.chisquare(nu, size) / nu
        z = np.random.normal(0, 1, size)
        delta = alpha / np.sqrt(1 + alpha**2)

        x = mu + sigma * (delta * np.abs(z) + np.sqrt(1 - delta**2) * z) / np.sqrt(u)
        return x

    def moments(self, order=4):
        """Calculate distribution moments"""
        moments = {}

        # Numerical integration for raw moments
        for k in range(1, order + 1):
            integrand = lambda x: x**k * self.pdf(x)
            moment, _ = quad(integrand, -50, 50, limit=100)
            moments[f'raw_moment_{k}'] = moment

        # Derived statistics
        mean = moments['raw_moment_1']
        variance = moments['raw_moment_2'] - mean**2
        std = np.sqrt(variance)

        moments['mean'] = mean
        moments['variance'] = variance
        moments['std'] = std

        if order >= 3:
            mu3 = (moments['raw_moment_3'] - 3*mean*moments['raw_moment_2'] + 
                   2*mean**3)
            skewness = mu3 / std**3
            moments['skewness'] = skewness

        if order >= 4:
            mu4 = (moments['raw_moment_4'] - 4*mean*moments['raw_moment_3'] + 
                   6*mean**2*moments['raw_moment_2'] - 3*mean**4)
            excess_kurtosis = mu4 / variance**2 - 3
            moments['excess_kurtosis'] = excess_kurtosis
            moments['kurtosis'] = excess_kurtosis + 3

        return moments

    def summary(self):
        """Print distribution summary"""
        print("="*70)
        print("PARANORMAL DISTRIBUTION SUMMARY")
        print("="*70)
        print("\nComponent 1:")
        print(f"  Location (μ₁):     {self.mu1:8.4f}")
        print(f"  Scale (σ₁):        {self.sigma1:8.4f}")
        print(f"  Skewness (α₁):     {self.alpha1:8.4f}")
        print(f"  Tail DF (ν₁):      {self.nu1:8.4f}")
        print(f"  Weight (w₁):       {self.w1:8.4f}")

        print("\nComponent 2:")
        print(f"  Location (μ₂):     {self.mu2:8.4f}")
        print(f"  Scale (σ₂):        {self.sigma2:8.4f}")
        print(f"  Skewness (α₂):     {self.alpha2:8.4f}")
        print(f"  Tail DF (ν₂):      {self.nu2:8.4f}")
        print(f"  Weight (w₂):       {self.w2:8.4f}")

        try:
            moments = self.moments(order=4)
            print("\nDistribution Moments:")
            print(f"  Mean:              {moments['mean']:8.4f}")
            print(f"  Std Deviation:     {moments['std']:8.4f}")
            print(f"  Skewness:          {moments['skewness']:8.4f}")
            print(f"  Excess Kurtosis:   {moments['excess_kurtosis']:8.4f}")
        except:
            print("\n(Moment calculation unavailable)")

        print("="*70)


class ParanormalFitter:
    """
    Maximum Likelihood Estimation for Paranormal Distribution
    """

    @staticmethod
    def negative_log_likelihood(params, data):
        """Negative log-likelihood objective"""
        mu1, sigma1, alpha1, nu1, mu2, sigma2, alpha2, nu2, w1 = params

        # Parameter constraints
        if sigma1 <= 0.001 or sigma2 <= 0.001:
            return 1e10
        if nu1 <= 2.01 or nu2 <= 2.01:
            return 1e10
        if w1 <= 0.001 or w1 >= 0.999:
            return 1e10

        try:
            dist = ParanormalDistribution(mu1, sigma1, alpha1, nu1,
                                         mu2, sigma2, alpha2, nu2, w1)
            pdf_vals = dist.pdf(data)
            pdf_vals = np.maximum(pdf_vals, 1e-300)
            nll = -np.sum(np.log(pdf_vals))

            if np.isnan(nll) or np.isinf(nll):
                return 1e10
            return nll
        except:
            return 1e10

    @staticmethod
    def initialize_parameters(data):
        """Smart initialization via clustering"""
        from sklearn.cluster import KMeans

        data = np.asarray(data).reshape(-1, 1)

        # K-means with k=2
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data)

        # Extract clusters
        cluster1 = data[labels == 0].flatten()
        cluster2 = data[labels == 1].flatten()

        # Ensure cluster1 has lower mean (for identifiability)
        if np.mean(cluster1) > np.mean(cluster2):
            cluster1, cluster2 = cluster2, cluster1

        # Initialize parameters
        mu1 = np.mean(cluster1)
        sigma1 = np.std(cluster1) + 0.01
        mu2 = np.mean(cluster2)
        sigma2 = np.std(cluster2) + 0.01
        w1 = len(cluster1) / len(data)

        return mu1, sigma1, mu2, sigma2, w1

    @staticmethod
    def fit(data, method='differential_evolution', maxiter=100, verbose=True):
        """
        Fit Paranormal distribution via MLE

        Parameters
        ----------
        data : array-like
            Observed data
        method : str
            'differential_evolution' (global) or 'local' (L-BFGS-B)
        maxiter : int
            Maximum iterations
        verbose : bool
            Print progress

        Returns
        -------
        dict with fitted distribution, parameters, diagnostics
        """
        data = np.asarray(data).flatten()
        n = len(data)

        if verbose:
            print(f"\nFitting Paranormal distribution to {n} observations...")

        # Initialize
        mu1_init, sigma1_init, mu2_init, sigma2_init, w1_init =             ParanormalFitter.initialize_parameters(data)

        if verbose:
            print(f"Initial parameters: μ₁={mu1_init:.3f}, μ₂={mu2_init:.3f}, w₁={w1_init:.3f}")

        # Bounds
        data_range = np.ptp(data)
        data_mean = np.mean(data)
        data_std = np.std(data)

        bounds = [
            (data_mean - 3*data_std, data_mean + 3*data_std),  # mu1
            (0.01, data_std * 10),                              # sigma1
            (-10, 10),                                          # alpha1
            (2.05, 50),                                         # nu1
            (data_mean - 3*data_std, data_mean + 3*data_std),  # mu2
            (0.01, data_std * 10),                              # sigma2
            (-10, 10),                                          # alpha2
            (2.05, 50),                                         # nu2
            (0.05, 0.95)                                        # w1
        ]

        if method == 'differential_evolution':
            if verbose:
                print("Running differential evolution (global optimization)...")

            result = differential_evolution(
                ParanormalFitter.negative_log_likelihood,
                bounds,
                args=(data,),
                maxiter=maxiter,
                seed=42,
                atol=1e-6,
                tol=1e-6,
                workers=1,
                updating='deferred',
                disp=False
            )
        else:
            # Local optimization
            x0 = [mu1_init, sigma1_init, 0, 5, 
                  mu2_init, sigma2_init, 0, 5, w1_init]

            result = minimize(
                ParanormalFitter.negative_log_likelihood,
                x0,
                args=(data,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter}
            )

        # Extract results
        params = result.x
        dist = ParanormalDistribution(*params)

        nll = result.fun
        k = 9  # number of parameters
        log_likelihood = -nll
        aic = 2 * k + 2 * nll
        bic = k * np.log(n) + 2 * nll

        if verbose:
            print(f"Optimization completed: convergence = {result.success}")
            print(f"Log-likelihood: {log_likelihood:.2f}")
            print(f"AIC: {aic:.2f}, BIC: {bic:.2f}")

        return {
            'distribution': dist,
            'parameters': {
                'mu1': params[0], 'sigma1': params[1], 'alpha1': params[2], 'nu1': params[3],
                'mu2': params[4], 'sigma2': params[5], 'alpha2': params[6], 'nu2': params[7],
                'w1': params[8], 'w2': 1 - params[8]
            },
            'log_likelihood': log_likelihood,
            'AIC': aic,
            'BIC': bic,
            'n_obs': n,
            'convergence': result.success,
            'n_iterations': result.nit if hasattr(result, 'nit') else maxiter
        }

    @staticmethod
    def compare_models(data, verbose=True):
        """Compare Paranormal against standard distributions"""
        data = np.asarray(data).flatten()
        n = len(data)

        results = {}

        # Normal
        mu_norm, std_norm = stats.norm.fit(data)
        ll_norm = np.sum(stats.norm.logpdf(data, mu_norm, std_norm))
        results['Normal'] = {
            'll': ll_norm,
            'aic': 2*2 - 2*ll_norm,
            'bic': 2*np.log(n) - 2*ll_norm,
            'params': {'mu': mu_norm, 'sigma': std_norm}
        }

        # Student-t
        params_t = stats.t.fit(data)
        ll_t = np.sum(stats.t.logpdf(data, *params_t))
        results['Student-t'] = {
            'll': ll_t,
            'aic': 2*3 - 2*ll_t,
            'bic': 3*np.log(n) - 2*ll_t,
            'params': {'df': params_t[0], 'loc': params_t[1], 'scale': params_t[2]}
        }

        # Fit Paranormal
        if verbose:
            print("\nFitting Paranormal distribution...")
        fit_para = ParanormalFitter.fit(data, verbose=False)
        results['Paranormal'] = {
            'll': fit_para['log_likelihood'],
            'aic': fit_para['AIC'],
            'bic': fit_para['BIC'],
            'params': fit_para['parameters']
        }

        # Display results
        if verbose:
            print("\n" + "="*70)
            print("MODEL COMPARISON")
            print("="*70)
            print(f"{'Model':<15} {'Log-Lik':<12} {'AIC':<12} {'BIC':<12}")
            print("-"*70)
            for model_name, res in results.items():
                print(f"{model_name:<15} {res['ll']:<12.2f} {res['aic']:<12.2f} {res['bic']:<12.2f}")
            print("="*70)

            # Find best model by AIC
            best_model = min(results.keys(), key=lambda k: results[k]['aic'])
            print(f"\nBest model by AIC: {best_model}")

        return results, fit_para['distribution']


def goodness_of_fit(data, fitted_dist, n_bins=50):
    """
    Goodness-of-fit diagnostics

    Returns chi-square statistic and p-value
    """
    data = np.asarray(data).flatten()

    # Create bins
    hist, bin_edges = np.histogram(data, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Expected frequencies
    expected = []
    for i in range(len(bin_edges) - 1):
        prob = quad(lambda x: fitted_dist.pdf(x), bin_edges[i], bin_edges[i+1])[0]
        expected.append(prob * len(data))
    expected = np.array(expected)

    # Chi-square test (merge bins with low expected counts)
    mask = expected >= 5
    if np.sum(mask) < 5:  # Need at least 5 bins
        return None, None

    chi2_stat = np.sum((hist[mask] - expected[mask])**2 / expected[mask])
    df = np.sum(mask) - 9 - 1  # bins - parameters - 1

    if df <= 0:
        return None, None

    p_value = 1 - stats.chi2.cdf(chi2_stat, df)

    return chi2_stat, p_value
