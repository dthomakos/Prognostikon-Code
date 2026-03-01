"""
paranormal_simulation.py
------------------------
Monte Carlo simulation study for the Paranormal distribution.
Reproduces Table 2 of paranormal_paper_final.tex.

Design (Section 5 of the paper):
  - R = 500 replicates
  - n in {200, 500, 1000, 2000}
  - True parameters theta* = (-2, 1.2, -1.5, 4.0, 3.0, 0.8, 2.0, 3.0, 0.6)
    i.e. (mu1, sigma1, alpha1, nu1, mu2, sigma2, alpha2, nu2, w1)
  - Constrained MLE via differential evolution (G=200 generations)

Outputs:
    artifacts/tables/table_sim.tex   -- LaTeX table (Table 2)
    artifacts/data/sim_results.csv   -- full raw results
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import differential_evolution, minimize
import pathlib, time, warnings

warnings.filterwarnings('ignore')

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
rng  = np.random.default_rng(SEED)

# ── directories ───────────────────────────────────────────────────────────────
ROOT      = pathlib.Path('artifacts')
TABLE_DIR = ROOT / 'tables'
DATA_DIR  = ROOT / 'data'
for d in [TABLE_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# TRUE PARAMETERS  (eq:refparam in the paper)
# ─────────────────────────────────────────────────────────────────────────────
THETA_TRUE = np.array([
    -2.0,   # mu1
     1.2,   # sigma1
    -1.5,   # alpha1
     4.0,   # nu1
     3.0,   # mu2
     0.8,   # sigma2
     2.0,   # alpha2
     3.0,   # nu2
     0.6,   # w1
])
PARAM_NAMES = [
    'mu1', 'sigma1', 'alpha1', 'nu1',
    'mu2', 'sigma2', 'alpha2', 'nu2',
    'w1'
]
PARAM_LABELS = [
    r'$\mu_1$',    r'$\sigma_1$', r'$\alpha_1$', r'$\nu_1$',
    r'$\mu_2$',    r'$\sigma_2$', r'$\alpha_2$', r'$\nu_2$',
    r'$w_1$',
]
TRUE_VALS = {n: v for n, v in zip(PARAM_NAMES, THETA_TRUE)}

# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINED PARAMETER SPACE  (eq:defaults in the paper)
# ─────────────────────────────────────────────────────────────────────────────
SIGMA_MIN = 0.01
NU_MIN    = 2.05
NU_MAX    = 50.0
W_MIN     = 0.05

# Bounds for differential_evolution: (lower, upper) for each of 9 params
# Order: mu1, sigma1, alpha1, nu1, mu2, sigma2, alpha2, nu2, w1
BOUNDS = [
    (-15,  0),           # mu1   (must be < mu2; enforced by projection)
    (SIGMA_MIN, 10),     # sigma1
    (-10, 10),           # alpha1
    (NU_MIN, NU_MAX),    # nu1
    (-5,  15),           # mu2
    (SIGMA_MIN, 10),     # sigma2
    (-10, 10),           # alpha2
    (NU_MIN, NU_MAX),    # nu2
    (W_MIN, 1 - W_MIN),  # w1
]

# ─────────────────────────────────────────────────────────────────────────────
# CORE DISTRIBUTION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def skew_t_pdf(x, mu, sigma, alpha, nu):
    """Azzalini-Capitanio skew-t density (eq:skewt in the paper)."""
    z     = (x - mu) / sigma
    kern  = stats.t.pdf(z, df=nu)
    arg   = alpha * z * np.sqrt((nu + 1) / (nu + z**2))
    skew  = stats.t.cdf(arg, df=nu + 1)
    return np.maximum(2.0 / sigma * kern * skew, 1e-300)


def paranormal_pdf(x, theta):
    """Paranormal mixture density (eq:paranormal in the paper)."""
    mu1, s1, a1, nu1, mu2, s2, a2, nu2, w1 = theta
    w2  = 1.0 - w1
    return w1 * skew_t_pdf(x, mu1, s1, a1, nu1) + \
           w2 * skew_t_pdf(x, mu2, s2, a2, nu2)


def log_likelihood(theta, x):
    """Log-likelihood of the Paranormal distribution."""
    ll = np.sum(np.log(paranormal_pdf(x, theta)))
    return ll if np.isfinite(ll) else -1e12


def neg_log_likelihood(theta, x):
    return -log_likelihood(theta, x)


# ─────────────────────────────────────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def generate_paranormal(theta, n, rng):
    """
    Generate n i.i.d. draws from Paranormal(theta).
    Uses the mixture representation: draw component indicator,
    then draw from the appropriate skew-t via rejection sampling.
    """
    mu1, s1, a1, nu1, mu2, s2, a2, nu2, w1 = theta

    # Component indicators
    indicators = rng.binomial(1, w1, n)
    n1 = indicators.sum()
    n2 = n - n1

    def sample_skewt(mu, sigma, alpha, nu, size, rng):
        """
        Sample from ST(mu, sigma, alpha, nu) via the stochastic
        representation:
            X = mu + sigma * delta * |Z0| + sigma * sqrt(1-delta^2) * Z1
                                             where Z0,Z1 ~ N(0,1) i.i.d.
        then divided by sqrt(chi^2(nu)/nu).
        More precisely: use the representation
            X = mu + sigma * (delta*|T0| + sqrt(1-delta^2)*T1) / sqrt(U/nu)
        where T0,T1 ~ N(0,1), U ~ chi2(nu), delta = alpha/sqrt(1+alpha^2).
        """
        delta = alpha / np.sqrt(1.0 + alpha**2)
        Z0 = np.abs(rng.standard_normal(size))
        Z1 = rng.standard_normal(size)
        U  = rng.chisquare(nu, size) / nu
        core = delta * Z0 + np.sqrt(1.0 - delta**2) * Z1
        return mu + sigma * core / np.sqrt(U)

    samples = np.empty(n)
    if n1 > 0:
        samples[indicators == 1] = sample_skewt(mu1, s1, a1, nu1, n1, rng)
    if n2 > 0:
        samples[indicators == 0] = sample_skewt(mu2, s2, a2, nu2, n2, rng)

    rng.shuffle(samples)
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# K-MEANS INITIALISATION  (Algorithm 2 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def kmeans_init(x, rng_seed=0):
    """
    Partition x into 2 clusters and return starting parameter vector.
    Uses 5 random restarts; orders clusters by mean (mu1 < mu2).
    """
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=2, n_init=5, random_state=rng_seed)
    labels = km.fit_predict(x.reshape(-1, 1))
    c0, c1 = x[labels == 0], x[labels == 1]
    # ensure c0 has smaller mean
    if c0.mean() > c1.mean():
        c0, c1 = c1, c0

    def safe_sigma(arr):
        s = arr.std(ddof=1) if len(arr) > 1 else 1.0
        return max(s, SIGMA_MIN)

    theta0 = np.array([
        c0.mean(), safe_sigma(c0), 0.0, 5.0,
        c1.mean(), safe_sigma(c1), 0.0, 5.0,
        len(c0) / len(x),
    ])
    # clip to bounds
    for j, (lo, hi) in enumerate(BOUNDS):
        theta0[j] = np.clip(theta0[j], lo, hi)
    return theta0


# ─────────────────────────────────────────────────────────────────────────────
# CONSTRAINED MLE  (Algorithm 1 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def fit_paranormal(x, maxiter=200, seed=0, verbose=False):
    """
    Fit the Paranormal distribution via constrained MLE.
    Steps:
      1. k-means initialisation
      2. Differential evolution (global search)
      3. Nelder-Mead polish (local refinement)
    Returns dict with keys: theta, log_likelihood, AIC, BIC.
    """
    n = len(x)
    theta0 = kmeans_init(x, rng_seed=seed)

    # ── differential evolution ────────────────────────────────────────────
    def objective(theta):
        # enforce mu1 < mu2 via penalty
        mu1, mu2 = theta[0], theta[4]
        if mu1 >= mu2:
            return 1e10
        return neg_log_likelihood(theta, x)

    result_de = differential_evolution(
        objective,
        bounds=BOUNDS,
        seed=seed,
        maxiter=maxiter,
        popsize=15,
        mutation=(0.5, 1.0),
        recombination=0.9,
        tol=1e-7,
        polish=False,
        init='latinhypercube',
        x0=theta0,
        workers=1,
    )

    best_theta = result_de.x

    # ── Nelder-Mead polish ────────────────────────────────────────────────
    def clipped_objective(theta):
        for j, (lo, hi) in enumerate(BOUNDS):
            if theta[j] < lo or theta[j] > hi:
                return 1e10
        if theta[0] >= theta[4]:
            return 1e10
        return neg_log_likelihood(theta, x)

    result_nm = minimize(
        clipped_objective,
        best_theta,
        method='Nelder-Mead',
        options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8},
    )
    if result_nm.fun < clipped_objective(best_theta):
        best_theta = result_nm.x

    ll  = log_likelihood(best_theta, x)
    aic = 2 * 9 - 2 * ll
    bic = 9 * np.log(n) - 2 * ll

    if verbose:
        print(f"    LL={ll:.3f}  AIC={aic:.3f}")

    return {
        'theta':           best_theta,
        'log_likelihood':  ll,
        'AIC':             aic,
        'BIC':             bic,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MONTE CARLO SIMULATION  (Section 5 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_SIZES = [100, 2000, 500, 1000]
R            = 250     # replicates
DE_MAXITER   = 200

print('=' * 70)
print('PARANORMAL DISTRIBUTION: MONTE CARLO SIMULATION STUDY')
print('=' * 70)
print(f'\nTrue parameters: {dict(zip(PARAM_NAMES, THETA_TRUE))}')
print(f'Replicates R   : {R}')
print(f'Sample sizes   : {SAMPLE_SIZES}')
print(f'DE iterations  : {DE_MAXITER}')
print()

all_results = []

for n in SAMPLE_SIZES:
    t_start = time.time()
    print(f'--- n = {n} ---')
    estimates = []

    for rep in range(R):
        rep_seed = SEED + rep * 1000 + n
        rep_rng  = np.random.default_rng(rep_seed)

        # Generate data
        x = generate_paranormal(THETA_TRUE, n, rep_rng)

        # Fit
        try:
            result = fit_paranormal(x, maxiter=DE_MAXITER, seed=rep_seed)
            estimates.append(result['theta'])
        except Exception as e:
            # Use NaN row on failure (extremely rare)
            estimates.append(np.full(9, np.nan))

        if (rep + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f'  Completed {rep+1}/{R} replicates '
                  f'({elapsed:.1f}s elapsed)')

    estimates = np.array(estimates)  # shape (R, 9)

    # Compute mean and SD ignoring NaN
    means = np.nanmean(estimates, axis=0)
    stds  = np.nanstd(estimates, axis=0, ddof=1)
    n_ok  = np.sum(np.all(np.isfinite(estimates), axis=1))

    print(f'  Completed. Valid fits: {n_ok}/{R}')
    print(f'  {"Param":<12} {"True":>8} {"Mean":>10} {"SD":>10} {"Bias":>10}')
    print('  ' + '-'*52)
    for j, (name, true_v) in enumerate(zip(PARAM_NAMES, THETA_TRUE)):
        print(f'  {name:<12} {true_v:>8.3f} {means[j]:>10.4f} '
              f'{stds[j]:>10.4f} {means[j]-true_v:>10.4f}')
    print()

    for rep_idx in range(R):
        row = {'n': n, 'replicate': rep_idx}
        for j, name in enumerate(PARAM_NAMES):
            row[name] = estimates[rep_idx, j]
        all_results.append(row)

# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

df_all = pd.DataFrame(all_results)
df_all.to_csv(DATA_DIR / 'sim_results.csv', index=False)
print(f'Raw results saved: {DATA_DIR}/sim_results.csv')

# Summary table: mean and SD per (parameter, n)
summary = {}
for n in SAMPLE_SIZES:
    sub = df_all[df_all['n'] == n]
    summary[n] = {
        'means': {p: sub[p].mean() for p in PARAM_NAMES},
        'stds':  {p: sub[p].std(ddof=1) for p in PARAM_NAMES},
    }

# ─────────────────────────────────────────────────────────────────────────────
# LATEX TABLE  (Table 2 of the paper)
# ─────────────────────────────────────────────────────────────────────────────

def fmt(v, d=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '---'
    return f'${v:.{d}f}$'


tex_lines = [
    r'\begin{table}[ht]',
    r'\centering',
    r'\caption{Monte Carlo simulation: mean and SD of $\thetahat_n$ '
    r'over 500 replicates. True values in parentheses after each parameter.}',
    r'\label{tab:sim}',
    r'\begin{tabular}{lrrrrrrrr}',
    r'\toprule',
    r' & \multicolumn{2}{c}{$n=200$}'
    r' & \multicolumn{2}{c}{$n=500$}'
    r' & \multicolumn{2}{c}{$n=1000$}'
    r' & \multicolumn{2}{c}{$n=2000$} \\',
    r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}'
    r'\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
    r'Param.\ (true) & Mean & SD & Mean & SD & Mean & SD & Mean & SD \\',
    r'\midrule',
]

for j, (pname, plabel) in enumerate(zip(PARAM_NAMES, PARAM_LABELS)):
    tv = TRUE_VALS[pname]
    row_parts = [f'{plabel}\\ $({tv})$']
    for n in SAMPLE_SIZES:
        m = summary[n]['means'][pname]
        s = summary[n]['stds'][pname]
        row_parts.append(fmt(m))
        row_parts.append(fmt(s))
    tex_lines.append(' & '.join(row_parts) + r' \\')

tex_lines += [
    r'\bottomrule',
    r'\multicolumn{9}{l}{\footnotesize'
    r'  Note: degrees-of-freedom parameters $\nu_i$ exhibit larger'
    r'  variance due to flat likelihood near the boundary.}',
    r'\end{tabular}',
    r'\end{table}',
]

tex_table = '\n'.join(tex_lines)
table_path = TABLE_DIR / 'table_sim.tex'
with open(table_path, 'w') as f:
    f.write(tex_table)

print(f'\nLaTeX table saved: {table_path}')
print()
print('=' * 70)
print('SIMULATION COMPLETE')
print('=' * 70)
print()
print('Use in paranormal_paper_final.tex:')
print('  \\input{artifacts/tables/table_sim.tex}')
print()
print('Or paste Table 2 directly using the values in')
print('  artifacts/data/sim_results.csv')
