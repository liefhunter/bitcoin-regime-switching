#!/usr/bin/env python3
"""
Parametric Bootstrap for MS-VAR Impulse Response Functions.

Generates 90% confidence intervals via 500 bootstrap replications
for a two-regime Markov-Switching VAR(2) model estimated by EM.
"""

import time

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans


# ---------------------------------------------------------------------------
# MSVAR Model
# ---------------------------------------------------------------------------

class MSVAR:
    """Two-regime Markov-Switching VAR via EM (Hamilton filter + Kim smoother)."""

    def __init__(self, y, n_regimes=2, n_lags=2, min_var_frac=0.01, min_persist=0.7,
                 prior_nu=5.0, ridge_lambda=200.0):
        self.y_raw = y
        self.n_regimes = n_regimes
        self.n_lags = n_lags
        self.K = y.shape[1]
        self.T_full = y.shape[0]
        self._min_var_frac = min_var_frac
        self._min_persist = min_persist
        self._prior_nu = prior_nu
        self._ridge_lambda = ridge_lambda
        self._build_design()

    # -- Design matrix construction ------------------------------------------

    def _build_design(self):
        T, K, p = self.T_full, self.K, self.n_lags
        self.T = T - p
        self.y = self.y_raw[p:]
        X_list = [self.y_raw[p - lag:T - lag] for lag in range(1, p + 1)]
        self.X = np.hstack(X_list)
        uncond_var = np.var(self.y, axis=0)
        self._var_floor = self._min_var_frac * uncond_var
        self._Sigma_prior = np.cov(self.y, rowvar=False)
        X_aug = np.hstack([np.ones((self.T, 1)), self.X])
        self._beta_pooled = np.linalg.lstsq(X_aug, self.y, rcond=None)[0]

    # -- Parameter initialisation --------------------------------------------

    def _initialize_params(self, method='kmeans', seed=42):
        M, K, T = self.n_regimes, self.K, self.T
        Kp = K * self.n_lags
        rng = np.random.RandomState(seed)
        if method == 'kmeans':
            abs_btc = np.abs(self.y[:, 0]).reshape(-1, 1)
            labels = KMeans(n_clusters=M, n_init=10, random_state=seed).fit_predict(abs_btc)
        elif method == 'kmeans_all':
            labels = KMeans(n_clusters=M, n_init=10, random_state=seed).fit_predict(np.abs(self.y))
        elif method == 'rolling_vol':
            btc = self.y[:, 0]
            rv = pd.Series(btc).rolling(50, min_periods=5).var().bfill().values
            labels = (rv > np.median(rv)).astype(int)
        elif method == 'chronological':
            labels = np.zeros(T, dtype=int)
            seg = T // M
            for m in range(1, M):
                labels[m * seg:] = m
        else:
            labels = rng.randint(0, M, size=T)

        self.mu = np.zeros((M, K))
        self.A = np.zeros((M, K, Kp))
        self.Sigma = np.zeros((M, K, K))
        for m in range(M):
            idx = labels == m
            if idx.sum() < Kp + 5:
                idx = np.ones(T, dtype=bool)
            y_m, X_m = self.y[idx], self.X[idx]
            X_aug_m = np.hstack([np.ones((idx.sum(), 1)), X_m])
            beta = np.linalg.lstsq(X_aug_m, y_m, rcond=None)[0]
            self.mu[m] = beta[0]
            self.A[m] = beta[1:].T
            e = y_m - X_aug_m @ beta
            self.Sigma[m] = e.T @ e / max(e.shape[0] - 1, 1)
            self._enforce_pd(m)
        self.P = np.full((M, M), 0.05 / (M - 1))
        np.fill_diagonal(self.P, 0.95)

    # -- Numerical safeguards ------------------------------------------------

    def _enforce_pd(self, m):
        """Ensure Sigma[m] is symmetric positive definite with variance floor."""
        K = self.K
        for k in range(K):
            self.Sigma[m][k, k] = max(self.Sigma[m][k, k], self._var_floor[k])
        self.Sigma[m] = 0.5 * (self.Sigma[m] + self.Sigma[m].T)
        eigvals = np.linalg.eigvalsh(self.Sigma[m])
        if eigvals.min() < 1e-12:
            self.Sigma[m] += (1e-12 - eigvals.min()) * np.eye(K)

    def _enforce_persist(self, i):
        """Ensure P[i,i] >= min_persist by scaling off-diagonal elements."""
        M = self.n_regimes
        if self.P[i, i] >= self._min_persist:
            return
        off_diag_sum = 1.0 - self.P[i, i]
        new_off_sum = 1.0 - self._min_persist
        if off_diag_sum > 1e-300:
            scale = new_off_sum / off_diag_sum
            for j in range(M):
                if j != i:
                    self.P[i, j] *= scale
        self.P[i, i] = self._min_persist

    # -- Likelihood evaluation -----------------------------------------------

    def _mvn_logpdf(self, y_t, mu, sigma):
        K = len(mu)
        diff = y_t - mu
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(sigma + 1e-8 * np.eye(K))
        v = np.linalg.solve(L, diff)
        return -0.5 * (K * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + v @ v)

    # -- Hamilton filter & Kim smoother --------------------------------------

    def _hamilton_filter(self):
        T, M, K = self.T, self.n_regimes, self.K
        cond_mean = np.zeros((T, M, K))
        for m in range(M):
            cond_mean[:, m, :] = self.mu[m] + self.X @ self.A[m].T
        log_f = np.zeros((T, M))
        for t in range(T):
            for m in range(M):
                log_f[t, m] = self._mvn_logpdf(self.y[t], cond_mean[t, m], self.Sigma[m])

        # Ergodic distribution as initial state
        eigvals, eigvecs = np.linalg.eig(self.P.T)
        ergodic = np.abs(np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))]))
        ergodic /= ergodic.sum()

        filtered = np.zeros((T, M))
        predicted = np.zeros((T, M))
        ll = 0.0
        for t in range(T):
            pred = ergodic if t == 0 else filtered[t - 1] @ self.P
            pred = np.maximum(pred, 1e-300)
            predicted[t] = pred
            log_joint = log_f[t] + np.log(pred)
            log_marg = logsumexp(log_joint)
            ll += log_marg
            filtered[t] = np.exp(log_joint - log_marg)
            filtered[t] = np.maximum(filtered[t], 1e-300)
            filtered[t] /= filtered[t].sum()
        return filtered, predicted, ll

    def _kim_smoother(self, filtered, predicted):
        T, M = filtered.shape
        smoothed = np.zeros((T, M))
        smoothed[-1] = filtered[-1]
        for t in range(T - 2, -1, -1):
            for i in range(M):
                s = 0.0
                for j in range(M):
                    if predicted[t + 1, j] > 1e-300:
                        s += self.P[i, j] * smoothed[t + 1, j] / predicted[t + 1, j]
                smoothed[t, i] = filtered[t, i] * s
            smoothed[t] = np.maximum(smoothed[t], 1e-300)
            smoothed[t] /= smoothed[t].sum()
        return smoothed

    def _joint_probs(self, filtered, predicted, smoothed):
        T, M = filtered.shape
        joint = np.zeros((T - 1, M, M))
        for t in range(T - 1):
            for i in range(M):
                for j in range(M):
                    if predicted[t + 1, j] > 1e-300:
                        joint[t, i, j] = (filtered[t, i] * self.P[i, j]
                                          * smoothed[t + 1, j] / predicted[t + 1, j])
            s = joint[t].sum()
            if s > 1e-300:
                joint[t] /= s
        return joint

    # -- EM step -------------------------------------------------------------

    def _em_step(self):
        M, K, T = self.n_regimes, self.K, self.T
        Kp = K * self.n_lags
        n_aug = 1 + Kp
        filtered, predicted, ll = self._hamilton_filter()
        smoothed = self._kim_smoother(filtered, predicted)
        joint = self._joint_probs(filtered, predicted, smoothed)

        # M-step: transition matrix
        for i in range(M):
            num = joint[:, i, :].sum(axis=0)
            d = num.sum()
            self.P[i] = num / d if d > 1e-300 else np.ones(M) / M
            self.P[i] = np.maximum(self.P[i], 1e-6)
            self._enforce_persist(i)

        # M-step: regression coefficients (ridge) and covariance (IW prior)
        lam = self._ridge_lambda
        nu = self._prior_nu
        X_aug = np.hstack([np.ones((T, 1)), self.X])
        for m in range(M):
            w = smoothed[:, m]
            w_sum = w.sum()
            if w_sum < 1.0:
                continue
            Xw = X_aug * w[:, None]
            XwX = Xw.T @ X_aug + lam * np.eye(n_aug)
            Xwy = Xw.T @ self.y + lam * self._beta_pooled
            beta = np.linalg.solve(XwX, Xwy)
            self.mu[m] = beta[0]
            self.A[m] = beta[1:].T
            resid = self.y - (self.mu[m] + self.X @ self.A[m].T)
            wRR = (resid * w[:, None]).T @ resid
            self.Sigma[m] = (wRR + nu * self._Sigma_prior) / (w_sum + nu)
            self.Sigma[m] = 0.5 * (self.Sigma[m] + self.Sigma[m].T)
            self._enforce_pd(m)
        return ll

    # -- Fit with multiple restarts ------------------------------------------

    def fit(self, max_iter=500, tol=1e-6, n_restarts=20, verbose=True):
        best_ll = -np.inf
        best_params = None
        methods = (['kmeans', 'kmeans_all', 'rolling_vol', 'chronological']
                   + ['random'] * max(0, n_restarts - 4))
        for r in range(n_restarts):
            method = methods[r] if r < len(methods) else 'random'
            self._initialize_params(method=method, seed=r * 137 + 42)
            prev_ll, converged = -np.inf, False
            for it in range(max_iter):
                try:
                    ll = self._em_step()
                except (np.linalg.LinAlgError, ValueError):
                    ll = -np.inf; break
                if not np.isfinite(ll):
                    ll = -np.inf; break
                if abs(ll - prev_ll) < tol and it > 5:
                    converged = True; break
                prev_ll = ll
            if verbose and (r < 5 or ll > best_ll):
                st = 'converged' if converged else f'iter {it}'
                print(f'  Restart {r:2d} ({method:>14s}): LL={ll:12.2f} ({st})')
            if ll > best_ll:
                best_ll = ll
                best_params = {k: getattr(self, k).copy() for k in ['mu', 'A', 'Sigma', 'P']}
        for k, v in best_params.items():
            setattr(self, k, v)
        self.log_lik = best_ll
        self.filtered, self.predicted, _ = self._hamilton_filter()
        self.smoothed = self._kim_smoother(self.filtered, self.predicted)
        self._label_regimes()
        if verbose:
            print(f'\nBest LL: {self.log_lik:.2f}')

    # -- Post-estimation helpers ---------------------------------------------

    def _label_regimes(self):
        """Order regimes so that Regime 0 = high-vol, Regime 1 = low-vol."""
        if self.Sigma[0][0, 0] < self.Sigma[1][0, 0]:
            for attr in ['mu', 'A', 'Sigma']:
                setattr(self, attr, getattr(self, attr)[::-1].copy())
            self.P = self.P[::-1, ::-1].copy()
            for attr in ['filtered', 'predicted', 'smoothed']:
                setattr(self, attr, getattr(self, attr)[:, ::-1].copy())

    def get_transition_matrix(self): return self.P
    def get_expected_durations(self): return 1.0 / (1.0 - np.diag(self.P))

    def get_n_params(self):
        M, K, p = self.n_regimes, self.K, self.n_lags
        return M * (K + K * K * p + K * (K + 1) // 2) + M * (M - 1)

    def compute_irf(self, regime, horizon=10, shock_var=0, shock_size=1.0):
        """Generalised IRF (Pesaran & Shin 1998) for a given regime."""
        K, p = self.K, self.n_lags
        Sigma, A = self.Sigma[regime], self.A[regime]

        # Companion form
        A_comp = np.zeros((K * p, K * p))
        for i in range(p):
            A_comp[:K, i * K:(i + 1) * K] = A[:, i * K:(i + 1) * K]
        if p > 1:
            A_comp[K:, :K * (p - 1)] = np.eye(K * (p - 1))

        # Initial shock scaled by conditional covariance
        e_j = np.zeros(K); e_j[shock_var] = 1.0
        delta = (shock_size / np.sqrt(Sigma[shock_var, shock_var])) * Sigma @ e_j

        # Propagate through VAR dynamics
        irf = np.zeros((horizon + 1, K))
        irf[0] = delta
        state = np.zeros(K * p); state[:K] = delta
        for h in range(1, horizon + 1):
            state = A_comp @ state; irf[h] = state[:K]
        return irf


# ---------------------------------------------------------------------------
# Simulation and bootstrap functions
# ---------------------------------------------------------------------------

def simulate_msvar(model, T, seed=None):
    """Simulate regime path and observations from a fitted MSVAR.

    Returns:
        y_sim: (T, K) simulated observations
        regimes: (T,) regime indicators
    """
    rng = np.random.RandomState(seed)
    M, K = model.n_regimes, model.K

    # Ergodic initial distribution
    eigvals, eigvecs = np.linalg.eig(model.P.T)
    ergodic = np.abs(np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))]))
    ergodic /= ergodic.sum()

    # Simulate regime path
    regimes = np.zeros(T, dtype=int)
    regimes[0] = rng.choice(M, p=ergodic)
    for t in range(1, T):
        regimes[t] = rng.choice(M, p=model.P[regimes[t - 1]])

    # Simulate observations with burn-in from lag structure
    p = model.n_lags
    y_sim = np.zeros((T + p, K))
    for t in range(p):
        regime = regimes[0] if t < len(regimes) else regimes[-1]
        y_sim[t] = rng.multivariate_normal(model.mu[regime], model.Sigma[regime])
    for t in range(T):
        regime = regimes[t]
        X_t = np.hstack([y_sim[p + t - lag] for lag in range(1, p + 1)])
        mean_t = model.mu[regime] + X_t @ model.A[regime].T
        y_sim[p + t] = rng.multivariate_normal(mean_t, model.Sigma[regime])

    return y_sim[p:], regimes


def bootstrap_irf(model, n_boot=500, horizon=10, max_iter=100, verbose=True):
    """Parametric bootstrap for IRF confidence intervals.

    Args:
        model: Fitted MSVAR instance
        n_boot: Number of bootstrap replications
        horizon: IRF horizon
        max_iter: Max EM iterations per bootstrap fit
        verbose: Print progress

    Returns:
        irf_boot: (n_boot, 2, 9, horizon+1) array
                  [b, regime, panel, h] where panel indexes 3x3 shock-response grid
    """
    K, T = model.K, model.T
    n_panels = K * K
    irf_boot = np.zeros((n_boot, 2, n_panels, horizon + 1))

    start_time = time.time()

    for b in range(n_boot):
        # Simulate data from fitted model
        y_boot, _ = simulate_msvar(model, T, seed=b * 42)

        # Re-fit MSVAR (fewer restarts for speed)
        try:
            model_boot = MSVAR(
                y_boot,
                n_regimes=2,
                n_lags=2,
                min_persist=0.7,
                prior_nu=5.0,
                ridge_lambda=200.0
            )
            model_boot.fit(max_iter=max_iter, n_restarts=5, verbose=False)

            # Compute all 9 IRF panels for both regimes
            for regime in range(2):
                panel_idx = 0
                for shock_var in range(K):
                    for response_var in range(K):
                        irf_full = model_boot.compute_irf(
                            regime=regime,
                            horizon=horizon,
                            shock_var=shock_var,
                            shock_size=1.0
                        )
                        irf_boot[b, regime, panel_idx] = irf_full[:, response_var]
                        panel_idx += 1

        except Exception as e:
            if verbose and b % 50 == 0:
                print(f"  Bootstrap {b:3d}: Failed ({str(e)[:50]})")
            continue  # Leave zeros for failed replications

        # Progress report
        if verbose and (b + 1) % 50 == 0:
            elapsed = time.time() - start_time
            rate = (b + 1) / elapsed
            remaining = (n_boot - b - 1) / rate
            print(f"  Bootstrap {b+1:3d}/{n_boot}: {elapsed/60:.1f}m elapsed, "
                  f"~{remaining/60:.1f}m remaining")

    return irf_boot


def compute_point_estimates(model, horizon=10):
    """Compute point estimate IRFs for all 9 shock-response panels."""
    K = model.K
    n_panels = K * K
    irf_point = np.zeros((2, n_panels, horizon + 1))

    for regime in range(2):
        panel_idx = 0
        for shock_var in range(K):
            for response_var in range(K):
                irf_full = model.compute_irf(
                    regime=regime,
                    horizon=horizon,
                    shock_var=shock_var,
                    shock_size=1.0
                )
                irf_point[regime, panel_idx] = irf_full[:, response_var]
                panel_idx += 1

    return irf_point


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_irf_with_ci(irf_point, irf_boot, var_names, save_path):
    """Generate 3x3 grid of IRF plots with 90% bootstrap confidence intervals.

    Args:
        irf_point: (2, 9, horizon+1) point estimates
        irf_boot: (n_boot, 2, 9, horizon+1) bootstrap draws
        var_names: List of variable names
        save_path: File path to save figure
    """
    import matplotlib.pyplot as plt

    K = len(var_names)
    horizon = irf_point.shape[-1] - 1

    # 5th and 95th percentiles for 90% CI
    irf_lower = np.percentile(irf_boot, 5, axis=0)
    irf_upper = np.percentile(irf_boot, 95, axis=0)

    fig, axes = plt.subplots(K, K, figsize=(12, 10), sharex=True)
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    h_grid = np.arange(horizon + 1)

    for shock_var in range(K):
        for response_var in range(K):
            panel_idx = shock_var * K + response_var
            ax = axes[response_var, shock_var]

            # Regime 1 (High Vol)
            ax.plot(h_grid, irf_point[0, panel_idx], 'r-', linewidth=1.5,
                    label='R1 (High Vol)', zorder=3)
            ax.fill_between(h_grid,
                            irf_lower[0, panel_idx],
                            irf_upper[0, panel_idx],
                            color='red', alpha=0.15, zorder=1)

            # Regime 2 (Low Vol)
            ax.plot(h_grid, irf_point[1, panel_idx], 'b--', linewidth=1.5,
                    label='R2 (Low Vol)', zorder=3)
            ax.fill_between(h_grid,
                            irf_lower[1, panel_idx],
                            irf_upper[1, panel_idx],
                            color='blue', alpha=0.15, zorder=1)

            ax.axhline(0, color='black', linewidth=0.5, linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=9)

            if response_var == K - 1:
                ax.set_xlabel(f'{var_names[shock_var]} shock', fontsize=10)
            if shock_var == 0:
                ax.set_ylabel(f'{var_names[response_var]} response', fontsize=10)
            if shock_var == K - 1 and response_var == 0:
                ax.legend(fontsize=8, loc='best', framealpha=0.9)

    fig.text(0.5, 0.02, 'Horizon (5-day intervals)', ha='center', fontsize=11)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.close(fig)


def print_significance_table(irf_point, irf_boot, var_names):
    """Print table showing which IRF panels have CIs excluding zero."""
    K = len(var_names)
    irf_lower = np.percentile(irf_boot, 5, axis=0)
    irf_upper = np.percentile(irf_boot, 95, axis=0)

    print("\n" + "=" * 90)
    print("SIGNIFICANCE TABLE: 90% Confidence Intervals Excluding Zero")
    print("=" * 90)
    print(f"{'Panel':<25} {'Regime':<10} {'h=0':<15} {'h=5':<15}")
    print("-" * 90)

    for shock_var in range(K):
        for response_var in range(K):
            panel_idx = shock_var * K + response_var
            panel_name = f"{var_names[shock_var]} -> {var_names[response_var]}"

            for regime in range(2):
                regime_name = f"R{regime + 1}"

                h0_excludes = ((irf_lower[regime, panel_idx, 0] > 0) or
                               (irf_upper[regime, panel_idx, 0] < 0))
                h0_str = "* Significant" if h0_excludes else "  Not signif."

                h5_excludes = ((irf_lower[regime, panel_idx, 5] > 0) or
                               (irf_upper[regime, panel_idx, 5] < 0))
                h5_str = "* Significant" if h5_excludes else "  Not signif."

                if regime == 0:
                    print(f"{panel_name:<25} {regime_name:<10} {h0_str:<15} {h5_str:<15}")
                else:
                    print(f"{'':<25} {regime_name:<10} {h0_str:<15} {h5_str:<15}")
            print("-" * 90)

    print("=" * 90)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("PARAMETRIC BOOTSTRAP FOR MSVAR IMPULSE RESPONSE FUNCTIONS")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading data...")
    data = pd.read_csv('msvar_processed.csv')
    y = data[['btc', 'd_bond10', 'd_libor']].values
    var_names = ['BTC', 'Bond', 'LIBOR']
    print(f"  Data shape: {y.shape}")

    # Fit original model
    print("\n[2/6] Fitting original MSVAR (M=2, p=2)...")
    model = MSVAR(
        y,
        n_regimes=2,
        n_lags=2,
        min_persist=0.7,
        prior_nu=5.0,
        ridge_lambda=200.0
    )
    model.fit(max_iter=500, n_restarts=20, verbose=True)

    # Compute point estimates
    print("\n[3/6] Computing point estimate IRFs...")
    horizon = 10
    irf_point = compute_point_estimates(model, horizon=horizon)
    print(f"  Point estimates shape: {irf_point.shape}")

    # Bootstrap (500 replications, ~20-30 minutes)
    print("\n[4/6] Running parametric bootstrap (500 replications)...")
    print("  This will take approximately 20-30 minutes...")
    n_boot = 500
    irf_boot = bootstrap_irf(
        model,
        n_boot=n_boot,
        horizon=horizon,
        max_iter=100,
        verbose=True
    )
    print(f"\n  Bootstrap complete. Shape: {irf_boot.shape}")

    # Save results
    print("\n[5/6] Saving bootstrap results...")
    np.savez_compressed(
        'bootstrap_irf_results.npz',
        irf_point=irf_point,
        irf_boot=irf_boot,
        var_names=var_names,
        horizon=horizon,
        n_boot=n_boot
    )
    print("  Saved to: bootstrap_irf_results.npz")

    # Generate figure
    print("\n[6/6] Generating IRF figure with bootstrap CIs...")
    plot_irf_with_ci(irf_point, irf_boot, var_names,
                     save_path='fig5_2_regime_irf_bootstrap.png')

    # Print significance table
    print_significance_table(irf_point, irf_boot, var_names)

    print("\n" + "=" * 80)
    print("BOOTSTRAP ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
