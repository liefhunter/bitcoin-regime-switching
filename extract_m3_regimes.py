#!/usr/bin/env python3
"""Extract and compare M=2 vs M=3 regime structures.

Fits a 3-regime MS-VAR and compares conditional variances, ergodic
probabilities, expected durations, and information criteria with the
2-regime baseline model.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# MSVAR Model (supports 2 or 3 regimes)
# ---------------------------------------------------------------------------

class MSVAR:
    """Markov-Switching VAR via EM (Hamilton filter + Kim smoother)."""

    def __init__(self, y: np.ndarray, n_regimes: int = 2, n_lags: int = 2,
                 min_var_frac: float = 0.01, min_persist: float = 0.7,
                 prior_nu: float = 5.0, ridge_lambda: float = 200.0):
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

    def _initialize_params(self, method: str = 'kmeans', seed: int = 42):
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

    def _enforce_pd(self, m: int):
        """Ensure Sigma[m] is symmetric positive definite with variance floor."""
        K = self.K
        for k in range(K):
            self.Sigma[m][k, k] = max(self.Sigma[m][k, k], self._var_floor[k])
        self.Sigma[m] = 0.5 * (self.Sigma[m] + self.Sigma[m].T)
        eigvals = np.linalg.eigvalsh(self.Sigma[m])
        if eigvals.min() < 1e-12:
            self.Sigma[m] += (1e-12 - eigvals.min()) * np.eye(K)

    def _enforce_persist(self, i: int):
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

    def _mvn_logpdf(self, y_t, mu, sigma):
        K = len(mu)
        diff = y_t - mu
        try:
            L = np.linalg.cholesky(sigma)
        except np.linalg.LinAlgError:
            L = np.linalg.cholesky(sigma + 1e-8 * np.eye(K))
        v = np.linalg.solve(L, diff)
        return -0.5 * (K * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(L))) + v @ v)

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
                    ll = -np.inf
                    break
                if not np.isfinite(ll):
                    ll = -np.inf
                    break
                if abs(ll - prev_ll) < tol and it > 5:
                    converged = True
                    break
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

    def _label_regimes(self):
        """Order regimes by BTC variance: highest first."""
        if self.n_regimes == 2:
            if self.Sigma[0][0, 0] < self.Sigma[1][0, 0]:
                for attr in ['mu', 'A', 'Sigma']:
                    setattr(self, attr, getattr(self, attr)[::-1].copy())
                self.P = self.P[::-1, ::-1].copy()
                for attr in ['filtered', 'predicted', 'smoothed']:
                    setattr(self, attr, getattr(self, attr)[:, ::-1].copy())
        else:
            btc_vars = [self.Sigma[m][0, 0] for m in range(self.n_regimes)]
            order = np.argsort(btc_vars)[::-1]
            for attr in ['mu', 'A', 'Sigma']:
                setattr(self, attr, getattr(self, attr)[order].copy())
            self.P = self.P[order][:, order].copy()
            for attr in ['filtered', 'predicted', 'smoothed']:
                setattr(self, attr, getattr(self, attr)[:, order].copy())

    def get_transition_matrix(self):
        return self.P

    def get_expected_durations(self):
        return 1.0 / (1.0 - np.diag(self.P))

    def get_ergodic_probs(self):
        """Compute ergodic (stationary) distribution."""
        eigvals, eigvecs = np.linalg.eig(self.P.T)
        ergodic = np.abs(np.real(eigvecs[:, np.argmin(np.abs(eigvals - 1.0))]))
        ergodic /= ergodic.sum()
        return ergodic

    def get_n_params(self):
        M, K, p = self.n_regimes, self.K, self.n_lags
        return M * (K + K * K * p + K * (K + 1) // 2) + M * (M - 1)


# ---------------------------------------------------------------------------
# Main: M=2 vs M=3 comparison
# ---------------------------------------------------------------------------

def main():
    # Load data
    df = pd.read_csv('msvar_processed.csv', parse_dates=['date'])
    y_full = df[['btc', 'd_bond10', 'd_libor']].values
    dates = df['date'].values

    print('=' * 80)
    print('M=2 vs M=3 Regime Comparison')
    print('=' * 80)
    print()

    # Fit M=2 baseline
    print('Fitting M=2 model...')
    model2 = MSVAR(y_full, n_regimes=2, n_lags=2, ridge_lambda=200.0, prior_nu=5.0)
    model2.fit(n_restarts=20, max_iter=500, verbose=False)
    print(f'M=2: LL = {model2.log_lik:.2f}')
    print()

    # Fit M=3 alternative
    print('Fitting M=3 model...')
    model3 = MSVAR(y_full, n_regimes=3, n_lags=2, ridge_lambda=200.0, prior_nu=5.0,
                   min_persist=0.5)
    model3.fit(n_restarts=20, max_iter=500, verbose=False)
    print(f'M=3: LL = {model3.log_lik:.2f}')
    print()

    # --- Conditional variances ---
    print('=' * 80)
    print('Conditional Variances (diagonal elements of Sigma_m)')
    print('=' * 80)
    print()

    var_labels = ['BTC', 'BOND10', 'LIBOR']

    print('M=2 Model:')
    print('-' * 60)
    for k, v in enumerate(var_labels):
        r1_var = model2.Sigma[0][k, k]
        r2_var = model2.Sigma[1][k, k]
        ratio = r1_var / r2_var
        print(f'  sigma^2_{v:>6s}:  R1={r1_var:.5f}  R2={r2_var:.5f}  (ratio={ratio:.1f}x)')
    print()

    print('M=3 Model:')
    print('-' * 60)
    for k, v in enumerate(var_labels):
        r1_var = model3.Sigma[0][k, k]
        r2_var = model3.Sigma[1][k, k]
        r3_var = model3.Sigma[2][k, k]
        ratio_12 = r1_var / r2_var
        ratio_13 = r1_var / r3_var
        ratio_23 = r2_var / r3_var
        print(f'  sigma^2_{v:>6s}:  R1={r1_var:.5f}  R2={r2_var:.5f}  R3={r3_var:.5f}')
        print(f'                    (R1/R2={ratio_12:.1f}x, R1/R3={ratio_13:.1f}x, R2/R3={ratio_23:.1f}x)')
    print()

    # --- Ergodic probabilities ---
    print('=' * 80)
    print('Ergodic (Stationary) Probabilities')
    print('=' * 80)
    print()

    ergodic2 = model2.get_ergodic_probs()
    ergodic3 = model3.get_ergodic_probs()

    print('M=2 Model:')
    print(f'  P(R1) = {ergodic2[0]:.4f}')
    print(f'  P(R2) = {ergodic2[1]:.4f}')
    print()

    print('M=3 Model:')
    print(f'  P(R1) = {ergodic3[0]:.4f}')
    print(f'  P(R2) = {ergodic3[1]:.4f}')
    print(f'  P(R3) = {ergodic3[2]:.4f}')
    print()

    # --- Expected durations ---
    print('=' * 80)
    print('Expected Durations (5-day intervals)')
    print('=' * 80)
    print()

    dur2 = model2.get_expected_durations()
    dur3 = model3.get_expected_durations()

    print('M=2 Model:')
    print(f'  E[Duration R1] = {dur2[0]:.2f}')
    print(f'  E[Duration R2] = {dur2[1]:.2f}')
    print()

    print('M=3 Model:')
    print(f'  E[Duration R1] = {dur3[0]:.2f}')
    print(f'  E[Duration R2] = {dur3[1]:.2f}')
    print(f'  E[Duration R3] = {dur3[2]:.2f}')
    print()

    # --- Regime coverage by time period ---
    print('=' * 80)
    print('Regime Coverage by Time Period')
    print('=' * 80)
    print()

    dates_plot = dates[model2.n_lags:]

    regime2 = np.argmax(model2.smoothed, axis=1)
    print('M=2 Model:')
    print(f'  Regime 1 (high-vol): {(regime2 == 0).sum()} obs ({100 * (regime2 == 0).mean():.1f}%)')
    print(f'  Regime 2 (low-vol):  {(regime2 == 1).sum()} obs ({100 * (regime2 == 1).mean():.1f}%)')
    print()

    # Identify major R1 episodes (>10 consecutive days)
    print('  Major R1 episodes (>10 consecutive days):')
    in_r1 = (regime2 == 0)
    i = 0
    while i < len(in_r1):
        if in_r1[i]:
            start = i
            while i < len(in_r1) and in_r1[i]:
                i += 1
            end = i - 1
            if end - start >= 10:
                print(f'    {dates_plot[start]} to {dates_plot[end]} ({end - start + 1} days)')
        else:
            i += 1
    print()

    regime3 = np.argmax(model3.smoothed, axis=1)
    print('M=3 Model:')
    print(f'  Regime 1 (highest-vol): {(regime3 == 0).sum()} obs ({100 * (regime3 == 0).mean():.1f}%)')
    print(f'  Regime 2 (mid-vol):     {(regime3 == 1).sum()} obs ({100 * (regime3 == 1).mean():.1f}%)')
    print(f'  Regime 3 (low-vol):     {(regime3 == 2).sum()} obs ({100 * (regime3 == 2).mean():.1f}%)')
    print()

    for m, label in [(0, 'R1 (highest-vol)'), (1, 'R2 (mid-vol)'), (2, 'R3 (low-vol)')]:
        print(f'  Major {label} episodes (>10 consecutive days):')
        in_rm = (regime3 == m)
        i = 0
        found_any = False
        while i < len(in_rm):
            if in_rm[i]:
                start = i
                while i < len(in_rm) and in_rm[i]:
                    i += 1
                end = i - 1
                if end - start >= 10:
                    print(f'    {dates_plot[start]} to {dates_plot[end]} ({end - start + 1} days)')
                    found_any = True
            else:
                i += 1
        if not found_any:
            print('    (no episodes >10 days)')
        print()

    # --- Information criteria comparison ---
    print('=' * 80)
    print('Model Comparison')
    print('=' * 80)
    print()

    n_params2 = model2.get_n_params()
    n_params3 = model3.get_n_params()

    aic2 = -2 * model2.log_lik + 2 * n_params2
    bic2 = -2 * model2.log_lik + np.log(model2.T) * n_params2

    aic3 = -2 * model3.log_lik + 2 * n_params3
    bic3 = -2 * model3.log_lik + np.log(model3.T) * n_params3

    print(f'{"":>15s} {"M=2":>14s} {"M=3":>14s}')
    print('-' * 50)
    print(f'{"LL":>15s} {model2.log_lik:>14.2f} {model3.log_lik:>14.2f}')
    print(f'{"#Params":>15s} {n_params2:>14d} {n_params3:>14d}')
    print(f'{"AIC":>15s} {aic2:>14.2f} {aic3:>14.2f}')
    print(f'{"BIC":>15s} {bic2:>14.2f} {bic3:>14.2f}')
    print()

    if aic3 < aic2:
        print('AIC prefers M=3')
    else:
        print('AIC prefers M=2')

    if bic3 < bic2:
        print('BIC prefers M=3')
    else:
        print('BIC prefers M=2')
    print()

    # --- Transition matrices ---
    print('=' * 80)
    print('Transition Matrices')
    print('=' * 80)
    print()

    P2 = model2.get_transition_matrix()
    P3 = model3.get_transition_matrix()

    print('M=2 Model:')
    print('           To R1    To R2')
    print(f'From R1   {P2[0, 0]:.4f}  {P2[0, 1]:.4f}')
    print(f'From R2   {P2[1, 0]:.4f}  {P2[1, 1]:.4f}')
    print()

    print('M=3 Model:')
    print('           To R1    To R2    To R3')
    print(f'From R1   {P3[0, 0]:.4f}  {P3[0, 1]:.4f}  {P3[0, 2]:.4f}')
    print(f'From R2   {P3[1, 0]:.4f}  {P3[1, 1]:.4f}  {P3[1, 2]:.4f}')
    print(f'From R3   {P3[2, 0]:.4f}  {P3[2, 1]:.4f}  {P3[2, 2]:.4f}')
    print()

    # --- Interpretation ---
    print('=' * 80)
    print('Interpretation')
    print('=' * 80)
    print()
    print('M=2 Model identifies two distinct regimes:')
    print('  - R1 (high-vol): captures crisis periods (COVID-19, FTX, etc.)')
    print('  - R2 (low-vol): normal market conditions')
    print()
    print('M=3 Model adds granularity:')
    print('  - R1 (highest-vol): acute crisis episodes')
    print('  - R2 (mid-vol): elevated uncertainty or transition periods')
    print('  - R3 (low-vol): stable market conditions')
    print()

    if aic3 < aic2 and bic3 > bic2:
        print('AIC prefers M=3 but BIC penalises the extra parameters.')
        print('M=2 is retained for parsimony (Occam\'s razor).')


if __name__ == '__main__':
    main()
