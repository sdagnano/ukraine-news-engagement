from pathlib import Path
import os, sys, json, hashlib
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parents[1]
for name in ['pydeps', 'readr_deps']:
    p = BASE / 'work' / name
    if p.exists():
        sys.path.insert(0, str(p))
import numpy as np, pandas as pd
from scipy import stats, linalg, optimize

def codes(a):
    return pd.factorize(a, sort=True)[0].astype(int)

def aggregate(a, g):
    a = np.asarray(a)
    one = a.ndim == 1
    if one:
        a = a[:, None]
    v = np.column_stack([np.bincount(g, weights=a[:, j], minlength=g.max() + 1) for j in range(a.shape[1])])
    return v[:, 0] if one else v

def demean(a, g):
    return a - aggregate(a, g)[g] / np.bincount(g)[g, None]

def absorb(a, groups, tol=1e-10):
    v = a.copy()
    scale = np.maximum(1, np.max(np.abs(a), axis=0))
    for it in range(10000):
        old = v.copy()
        for g in groups:
            v = demean(v, g)
        if np.max(np.abs(v - old) / scale) < tol:
            return (v, it + 1)
    raise RuntimeError('Absorption failed')

def cr1(g, n, rank):
    G = int(g.max() + 1)
    return G / (G - 1) * (n - 1) / (n - rank)

def covariance(iobs, g, rank):
    s = aggregate(iobs, g)
    return s.T @ s * cr1(g, len(g), rank)

def multi_cov(iobs, g, h, rank):
    cross = codes(pd.MultiIndex.from_arrays([g, h]))
    a = covariance(iobs, g, rank)
    b = covariance(iobs, h, rank)
    c = covariance(iobs, cross, rank)
    return {'source': a, 'time': b, 'two_way': a + b - c}

def ols(X, Y, groups, rank_fe, selected):
    xy, it = absorb(np.column_stack([X, Y]), groups)
    p = X.shape[1]
    x = xy[:, :p]
    y = xy[:, p:]
    xx = x.T @ x
    if np.linalg.matrix_rank(xx) != p:
        raise RuntimeError('OLS rank deficiency')
    bread = np.linalg.inv(xx)
    b = bread @ (x.T @ y)
    res = y - x @ b
    obs = {j: (x @ bread[:, j])[:, None] * res for j in selected}
    return dict(beta=b, obs=obs, rank=rank_fe + p, iterations=it, condition=float(np.linalg.cond(xx)), x=x, res=res)

def cancellation(b):
    gross = float(np.abs(b).sum())
    return float(np.clip(1 - abs(b.sum()) / gross, 0, 1)) if gross else 0.0

def summary(b, V, df, family):
    if np.any(np.diag(V) < 0):
        return {'valid_diagonal': False, 'C': cancellation(b)}
    se = np.sqrt(np.diag(V))
    crit = stats.t.ppf(1 - 0.05 / (2 * family), df)
    lo = b - crit * se
    hi = b + crit * se
    P = np.maximum(lo, 0).sum()
    N = np.maximum(-hi, 0).sum()
    return dict(valid_diagonal=True, C=cancellation(b), positive_certified=bool(np.any(lo > 0)), negative_certified=bool(np.any(hi < 0)), both_signs_certified=bool(np.any(lo > 0) and np.any(hi < 0)), all_positive_certified=bool(np.all(lo > 0)), all_negative_certified=bool(np.all(hi < 0)), absolute_opposing_lower=2 * float(min(P, N)), cov_min_eigenvalue=float(np.linalg.eigvalsh(V).min()), total=float(b.sum()), total_se=float(np.sqrt(max(0, V.sum()))), family=int(family), df=int(df))

def partitions(items):
    if not items:
        yield []
        return
    for p in partitions(items[1:]):
        yield ([[items[0]]] + [g[:] for g in p])
        for k in range(len(p)):
            q = [g[:] for g in p]
            q[k] = [items[0]] + q[k]
            yield q

def partition_diagnostics(b, V, df, family, channels):
    crit = stats.t.ppf(1 - 0.05 / (2 * family), df)
    se = np.sqrt(np.diag(V))
    lo = b - crit * se
    hi = b + crit * se
    rows = []
    for p in partitions(list(range(len(b)))):
        bp = np.array([b[g].sum() for g in p])
        L = np.array([lo[g].sum() for g in p])
        U = np.array([hi[g].sum() for g in p])
        rows.append(dict(groups=len(p), partition=' | '.join(('+'.join((channels[k] for k in g)) for g in p)), C=cancellation(bp), both_signs_certified_rectangle=bool(np.any(L > 0) and np.any(U < 0)), total=float(bp.sum())))
    return rows

def weight_frontier(b, V, df, first, last):
    ix = [first, last]
    vv = V[np.ix_(ix, ix)]
    bb = b[ix]
    if np.linalg.eigvalsh(vv).min() < -1e-08:
        return []
    critical = np.sqrt(2 * stats.f.ppf(0.95, 2, df))
    rows = []
    for w in np.linspace(0, 10, 201):
        a = np.array([1, w])
        est = float(a @ bb)
        se = float(np.sqrt(max(0, a @ vv @ a)))
        lo = est - critical * se
        hi = est + critical * se
        rows.append(dict(weight=float(w), estimate=est, simultaneous_low=lo, simultaneous_high=hi, sign_simultaneous='positive' if lo > 0 else 'negative' if hi < 0 else 'unresolved'))
    return rows

def ppml(X, y, g, selected):
    n, p = X.shape
    G = g.max() + 1
    tot = aggregate(y, g)
    normalizer = float(y.sum())
    ng = np.bincount(g)
    if normalizer <= 0:
        raise RuntimeError('Zero response')

    def evaluate(beta, grad=True):
        eta = X @ beta
        maximum = np.full(G, -np.inf)
        np.maximum.at(maximum, g, eta)
        e = np.exp(eta - maximum[g])
        den = aggregate(e, g)
        mu = tot[g] * e / den[g]
        logden = maximum + np.log(den)
        value = (tot @ logden - y @ eta) / normalizer
        return (value, X.T @ (mu - y) / normalizer, mu)
    result = optimize.minimize(lambda b: evaluate(b)[:2], np.zeros(p), jac=True, method='L-BFGS-B', options={'ftol': 1e-14, 'gtol': 2e-09, 'maxiter': 1000, 'maxls': 40, 'maxcor': 20})
    beta = result.x.copy()
    polished = 0
    value, grad, mu = evaluate(beta)
    for iteration in range(4):
        if np.max(np.abs(grad)) < 1e-10:
            break
        wx = np.column_stack([aggregate(mu * X[:, j], g) for j in range(p)])
        wx = np.divide(wx, tot[:, None], out=np.zeros_like(wx), where=tot[:, None] > 0)
        xw = X - wx[g]
        H = xw.T @ (mu[:, None] * xw)
        step = np.linalg.solve(H, grad * normalizer)
        val2, grad2, mu2 = evaluate(beta - step)
        if val2 > value + 1e-10:
            break
        beta -= step
        value, grad, mu = (val2, grad2, mu2)
        polished += 1
    meanx = np.zeros((G, p))
    for j in range(p):
        meanx[:, j] = np.divide(aggregate(mu * X[:, j], g), tot, out=np.zeros(G), where=tot > 0)
    xw = X - meanx[g]
    H = xw.T @ (mu[:, None] * xw)
    if np.linalg.matrix_rank(H) != p:
        raise RuntimeError('PPML Hessian rank deficient')
    bread = np.linalg.inv(H)
    res = y - mu
    infl = {j: aggregate(xw @ bread[:, j] * res, g) * np.sqrt(cr1(g, n, G + p)) for j in selected}
    return dict(beta=beta, mu=mu, influence=infl, audit=dict(success=bool(result.success), message=str(result.message), iterations=int(result.nit), Newton_polish_steps=polished, gradient_max=float(np.max(np.abs(grad))), source_sum_max_error=float(np.max(np.abs(aggregate(mu, g) - tot))), zero_total_sources=int((tot == 0).sum()), Hessian_condition=float(np.linalg.cond(H)), deviance=2 * float(np.sum(np.where(y > 0, y * np.log(np.maximum(y, 1e-300) / np.maximum(mu, 1e-300)), 0) - (y - mu)))))

def ukrainian_design(d):
    z = pd.DataFrame(index=d.index)
    z['solidarity'] = (d.solidarity_pred != 'Other').astype(float)
    z['hostility'] = (d.hostility_pred != 'Other').astype(float)
    z['ingroup_mention'] = d.ingroup.gt(0).astype(float)
    z['outgroup_mention'] = d.outgroup.gt(0).astype(float)
    for c in ['moral_emotional', 'positive', 'negative']:
        z[c + '_z'] = (d[c] - d[c].mean()) / d[c].std(ddof=1)
    for c in ['has_media', 'has_URL', 'is_retweet']:
        if c in d:
            a = d[c].astype(str).str.lower().map({'true': 1.0, 'false': 0.0, '1': 1.0, '0': 0.0})
            if a.nunique() > 1:
                z[c] = a
    return z
