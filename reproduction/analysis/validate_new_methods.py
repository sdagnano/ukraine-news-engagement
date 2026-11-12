from cell_models import *
from working_covariance import cr2_general
from itertools import combinations
rng = np.random.default_rng(219348)
G = 8
T = 5
n = 480
source = np.repeat(np.arange(G), 60)
date = np.tile(np.arange(T), 96)
cell = codes(pd.MultiIndex.from_arrays([source, date]))
C = cell.max() + 1
X = np.column_stack([rng.normal(size=n), rng.binomial(1, 0.4, n), rng.normal(size=n)])
D = np.column_stack([np.eye(C)[cell], X])
y = rng.normal(size=(n, 3))
a = cell_ols(X, y, cell)
bb = linalg.lstsq(D, y)[0]
checks = {'cell_OLS_coefficient_error': float(abs(a['beta'] - bb[-3:]).max())}
counts = rng.poisson(np.exp(rng.normal(size=C)[cell] + 0.3 * X[:, 0] - 0.25 * X[:, 1] + 0.1 * X[:, 2])).astype(float)
for c in range(C):
    if counts[cell == c].sum() == 0:
        counts[np.flatnonzero(cell == c)[0]] = 1
f = ppml(X, counts, cell)
H = D.T @ (f['mu'][:, None] * D)
Bfull = linalg.inv(H)
u = counts - f['mu']
checks['profile_score_full_dummy'] = float(abs(D.T @ u).max())
val, oi, lr = functional(f, X, counts, cell)
u0 = f['mu'] * np.exp(-f['beta'][1] * X[:, 1])
rr = np.exp(f['beta'][1])
j = C + 1
gb = (rr - 1) * (D.T @ u0) / n
gb[j] -= (rr - 1) * (u0 @ X[:, 1]) / n
gb[j] += rr * u0.mean()
direct = u * (D @ Bfull @ gb)
checks['cell_PPML_delta_gradient_error'] = float(abs(direct - oi[:, 2]).max())
Z = np.column_stack([X, X[:, 1] * (source >= 4)])
AA = np.zeros_like(Z)
AA[:, 1] = 1
AA[:, 3] = source >= 4
ff = ppml(Z, counts, cell)
vv, oo, ll = functional(ff, Z, counts, cell, AA)
DD = np.column_stack([np.eye(C)[cell], Z])
AAA = np.column_stack([np.zeros((n, C)), AA])
alpha = np.zeros(C)
eta = Z @ ff['beta']
den = ag(np.exp(eta), cell)
alpha = np.log(ff['yt'] / den)
theta = np.r_[alpha, ff['beta']]

def delta(th):
    mu0 = np.exp((DD - X[:, 1, None] * AAA) @ th)
    return np.mean(mu0 * np.expm1(AAA @ th))
gd = []
for j0 in range(len(theta)):
    e = np.zeros(len(theta))
    e[j0] = 1e-05
    gd.append((delta(theta + e) - delta(theta - e)) / 2e-05)
BB = linalg.inv(DD.T @ (ff['mu'][:, None] * DD))
di = (counts - ff['mu']) * (DD @ BB @ np.array(gd))
checks['heterogeneous_PPML_finite_gradient_error'] = float(abs(di - oo[:, 2]).max())
Sg = []
sc = []
for sg in range(G):
    ix = source == sg
    Sg.append(a['x'][ix].T @ a['x'][ix])
    sc.append(a['x'][ix].T @ a['res'][ix, 0])
Sg = np.array(Sg)
sc = np.array(sc)
tau = np.exp(np.linspace(-1, 1, G))
v, df = cr2_general(Sg, sc, tau)
BD = linalg.inv(D.T @ D)
M = np.eye(n) - D @ BD @ D.T
Omega = np.diag(tau[source])
R = M @ Omega @ M
inf = []
Q = []
ci = np.eye(D.shape[1])[:, C + 1]
for sg in range(G):
    ix = np.flatnonzero(source == sg)
    ev, U = linalg.eigh(R[np.ix_(ix, ix)])
    keep = ev > 1e-09
    AA = U[:, keep] * np.sqrt(tau[sg] / ev[keep]) @ U[:, keep].T
    inf.append(ci @ BD @ D[ix].T @ AA @ (y - D @ bb)[ix, 0])
    q = np.zeros(n)
    q[ix] = AA @ D[ix] @ BD @ ci
    Q.append(q)
Q = np.array(Q)
GG = Q @ R @ Q.T
dd = np.trace(GG) ** 2 / np.sum(GG * GG)
checks['generalized_CR2_variance_error'] = float(abs(v - np.sum(np.array(inf) ** 2)))
checks['generalized_CR2_df_error'] = float(abs(df - dd))
N = 12
h = np.r_[np.ones(6), np.zeros(6)]
Nuis = np.column_stack([np.ones(N), rng.normal(size=N)])
yy = rng.normal(size=N)
yr = yy - Nuis @ linalg.lstsq(Nuis, yy)[0]
num = h @ yr
sgn = np.sign(num)
gain = sgn * (2 * h - 1) * yr
order = np.argsort(-gain)
cum = np.cumsum(gain[order])
k = int(np.searchsorted(cum, abs(num)) + 1)
found = None
for size in range(N + 1):
    for ix in combinations(range(N), size):
        hh = h.copy()
        ii = np.array(ix, dtype=int)
        hh[ii] = 1 - hh[ii]
        if sgn * (hh @ yr) <= 1e-12:
            found = size
            break
    if found is not None:
        break
checks['adversarial_tipping_matches_exhaustive'] = bool(found == k)
from scale_and_influence import reml
M = np.column_stack([np.ones(n), X])
YY = rng.normal(size=(n, 1)) + rng.normal(size=(G, 1))[source]
br, _, _ = reml(M, YY, source)
Zs = np.eye(G)[source]

def dense_reml(lg, full=False):
    VV = np.eye(n) + np.exp(lg) * (Zs @ Zs.T)
    cho = linalg.cho_factor(VV)
    WM = linalg.cho_solve(cho, M)
    Wy = linalg.cho_solve(cho, YY)
    SS = M.T @ WM
    RR = M.T @ Wy
    bb = linalg.solve(SS, RR)
    res = YY - M @ bb
    rss = float((res.T @ linalg.cho_solve(cho, res))[0, 0])
    v = (n - M.shape[1]) * np.log(rss / (n - M.shape[1])) + 2 * np.log(np.diag(cho[0])).sum() + np.linalg.slogdet(SS)[1]
    return bb if full else v
opt = optimize.minimize_scalar(dense_reml, bounds=(-20, 20), method='bounded', options={'xatol': 1e-08})
bd = dense_reml(opt.x, True)
checks['REML_profile_vs_dense_error'] = float(abs(br[0] - bd[2, 0]))
checks['passed'] = bool(all((v is True if isinstance(v, bool) else v < 1e-07 for v in checks.values())))
(ROOT / 'audit/new_method_validation.json').write_text(json.dumps(checks, indent=2))
print(json.dumps(checks, indent=2))
assert checks['passed']
