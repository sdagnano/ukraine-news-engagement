from cell_models import *
from scipy import special
import time
SEED = 918347

def wilson(x, n):
    z = stats.norm.ppf(0.975)
    p = x / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)

def independent_columns(X, cell):
    ng = np.bincount(cell)
    x = X - ag(X, cell)[cell] / ng[cell, None]
    chosen = []
    rank = 0
    for j in [1, 0] + list(range(2, X.shape[1])):
        nr = np.linalg.matrix_rank(x[:, chosen + [j]])
        if nr > rank:
            chosen.append(j)
            rank = nr
    return chosen

def pools(d, X, Y, s, t, c):
    names = CHANNELS['facebook']
    cached = ROOT / 'results/calibration_source_slopes.csv'
    existing = pd.read_csv(cached if cached.exists() and '--reuse-pools' in sys.argv else ROOT / 'results/ppml_source_slopes.csv')
    rows = existing.to_dict('records')
    for k, ch in enumerate(names):
        if ch in existing.channel.unique():
            continue
        for sg in range(s.max() + 1):
            ix = s == sg
            h = X[ix, 1]
            if ix.sum() < 50 or min((h == 0).sum(), (h == 1).sum()) < 5:
                continue
            cc = codes(c[ix])
            keep = independent_columns(X[ix], cc)
            try:
                f = ppml(X[ix][:, keep], Y[ix, k], cc)
                rows.append(dict(source=sg, channel=ch, finite=True, beta=f['beta'][keep.index(1)]))
            except (RuntimeError, linalg.LinAlgError) as e:
                rows.append(dict(source=sg, channel=ch, finite=False, reason=str(e)))
    frame = pd.DataFrame(rows)
    frame.to_csv(ROOT / 'results/calibration_source_slopes.csv', index=False)
    b = frame[frame.finite].pivot(index='source', columns='channel', values='beta').reindex(columns=names).dropna()
    source_pool = b.to_numpy(copy=True)
    source_pool -= special.logsumexp(source_pool, axis=0) - np.log(len(source_pool))
    temp = []
    for period in np.unique(t // 14):
        ix = t // 14 == period
        bb = []
        for k in range(len(names)):
            f = ppml(X[ix], Y[ix, k], codes(c[ix]))
            bb.append(f['beta'][1])
        temp.append(bb)
    time_pool = np.array(temp)
    time_pool -= special.logsumexp(time_pool, axis=0) - np.log(len(time_pool))
    np.savez_compressed(ROOT / 'results/calibration_pools.npz', source=source_pool, time=time_pool, source_ids=b.index.to_numpy())
    return (source_pool, time_pool)

def activation_truth(mu, H, alpha, sp, tp):
    n, K = mu.shape
    out = np.zeros_like(mu)
    ix = H == 1

    def p0(m):
        return np.exp(-m) if alpha == 0 else np.exp(-np.log1p(alpha * m) / alpha)
    for k in range(K):
        a = alpha[k]

        def calc(m):
            return np.exp(-m) if a < 1e-10 else np.exp(-np.log1p(a * m) / a)
        out[~ix, k] = 1 - calc(mu[~ix, k])
        total = np.zeros(ix.sum())
        for sv in sp[:, k]:
            for tv in tp[:, k]:
                total += 1 - calc(mu[ix, k] * np.exp(sv + tv))
        out[ix, k] = total / (len(sp) * len(tp))
    return out

def run_design(label, X, Y, s, t, c, mu, spool, tpool, replications, scenarios, rng):
    n, K = Y.shape
    G = s.max() + 1
    blocks = codes(t // 14)
    df = min(G - 1, blocks.max())
    rank = c.max() + 1 + X.shape[1]
    q = stats.t.ppf(1 - 0.05 / (2 * K), df)
    x = X - ag(X, c)[c] / np.bincount(c)[c, None]
    B = linalg.inv(x.T @ x)
    w = x @ B[:, 1]
    results = []
    alphas = []
    truthrows = []
    timing = []
    for scenario in scenarios:
        sp = np.zeros((1, K)) if scenario == 'independent_NB' else spool
        tp = tpool if scenario == 'source_and_block_slopes' else np.zeros((1, K))
        sample = rng.choice(n, size=min(n, 100000), replace=False)
        samplefactor = sp[rng.integers(len(sp), size=len(sample))] + tp[rng.integers(len(tp), size=len(sample))]
        mixed_mu = mu[sample] * np.exp(X[sample, 1, None] * samplefactor)
        observed_zero = (Y == 0).mean(0)
        alpha = []
        for k in range(K):

            def p0(a):
                return np.exp(-mixed_mu[:, k]).mean() if a < 1e-12 else np.exp(-np.log1p(a * mixed_mu[:, k]) / a).mean()
            binding = p0(0) >= observed_zero[k]
            a = 0.0 if binding else optimize.brentq(lambda a: p0(a) - observed_zero[k], 1e-10, 1000000.0)
            alpha.append(a)
            alphas.append(dict(design=label, scenario=scenario, channel=CHANNELS['facebook'][k], alpha=a, observed_zero=observed_zero[k], calibrated_zero=p0(a), Poisson_lower_bound_binding=binding))
        alpha = np.array(alpha)
        pi = activation_truth(mu, X[:, 1], alpha, sp, tp)
        truth = w @ pi
        for k, ch in enumerate(CHANNELS['facebook']):
            truthrows.append(dict(design=label, scenario=scenario, channel=ch, any_projection_truth=truth[k], PPML_logRR_truth=0.0))
        counts = {m: dict(valid=0, bad=0, opposition=0, invalid=0) for m in ['PPML', 'any_OLS']}
        zero = np.zeros(K)
        q99 = []
        start = time.perf_counter()
        for rep in range(replications):
            fs = sp[rng.integers(len(sp), size=G)]
            ft = tp[rng.integers(len(tp), size=blocks.max() + 1)]
            means = mu * np.exp(X[:, 1, None] * (fs[s] + ft[blocks]))
            yy = np.empty((n, K))
            for k in range(K):
                yy[:, k] = rng.poisson(means[:, k]) if alpha[k] < 1e-10 else rng.negative_binomial(1 / alpha[k], 1 / (1 + alpha[k] * means[:, k]))
            zero += (yy == 0).mean(0)
            q99.append(np.quantile(yy, 0.99, axis=0))
            Vlog = []
            bb = []
            allgood = True
            for k in range(K):
                try:
                    f = ppml(X, yy[:, k], c, tol=2e-08)
                    xb = X - f['mean'][c]
                    obs = xb @ f['bread'][:, 1] * (yy[:, k] - f['mu'])
                    V = multi_cov(obs[:, None], s, blocks, rank)['two_way'][0, 0]
                    if V <= 0 or not np.isfinite(V):
                        allgood = False
                        break
                    Vlog.append(V)
                    bb.append(f['beta'][1])
                except (RuntimeError, linalg.LinAlgError):
                    allgood = False
                    break
            if allgood:
                counts['PPML']['valid'] += 1
                T = np.array(bb) / np.sqrt(Vlog)
                reject = abs(T) > q
                counts['PPML']['bad'] += int(reject.any())
                counts['PPML']['opposition'] += int((reject & (T > 0)).any() and (reject & (T < 0)).any())
            else:
                counts['PPML']['invalid'] += 1
            binary = (yy > 0).astype(float)
            b = B @ x.T @ binary
            res = binary - ag(binary, c)[c] / np.bincount(c)[c, None] - x @ b
            V = multi_cov(w[:, None] * res, s, blocks, rank)['two_way']
            v = np.diag(V)
            if np.all(v > 0):
                counts['any_OLS']['valid'] += 1
                T = (b[1] - truth) / np.sqrt(v)
                reject = abs(T) > q
                counts['any_OLS']['bad'] += int(reject.any())
            else:
                counts['any_OLS']['invalid'] += 1
            if rep % 50 == 0:
                print(label, scenario, rep, '/', replications, flush=True)
        elapsed = time.perf_counter() - start
        timing.append(dict(design=label, scenario=scenario, replicates=replications, seconds=elapsed, n=n))
        for method, v in counts.items():
            valid = v['valid']
            lo, hi = wilson(v['bad'], valid) if valid else (np.nan, np.nan)
            results.append(dict(design=label, scenario=scenario, method=method, n=n, sources=int(G), cells=int(c.max() + 1), blocks=int(blocks.max() + 1), family=K, replicates=replications, valid=valid, invalid_fraction=v['invalid'] / replications, family_noncoverage=v['bad'] / valid if valid else np.nan, MC_low=lo, MC_high=hi, false_opposition=v['opposition'] / valid if valid and method == 'PPML' else np.nan))
        for k in range(K):
            alphas.append(dict(design=label, scenario=scenario, channel=CHANNELS['facebook'][k], simulated_zero=zero[k] / replications, observed_q99=np.quantile(Y[:, k], 0.99), mean_simulated_q99=np.mean(q99, axis=0)[k]))
        pd.DataFrame(results).to_csv(ROOT / 'results' / f'calibration_{label}.csv', index=False)
        pd.DataFrame(alphas).to_csv(ROOT / 'results' / f'calibration_{label}_fit.csv', index=False)
        pd.DataFrame(truthrows).to_csv(ROOT / 'results' / f'calibration_{label}_truth.csv', index=False)
        (ROOT / 'audit' / f'calibration_{label}_timing.json').write_text(json.dumps(timing, indent=2))

def main():
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    Y = d[CHANNELS['facebook']].to_numpy(float)
    s, t, c = grouping(d, 'facebook', 'date')
    rng = np.random.default_rng(SEED)
    poolfile = ROOT / 'results/calibration_pools.npz'
    if poolfile.exists() and '--reuse-pools' in sys.argv:
        pool = np.load(poolfile)
        spool, tpool = (pool['source'], pool['time'])
    else:
        spool, tpool = pools(d, X, Y, s, t, c)
    mu = []
    for k in range(Y.shape[1]):
        mu.append(ppml(np.delete(X, 1, axis=1), Y[:, k], c)['mu'])
    mu = np.array(mu).T
    chosen = []
    for sg in np.unique(s):
        ix = np.flatnonzero(s == sg)
        m = min(len(ix), max(20, int(np.ceil(len(ix) * 0.1))))
        chosen.extend(rng.choice(ix, m, replace=False))
    chosen = np.sort(chosen)
    np.save(ROOT / 'results/calibration_subsample_rows.npy', chosen)
    run_design('subsample', X[chosen], Y[chosen], codes(s[chosen]), t[chosen], codes(c[chosen]), mu[chosen], spool, tpool, 500, ['independent_NB', 'source_slopes', 'source_and_block_slopes'], rng)
    run_design('full', X, Y, s, t, c, mu, spool, tpool, 100, ['source_and_block_slopes'], rng)
    (ROOT / 'audit/empirical_calibration.json').write_text(json.dumps(dict(seed=SEED, subsample_fraction=0.1, all_sources_retained=True, source_pool_rows=len(spool), time_pool_rows=len(tpool), scope='Finite fitted DGPs with outcome-informed structural-zero means and noisy empirical random-slope supports. Not an empirical bound, independent dataset, or validation of unknown tails/dependence.'), indent=2))
if __name__ == '__main__':
    main()
