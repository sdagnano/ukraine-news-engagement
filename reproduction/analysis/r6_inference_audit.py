from cell_models import *
from empirical_calibration import activation_truth, wilson, SEED
import argparse, time, datetime, hashlib, warnings
PREFIX = 'r6_inference_'

def save_csv(name, rows):
    pd.DataFrame(rows).to_csv(ROOT / 'results' / (PREFIX + name + '.csv'), index=False)

def cov_setup(s, t, n, rank):
    cross = codes(pd.MultiIndex.from_arrays([s, t]))
    return [(s, cr1(s, n, rank)), (t, cr1(t, n, rank)), (cross, cr1(cross, n, rank))]

def covariance_fast(obs, setup):
    parts = []
    for g, factor in setup:
        a = ag(obs, g)
        parts.append(a.T @ a * factor)
    return (parts[0] + parts[1] - parts[2], parts)

def failure_reason(error, X, y, c):
    message = str(error)
    if y.sum() <= 0:
        return ('no_positive_response', 0, message)
    positive_cells = ag(y, c) > 0
    ix = positive_cells[c]
    ng = np.bincount(c)
    xm = ag(X, c) / ng[:, None]
    rank = int(np.linalg.matrix_rank(X[ix] - xm[c[ix]]))
    if rank < X.shape[1]:
        return ('nonidentified_active_design', rank, message)
    if 'line search' in message or 'did not converge' in message:
        return ('solver_nonconvergence', rank, message)
    if 'Divergent' in message:
        return ('divergent_possible_separation_unresolved', rank, message)
    return ('ill_conditioned_information_possible_separation_unresolved', rank, message)

def observed_data():
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    Y = d[CHANNELS['facebook']].to_numpy(float)
    s, t, c = grouping(d, 'facebook', 'date')
    return (d, X, Y, s, t, c)

def calibrate(X, Y, s, t, c, mu, sp, tp, rng, label, scenario):
    n, K = Y.shape
    sample = rng.choice(n, size=min(n, 100000), replace=False)
    sf = sp[rng.integers(len(sp), size=len(sample))] + tp[rng.integers(len(tp), size=len(sample))]
    mixed = mu[sample] * np.exp(X[sample, 1, None] * sf)
    observed_zero = (Y == 0).mean(0)
    alpha = []
    rows = []
    for k, ch in enumerate(CHANNELS['facebook']):

        def zero(a):
            return np.exp(-mixed[:, k]).mean() if a < 1e-12 else np.exp(-np.log1p(a * mixed[:, k]) / a).mean()
        binding = zero(0) >= observed_zero[k]
        a = 0.0 if binding else optimize.brentq(lambda a: zero(a) - observed_zero[k], 1e-10, 1000000.0)
        alpha.append(a)
        rows.append(dict(design=label, scenario=scenario, channel=ch, alpha=a, observed_zero=observed_zero[k], calibrated_zero=zero(a), Poisson_lower_bound_binding=binding))
    return (np.array(alpha), rows)

def run_scenario(label, scenario, X, Y, s, t, c, mu, sp, tp, rng, reps, refit=True):
    n, K = Y.shape
    G = s.max() + 1
    blocks = codes(t // 14)
    rank = c.max() + 1 + X.shape[1]
    df = min(G - 1, blocks.max())
    q = stats.t.ppf(1 - 0.05 / (2 * K), df)
    q1 = stats.t.ppf(0.975, df)
    setup = cov_setup(s, blocks, n, rank)
    ng = np.bincount(c)
    x = X - ag(X, c)[c] / ng[c, None]
    B = linalg.inv(x.T @ x)
    w = x @ B[:, 1]
    alpha, arows = calibrate(X, Y, s, t, c, mu, sp, tp, rng, label, scenario)
    truth = w @ activation_truth(mu, X[:, 1], alpha, sp, tp) if refit else None
    rows = []
    trials = []
    start = time.perf_counter()
    for rep in range(reps):
        fs = sp[rng.integers(len(sp), size=G)]
        ft = tp[rng.integers(len(tp), size=blocks.max() + 1)]
        means = mu * np.exp(X[:, 1, None] * (fs[s] + ft[blocks]))
        yy = np.empty((n, K))
        for k in range(K):
            yy[:, k] = rng.poisson(means[:, k]) if alpha[k] < 1e-10 else rng.negative_binomial(1 / alpha[k], 1 / (1 + alpha[k] * means[:, k]))
        if not refit:
            if rep % 100 == 0:
                print(label, scenario, 'RNG replay', rep, flush=True)
            continue
        current = []
        for k, ch in enumerate(CHANNELS['facebook']):
            row = dict(design=label, scenario=scenario, replicate=rep, method='PPML', channel=ch, truth=0.0, estimate=np.nan, se=np.nan, variance=np.nan, fit_valid=False, channel_valid=False, status='', failure_message='', active_design_rank=np.nan, zero_cells=int((ag(yy[:, k], c) == 0).sum()), pointwise_noncoverage=np.nan, simultaneous_noncoverage=np.nan)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', linalg.LinAlgWarning)
                    f = ppml(X, yy[:, k], c, tol=2e-08)
                xb = X - f['mean'][c]
                obs = (xb @ f['bread'][:, 1] * (yy[:, k] - f['mu']))[:, None]
                V, parts = covariance_fast(obs, setup)
                v = float(V[0, 0])
                if rep == 0 and k == 0:
                    reference = multi_cov(obs, s, blocks, rank)['two_way'][0, 0]
                    assert abs(reference - v) < 1e-12 * max(1, abs(v))
                row.update(estimate=float(f['beta'][1]), variance=v, fit_valid=True, condition=f['condition'], gradient=f['gradient'], iterations=f['iterations'], source_variance=float(parts[0][0, 0]), block_variance=float(parts[1][0, 0]), intersection_variance=float(parts[2][0, 0]))
                if np.isfinite(v) and v > 0:
                    se = np.sqrt(v)
                    T = f['beta'][1] / se
                    row.update(se=float(se), channel_valid=True, status='valid', pointwise_noncoverage=bool(abs(T) > q1), simultaneous_noncoverage=bool(abs(T) > q))
                else:
                    row.update(status='nonpositive_two_way_variance' if np.isfinite(v) else 'nonfinite_two_way_variance')
            except (RuntimeError, linalg.LinAlgError) as error:
                status, dr, message = failure_reason(error, X, yy[:, k], c)
                row.update(status=status, active_design_rank=dr, failure_message=message)
            current.append(row)
        binary = (yy > 0).astype(float)
        b = B @ x.T @ binary
        res = binary - ag(binary, c)[c] / ng[c, None] - x @ b
        V, parts = covariance_fast(w[:, None] * res, setup)
        v = np.diag(V)
        for k, ch in enumerate(CHANNELS['facebook']):
            valid = bool(np.isfinite(v[k]) and v[k] > 0)
            se = float(np.sqrt(v[k])) if valid else np.nan
            T = (b[1, k] - truth[k]) / se
            current.append(dict(design=label, scenario=scenario, replicate=rep, method='any_OLS', channel=ch, truth=truth[k], estimate=b[1, k], se=se, variance=v[k], fit_valid=True, channel_valid=valid, status='valid' if valid else 'nonpositive_two_way_variance' if np.isfinite(v[k]) else 'nonfinite_two_way_variance', failure_message='', source_variance=parts[0][k, k], block_variance=parts[1][k, k], intersection_variance=parts[2][k, k], zero_cells=int((ag(binary[:, k], c) == 0).sum()), pointwise_noncoverage=bool(abs(T) > q1) if valid else np.nan, simultaneous_noncoverage=bool(abs(T) > q) if valid else np.nan))
        for method in ['PPML', 'any_OLS']:
            rr = [r for r in current if r['method'] == method]
            valid = all((r['channel_valid'] for r in rr))
            bad = any((r['simultaneous_noncoverage'] for r in rr)) if valid else False
            opposition = bool(any((r['simultaneous_noncoverage'] and r['estimate'] > 0 for r in rr)) and any((r['simultaneous_noncoverage'] and r['estimate'] < 0 for r in rr))) if valid and method == 'PPML' else np.nan
            statuses = sorted(set((r['status'] for r in rr if r['status'] != 'valid')))
            trial = dict(design=label, scenario=scenario, replicate=rep, method=method, n=n, sources=int(G), cells=int(c.max() + 1), blocks=int(blocks.max() + 1), df=int(df), family=K, valid=valid, joint_noncoverage=bad if valid else np.nan, false_opposition=opposition, operational_failure=bool(not valid or bad), failure_statuses=';'.join(statuses))
            trials.append(trial)
            for r in rr:
                r['family_valid'] = valid
        rows.extend(current)
        if rep % 25 == 0 or rep == reps - 1:
            print(label, scenario, rep + 1, '/', reps, 'seconds', round(time.perf_counter() - start, 1), flush=True)
            save_csv(label + '_' + scenario + '_channels', rows)
            save_csv(label + '_' + scenario + '_trials', trials)
    return (rows, trials, arows, dict(design=label, scenario=scenario, replicates=reps, refitted=refit, seconds=time.perf_counter() - start))

def simulate():
    d, X, Y, s, t, c = observed_data()
    rng = np.random.default_rng(SEED)
    pools = np.load(ROOT / 'results/calibration_pools.npz')
    spool = pools['source']
    tpool = pools['time']
    mu = np.column_stack([ppml(np.delete(X, 1, axis=1), Y[:, k], c)['mu'] for k in range(Y.shape[1])])
    chosen = []
    for sg in np.unique(s):
        ix = np.flatnonzero(s == sg)
        m = min(len(ix), max(20, int(np.ceil(len(ix) * 0.1))))
        chosen.extend(rng.choice(ix, m, replace=False))
    chosen = np.sort(chosen)
    assert np.array_equal(chosen, np.load(ROOT / 'results/calibration_subsample_rows.npy'))
    density = []
    for label, cc, nn in [('full', c, len(c)), ('subsample', codes(c[chosen]), len(chosen))]:
        counts = np.bincount(cc)
        density.append(dict(design=label, posts=nn, cells=len(counts), posts_per_cell=nn / len(counts), singleton_cells=int((counts == 1).sum()), median_cell_size=float(np.median(counts)), minimum_cell_size=int(counts.min()), maximum_cell_size=int(counts.max())))
    save_csv('design_density', density)
    timing = []
    allalpha = []
    for scenario in ['independent_NB', 'source_slopes', 'source_and_block_slopes']:
        sp = np.zeros((1, Y.shape[1])) if scenario == 'independent_NB' else spool
        tp = tpool if scenario == 'source_and_block_slopes' else np.zeros((1, Y.shape[1]))
        _, _, aa, tt = run_scenario('subsample', scenario, X[chosen], Y[chosen], codes(s[chosen]), t[chosen], codes(c[chosen]), mu[chosen], sp, tp, rng, 500, refit=scenario == 'independent_NB')
        allalpha.extend(aa)
        timing.append(tt)
    _, _, aa, tt = run_scenario('full', 'source_and_block_slopes', X, Y, s, t, c, mu, spool, tpool, rng, 100, refit=True)
    allalpha.extend(aa)
    timing.append(tt)
    save_csv('DGP_alpha', allalpha)
    (ROOT / 'audit' / (PREFIX + 'timing.json')).write_text(json.dumps(timing, indent=2))
    summarize()

def summarize():
    channels = pd.concat([pd.read_csv(p) for p in (ROOT / 'results').glob(PREFIX + '*_*_channels.csv')], ignore_index=True)
    trials = pd.concat([pd.read_csv(p) for p in (ROOT / 'results').glob(PREFIX + '*_*_trials.csv')], ignore_index=True)
    out = []
    operations = []
    causes = []
    reproduction = []
    estimators = []
    for key, g in channels.groupby(['design', 'scenario', 'method', 'channel'], sort=False):
        fitted = g[g.fit_valid]
        err = fitted.estimate - fitted.truth
        iv = g[g.channel_valid]
        estimators.append(dict(zip(['design', 'scenario', 'method', 'channel'], key), replicates=len(g), fit_valid=len(fitted), interval_valid=len(iv), bias=float(err.mean()), RMSE=float(np.sqrt(np.mean(err ** 2))), empirical_SD=float(fitted.estimate.std(ddof=1)), mean_SE_interval_valid=float(iv.se.mean()), RMS_SE_interval_valid=float(np.sqrt(np.mean(iv.se ** 2))), mean_SE_interval_valid_over_SD_all_fitted=float(iv.se.mean() / fitted.estimate.std(ddof=1)), pointwise_operational_failure_count=int((~g.channel_valid).sum() + iv.pointwise_noncoverage.sum()), Bonferroni_component_operational_failure_count=int((~g.channel_valid).sum() + iv.simultaneous_noncoverage.sum())))
        for conditioning in ['channel_valid', 'family_valid']:
            v = g[g[conditioning]]
            err = v.estimate - v.truth
            ns = len(v)
            for band, col in [('pointwise95', 'pointwise_noncoverage'), ('Bonferroni8', 'simultaneous_noncoverage')]:
                nc = int(v[col].sum())
                lo, hi = wilson(nc, ns) if ns else (np.nan, np.nan)
                out.append(dict(zip(['design', 'scenario', 'method', 'channel'], key), conditioning=conditioning, interval=band, replicates=len(g), valid=ns, invalid_fraction=1 - ns / len(g), bias=float(err.mean()), RMSE=float(np.sqrt(np.mean(err ** 2))), empirical_SD=float(v.estimate.std(ddof=1)), mean_SE=float(v.se.mean()), RMS_SE=float(np.sqrt(np.mean(v.se ** 2))), mean_SE_over_empirical_SD=float(v.se.mean() / v.estimate.std(ddof=1)), noncoverage=nc / ns if ns else np.nan, MC_low=lo, MC_high=hi))
        for status, h in g.groupby('status'):
            causes.append(dict(zip(['design', 'scenario', 'method', 'channel'], key), status=status, realizations=len(h), fraction=len(h) / len(g)))
    for key, g in trials.groupby(['design', 'scenario', 'method'], sort=False):
        v = g[g.valid]
        bad = int(v.joint_noncoverage.sum())
        invalid = int((~g.valid).sum())
        failure = int(g.operational_failure.sum())
        lo, hi = wilson(failure, len(g))
        row = dict(zip(['design', 'scenario', 'method'], key), replicates=len(g), valid=len(v), invalid=invalid, invalid_fraction=invalid / len(g), joint_noncoverage_count=bad, conditional_joint_noncoverage=bad / len(v), operational_failure_count=failure, operational_failure_rate=failure / len(g), operational_failure_MC_low=lo, operational_failure_MC_high=hi, false_opposition_count=int(v.false_opposition.sum()) if key[2] == 'PPML' else np.nan)
        operations.append(row)
        original = pd.read_csv(ROOT / 'results' / f'calibration_{key[0]}.csv')
        reference = original[(original.scenario == key[1]) & (original.method == key[2])].iloc[0]
        reproduction.append(dict(**dict(zip(['design', 'scenario', 'method'], key)), valid_matches=bool(len(v) == reference.valid), conditional_noncoverage_matches=bool(abs(row['conditional_joint_noncoverage'] - reference.family_noncoverage) < 1e-12), original_valid=int(reference.valid), recovered_valid=len(v), original_noncoverage=float(reference.family_noncoverage), recovered_noncoverage=row['conditional_joint_noncoverage']))
    save_csv('channel_summary', out)
    save_csv('estimator_summary', estimators)
    save_csv('operational_summary', operations)
    save_csv('failure_causes', causes)
    save_csv('reproduction', reproduction)
    (ROOT / 'audit' / (PREFIX + 'reproduction.json')).write_text(json.dumps(dict(seed=SEED, all_aggregate_results_reproduced=all((r['valid_matches'] and r['conditional_noncoverage_matches'] for r in reproduction)), checks=reproduction, conditioning='Channel-specific results are supplied both conditional on own-channel validity and on the validity of the full eight-channel family. Operational failure equals invalid family OR simultaneous noncoverage.', separation='Solver messages are classified using positive-cell design rank. Full-rank ill conditioning or divergence is not a mathematical separation diagnosis.', target='Full source/block DGP reference marginalizes redrawn slopes. Coverage is not assessed for a fixed finite-source conditional parameter, and does not estimate empirical-data error probabilities.'), indent=2))

def temporal():
    d, X, Y, s, t, c = observed_data()
    n, K = Y.shape
    G = s.max() + 1
    T = t.max() + 1
    rank = c.max() + 1 + X.shape[1]
    datevalues = pd.to_datetime(d.date).sort_values().unique()
    first = np.unique(c, return_index=True)[1]
    cs = s[first]
    ct = t[first]
    N = len(first)
    z = cell_ols(X, np.column_stack([Y, (Y > 0).astype(float)]), c)
    cellscore = {}
    beta = {}
    cellscore['count_OLS'] = ag(z['obs'][:, :K], c)
    beta['count_OLS'] = z['beta'][1, :K]
    cellscore['any_OLS'] = ag(z['obs'][:, K:], c)
    beta['any_OLS'] = z['beta'][1, K:]
    del z
    pp = []
    bb = []
    for k, ch in enumerate(CHANNELS['facebook']):
        f = ppml(X, Y[:, k], c)
        ob = (X - f['mean'][c]) @ f['bread'][:, 1] * (Y[:, k] - f['mu'])
        pp.append(ag(ob, c))
        bb.append(f['beta'][1])
    cellscore['PPML'] = np.column_stack(pp)
    beta['PPML'] = np.array(bb)
    rows = []
    acfs = []
    hac = []
    time_rows = []
    for method, score in cellscore.items():
        source = ag(score, cs)
        vs = source.T @ source * G / (G - 1) * (n - 1) / (n - rank)
        for width in [7, 14, 28]:
            for origin in range(width):
                block = codes((ct - origin) // width)
                B = block.max() + 1
                cross = codes(pd.MultiIndex.from_arrays([cs, block]))
                C = cross.max() + 1
                sb = ag(score, block)
                sc = ag(score, cross)
                V = vs + sb.T @ sb * B / (B - 1) * (n - 1) / (n - rank) - sc.T @ sc * C / (C - 1) * (n - 1) / (n - rank)
                df = min(G - 1, B - 1)
                q = stats.t.ppf(1 - 0.05 / (2 * K), df)
                for k, ch in enumerate(CHANNELS['facebook']):
                    v = V[k, k]
                    se = np.sqrt(v) if v > 0 else np.nan
                    rows.append(dict(method=method, channel=ch, width=width, origin_days=origin, blocks=B, df=df, estimate=beta[method][k], variance=v, se=se, low=beta[method][k] - q * se, high=beta[method][k] + q * se, valid=bool(v > 0)))
        daily = ag(score, ct)
        tensor = np.zeros((G, T, K))
        tensor[cs, ct] = score
        tensor -= tensor.mean(1, keepdims=True)
        centered = tensor.sum(0)
        for k, ch in enumerate(CHANNELS['facebook']):
            a = centered[:, k]
            energy = a @ a
            for j in range(T):
                time_rows.append(dict(method=method, channel=ch, date=str(datevalues[j])[:10], day=j, influence_sum=daily[j, k], centered_influence_sum=a[j]))
            for lag in range(1, 43):
                total = float(a[lag:] @ a[:-lag])
                within = float(np.sum(tensor[:, lag:, k] * tensor[:, :-lag, k]))
                between = total - within
                acfs.append(dict(method=method, channel=ch, lag=lag, daily_score_ACF=total / energy, overlap_Pearson=float(np.corrcoef(a[lag:], a[:-lag])[0, 1]), total_lag_cross_product=total, within_source_lag_cross_product=within, cross_source_lag_cross_product=between, cross_source_lag_product_over_daily_energy=between / energy))
            for lagmax in [3, 7, 14, 28, 42]:
                v = energy + 2 * sum(((1 - lag / (lagmax + 1)) * (a[lag:] @ a[:-lag]) for lag in range(1, lagmax + 1)))
                v *= T / (T - 1) * (n - 1) / (n - rank)
                hac.append(dict(method=method, channel=ch, lagmax=lagmax, daily_Bartlett_HAC_variance=v, daily_Bartlett_HAC_SE=np.sqrt(max(0, v)), note='Descriptive working SE under lag-decay/time-asymptotic assumptions; no validated interval or small-cluster remedy.'))
    save_csv('block_origins', rows)
    save_csv('score_ACF', acfs)
    save_csv('daily_scores', time_rows)
    save_csv('daily_HAC', hac)
    frame = pd.DataFrame(rows)
    summary = []
    for key, g in frame.groupby(['method', 'channel', 'width'], sort=False):
        summary.append(dict(zip(['method', 'channel', 'width'], key), origins=len(g), valid=int(g.valid.sum()), min_blocks=int(g.blocks.min()), max_blocks=int(g.blocks.max()), min_SE=g.se.min(), max_SE=g.se.max(), min_low=g.low.min(), max_low=g.low.max(), min_high=g.high.min(), max_high=g.high.max(), positive_intervals=int((g.low > 0).sum()), negative_intervals=int((g.high < 0).sum()), intervals_containing_zero=int(((g.low <= 0) & (g.high >= 0)).sum())))
    save_csv('block_origin_summary', summary)
    (ROOT / 'audit' / (PREFIX + 'temporal.json')).write_text(json.dumps(dict(n=n, sources=int(G), dates=int(T), cells=N, origins='All integer offsets 0,...,width-1 for widths 7,14,28, relative to 2022-02-25. Partial boundary blocks are retained; df changes with block count.', score='Post-level estimating-equation influence sums by source and calendar date. Daily ACF uses source-wise demeaning over all 200 dates; signed cross-source lag cross-products are descriptive and can exceed the daily-energy normalization.', limitation='Observed score autocorrelation cannot identify a true dependence horizon. Two-way source/block clustering omits different-source different-block covariance. Alternative origins and Bartlett HAC are diagnostics, not validated confidence procedures.'), indent=2))
    print('Temporal diagnostics complete', flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['calibration', 'temporal', 'summarize', 'all'])
    args = parser.parse_args()
    if args.mode in ['temporal', 'all']:
        temporal()
    if args.mode in ['calibration', 'all']:
        simulate()
    if args.mode == 'summarize':
        summarize()
