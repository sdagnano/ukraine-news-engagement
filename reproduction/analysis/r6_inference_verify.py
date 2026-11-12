import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from pathlib import Path
import json
import hashlib
import sys
ROOT = Path(__file__).resolve().parents[1]
for name in ['pydeps', 'readr_deps']:
    dependency_path = ROOT.parents[1] / 'work' / name
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path))
import numpy as np
import pandas as pd
from scipy import stats
R = ROOT / 'results'
A = ROOT / 'audit'
NAMES = ['Likes', 'Love', 'Angry', 'Haha', 'Sad', 'Wow', 'Comments', 'Shares']

def close(actual, expected, name, atol=2e-11, rtol=2e-09):
    assert np.allclose(actual, expected, atol=atol, rtol=rtol, equal_nan=True), name

def read(name):
    return pd.read_csv(R / ('r6_inference_' + name + '.csv'))

def bygroup(values, groups):
    out = np.zeros((int(groups.max()) + 1,) + values.shape[1:])
    np.add.at(out, groups, values)
    return out

def wilson95(x, n):
    z = stats.norm.ppf(0.975)
    den = n + z * z
    center = (x + z * z / 2) / den
    half = z * np.sqrt(x * (1 - x / n) + z * z / 4) / den
    return (center - half, center + half)

def verify_saved_replicates():
    all_channels = []
    all_trials = []
    for tag in ['subsample_independent_NB', 'full_source_and_block_slopes']:
        channels = read(tag + '_channels')
        trials = read(tag + '_trials')
        assert not channels.duplicated(['replicate', 'method', 'channel']).any()
        assert channels.groupby(['replicate', 'method']).size().eq(8).all()
        close(channels.variance, channels.source_variance + channels.block_variance - channels.intersection_variance, 'covariance inclusion-exclusion')
        for trial in trials.itertuples():
            g = channels[(channels.replicate == trial.replicate) & (channels.method == trial.method)]
            valid = bool(g.channel_valid.all())
            assert valid == trial.valid and g.family_valid.eq(valid).all()
            good = g[g.channel_valid]
            assert np.isfinite(good.se).all() and good.se.gt(0).all()
            close(good.se ** 2, good.variance, 'SE squared')
            t = np.abs((good.estimate - good.truth) / good.se)
            assert np.array_equal(t.gt(stats.t.ppf(0.975, trial.df)), good.pointwise_noncoverage.astype(bool))
            assert np.array_equal(t.gt(stats.t.ppf(1 - 0.05 / 16, trial.df)), good.simultaneous_noncoverage.astype(bool))
            if valid:
                bad = bool(g.simultaneous_noncoverage.any())
                assert bad == trial.joint_noncoverage
            else:
                bad = False
                assert pd.isna(trial.joint_noncoverage)
            assert bool(not valid or bad) == trial.operational_failure
        all_channels.append(channels)
        all_trials.append(trials)
    channels = pd.concat(all_channels, ignore_index=True)
    trials = pd.concat(all_trials, ignore_index=True)
    summary = read('channel_summary')
    for row in summary.itertuples():
        g = channels[(channels.design == row.design) & (channels.scenario == row.scenario) & (channels.method == row.method) & (channels.channel == row.channel)]
        v = g[g[row.conditioning]]
        error = v.estimate.to_numpy() - v.truth.to_numpy()
        assert len(v) == row.valid and len(g) == row.replicates
        close([error.mean(), np.sqrt(error @ error / len(error)), v.estimate.std(ddof=1), v.se.mean()], [row.bias, row.RMSE, row.empirical_SD, row.mean_SE], 'channel moments')
        close(row.RMSE ** 2, row.bias ** 2 + (len(v) - 1) / len(v) * row.empirical_SD ** 2, 'RMSE identity')
        col = 'pointwise_noncoverage' if row.interval == 'pointwise95' else 'simultaneous_noncoverage'
        x = int(v[col].sum())
        close([x / len(v), *wilson95(x, len(v))], [row.noncoverage, row.MC_low, row.MC_high], 'channel coverage and Wilson interval')
    estimator_summary = read('estimator_summary')
    for row in estimator_summary.itertuples():
        g = channels[(channels.design == row.design) & (channels.scenario == row.scenario) & (channels.method == row.method) & (channels.channel == row.channel)]
        fitted = g[g.fit_valid]
        iv = g[g.channel_valid]
        error = fitted.estimate.to_numpy() - fitted.truth.to_numpy()
        assert (len(g), len(fitted), len(iv)) == (row.replicates, row.fit_valid, row.interval_valid)
        close([error.mean(), np.sqrt(error @ error / len(error)), fitted.estimate.std(ddof=1), iv.se.mean()], [row.bias, row.RMSE, row.empirical_SD, row.mean_SE_interval_valid], 'all-fit estimator moments')
        assert int((~g.channel_valid).sum() + iv.pointwise_noncoverage.sum()) == row.pointwise_operational_failure_count
        assert int((~g.channel_valid).sum() + iv.simultaneous_noncoverage.sum()) == row.Bonferroni_component_operational_failure_count
    operational = read('operational_summary')
    reproduction = []
    for row in operational.itertuples():
        g = trials[(trials.design == row.design) & (trials.scenario == row.scenario) & (trials.method == row.method)]
        valid = g[g.valid]
        invalid = int((~g.valid).sum())
        bad = int(valid.joint_noncoverage.sum())
        failure = int(g.operational_failure.sum())
        assert (len(g), len(valid), invalid, bad, failure) == (row.replicates, row.valid, row.invalid, row.joint_noncoverage_count, row.operational_failure_count)
        assert failure == invalid + bad
        close([failure / len(g), *wilson95(failure, len(g))], [row.operational_failure_rate, row.operational_failure_MC_low, row.operational_failure_MC_high], 'operational Wilson interval')
        old = pd.read_csv(R / ('calibration_' + row.design + '.csv'))
        old = old[(old.scenario == row.scenario) & (old.method == row.method)].iloc[0]
        assert len(valid) == old.valid
        close(bad / len(valid), old.family_noncoverage, 'frozen calibration aggregate')
        reproduction.append(dict(design=row.design, method=row.method, valid=len(valid), simultaneous_noncoverage=bad, operational_failure=failure))
    causes = channels[~channels.channel_valid].groupby(['design', 'method', 'status']).size()
    assert channels.fit_valid.all(), 'All saved fitted models must be recorded as fitted.'
    assert set(channels[~channels.channel_valid].status) == {'nonpositive_two_way_variance'}
    return (channels, trials, dict(channel_rows=len(channels), trial_rows=len(trials), channel_summary_rows=len(summary), all_fit_estimator_summary_rows=len(estimator_summary), all_fits_valid=bool(channels.fit_valid.all()), invalid_causes={str(k): int(v) for k, v in causes.items()}, aggregate_reproduction=reproduction))

def verify_daily_saved_scores():
    daily = read('daily_scores')
    acf = read('score_ACF')
    hac = read('daily_HAC')
    p = len(np.load(R / 'cell_model_parameters.npz')['facebook_date_all_Likes'])
    for (method, channel), g in daily.groupby(['method', 'channel']):
        g = g.sort_values('day')
        a = g.influence_sum.to_numpy()
        a = a - a.mean()
        close(a, g.centered_influence_sum, 'daily source-wise versus total demeaning')
        energy = a @ a
        for row in acf[(acf.method == method) & (acf.channel == channel)].itertuples():
            total = a[row.lag:] @ a[:-row.lag]
            close([total, total / energy, np.corrcoef(a[row.lag:], a[:-row.lag])[0, 1]], [row.total_lag_cross_product, row.daily_score_ACF, row.overlap_Pearson], 'saved daily ACF')
            close(row.total_lag_cross_product, row.within_source_lag_cross_product + row.cross_source_lag_cross_product, 'lag cross-product decomposition')
            close(row.cross_source_lag_cross_product / energy, row.cross_source_lag_product_over_daily_energy, 'cross-source normalization')
        for row in hac[(hac.method == method) & (hac.channel == channel)].itertuples():
            lagmax = row.lagmax
            v = energy + 2 * sum(((1 - lag / (lagmax + 1)) * (a[lag:] @ a[:-lag]) for lag in range(1, lagmax + 1)))
            v *= 200 / 199 * (535936 - 1) / (535936 - 14689 - p)
            close([v, np.sqrt(max(v, 0))], [row.daily_Bartlett_HAC_variance, row.daily_Bartlett_HAC_SE], 'daily Bartlett HAC')
    return dict(daily_rows=len(daily), ACF_rows=len(acf), HAC_rows=len(hac))

def verify_observed_scores():
    from cell_models import load, ukrainian_design
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    Y = d[NAMES].to_numpy(float)
    s = pd.factorize(d['Page.Name'], sort=True)[0]
    dates = pd.to_datetime(d.date)
    t = ((dates - dates.min()) / pd.Timedelta(days=1)).to_numpy(int)
    c = pd.factorize(pd.MultiIndex.from_arrays([s, t]), sort=True)[0]
    n, p = X.shape
    G, T, C = (int(s.max() + 1), int(t.max() + 1), int(c.max() + 1))
    assert (n, G, T, C) == (535936, 84, 200, 14689)
    assert np.array_equal(np.unique(t), np.arange(T)), 'Temporal groups must be calendar days without gaps.'
    counts = np.bincount(c)
    x = X - bygroup(X, c)[c] / counts[c, None]
    b = np.linalg.solve(x.T @ x, x.T @ np.column_stack([Y, Y > 0]))
    resid = np.column_stack([Y, Y > 0]) - bygroup(np.column_stack([Y, Y > 0]), c)[c] / counts[c, None] - x @ b
    w = x @ np.linalg.solve(x.T @ x, np.eye(p)[:, 1])
    scores = {'count_OLS': w[:, None] * resid[:, :8], 'any_OLS': w[:, None] * resid[:, 8:]}
    betas = {'count_OLS': b[1, :8], 'any_OLS': b[1, 8:]}
    cached = np.load(R / 'cell_model_parameters.npz')
    ps = []
    pb = []
    profile_gradient = []
    for k, ch in enumerate(NAMES):
        beta = cached['facebook_date_all_' + ch]
        eta = X @ beta
        mx = np.full(C, -np.inf)
        np.maximum.at(mx, c, eta)
        e = np.exp(eta - mx[c])
        yt = bygroup(Y[:, k], c)
        mu = yt[c] * e / bygroup(e, c)[c]
        wx = bygroup(mu[:, None] * X, c)
        mean = np.divide(wx, yt[:, None], out=np.zeros_like(wx), where=yt[:, None] > 0)
        xb = X - mean[c]
        information = xb.T @ (mu[:, None] * xb)
        q = np.linalg.solve(information, np.eye(p)[:, 1])
        ps.append(xb @ q * (Y[:, k] - mu))
        pb.append(beta[1])
        profile_gradient.append(float(np.abs(X.T @ (Y[:, k] - mu)).max() / Y[:, k].sum()))
    scores['PPML'] = np.column_stack(ps)
    betas['PPML'] = np.array(pb)
    origins = read('block_origins')
    acf = read('score_ACF')
    daily = read('daily_scores')
    covariance_rows_checked = 0
    for method, score in scores.items():
        source = bygroup(score, s)
        vs = np.sum(source ** 2, axis=0) * G / (G - 1) * (n - 1) / (n - C - p)
        cube = np.zeros((G, T, 8))
        np.add.at(cube, (s, t), score)
        cube -= cube.mean(1, keepdims=True)
        for k, ch in enumerate(NAMES):
            g = daily[(daily.method == method) & (daily.channel == ch)].sort_values('day')
            close(bygroup(score[:, k], t), g.influence_sum, 'reconstructed daily scores', atol=2e-09)
            aa = cube[:, :, k].sum(0)
            for row in acf[(acf.method == method) & (acf.channel == ch)].itertuples():
                within = np.sum(cube[:, row.lag:, k] * cube[:, :-row.lag, k])
                close(within, row.within_source_lag_cross_product, 'within-source lag cross-products', atol=2e-09)
        for width in [7, 14, 28]:
            for origin in range(width):
                block = pd.factorize((t - origin) // width, sort=True)[0]
                cross = pd.factorize(pd.MultiIndex.from_arrays([s, block]), sort=True)[0]
                B, J = (int(block.max() + 1), int(cross.max() + 1))
                vb = np.sum(bygroup(score, block) ** 2, axis=0) * B / (B - 1) * (n - 1) / (n - C - p)
                vi = np.sum(bygroup(score, cross) ** 2, axis=0) * J / (J - 1) * (n - 1) / (n - C - p)
                variance = vs + vb - vi
                df = min(G - 1, B - 1)
                q = stats.t.ppf(1 - 0.05 / 16, df)
                g = origins[(origins.method == method) & (origins.width == width) & (origins.origin_days == origin)].set_index('channel').loc[NAMES]
                close([B, df], [g.blocks.iloc[0], g.df.iloc[0]], 'calendar block counts')
                close(variance, g.variance, 'post-level reconstructed origin covariance', atol=2e-08)
                close(betas[method], g.estimate, 'cached and reconstructed estimates', atol=2e-08)
                close(np.array([betas[method] - q * np.sqrt(variance), betas[method] + q * np.sqrt(variance)]), np.array([g.low, g.high]), 'origin working bands', atol=3e-08)
                covariance_rows_checked += len(g)
    density = read('design_density')
    chosen = np.load(R / 'calibration_subsample_rows.npy')
    for label, cc in [('full', c), ('subsample', pd.factorize(c[chosen], sort=True)[0])]:
        nn = np.bincount(cc)
        g = density[density.design == label].iloc[0]
        close([len(cc), len(nn), len(cc) / len(nn), (nn == 1).sum(), np.median(nn), nn.min(), nn.max()], [g.posts, g.cells, g.posts_per_cell, g.singleton_cells, g.median_cell_size, g.minimum_cell_size, g.maximum_cell_size], 'design density')
    return dict(block_origin_rows_independently_reconstructed=covariance_rows_checked, observed_profile_gradient_max=max(profile_gradient), all_200_codes_are_consecutive_calendar_days=True)

def verify_marginal_targets(channels):
    pools = np.load(R / 'calibration_pools.npz')
    close(np.exp(pools['source']).mean(0), np.ones(8), 'source marginal multiplicative normalization')
    close(np.exp(pools['time']).mean(0), np.ones(8), 'block marginal multiplicative normalization')
    for design in ['full', 'subsample']:
        old = pd.read_csv(R / ('calibration_' + design + '_truth.csv'))
        for method in ['PPML', 'any_OLS']:
            col = 'PPML_logRR_truth' if method == 'PPML' else 'any_projection_truth'
            for ch in NAMES:
                g = channels[(channels.design == design) & (channels.method == method) & (channels.channel == ch)]
                reference = old[(old.scenario == g.scenario.iloc[0]) & (old.channel == ch)][col].iloc[0]
                close(g.truth.to_numpy(), reference, 'unchanged marginal truth')
    return dict(source_pool_rows=len(pools['source']), block_pool_rows=len(pools['time']), multiplicative_means_are_one=True, all_saved_targets_match_frozen_targets=True)
if __name__ == '__main__':
    channels, trials, replicate_checks = verify_saved_replicates()
    checks = dict(scope='Independent arithmetic audit; no simulation or optimizer refit.', replicates=replicate_checks, daily_scores=verify_daily_saved_scores(), observed_design=verify_observed_scores(), targets=verify_marginal_targets(channels), all_checks_passed=True)
    checks['input_hashes'] = {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT / 'analysis/r6_inference_audit.py', ROOT / 'analysis/r6_inference_verify.py', *sorted(R.glob('r6_inference_*.csv'))]}
    (A / 'r6_inference_independent_checks.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
    print(json.dumps(checks, indent=2))
