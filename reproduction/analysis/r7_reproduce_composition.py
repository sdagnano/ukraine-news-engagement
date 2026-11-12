from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
sys.dont_write_bytecode = True
for variable in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[variable] = '1'
import numpy as np
import pandas as pd
import pyreadr
ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['Likes', 'Love', 'Angry', 'Haha', 'Sad', 'Wow', 'Comments', 'Shares']
NONLIKE = ['Love', 'Angry', 'Haha', 'Sad', 'Wow']
ATOL = 1e-10
SOURCE_FILES = ['input/fb_proua_full_osf.RDS', 'results/cell_OLS.csv', 'results/cell_PPML.csv', 'results/cell_support.csv', 'results/cell_joint.csv', 'results/scale_bridge.csv', 'results/cell_model_parameters.npz']

def codes(values):
    return pd.factorize(values, sort=True)[0].astype(int)

def aggregate(values, groups):
    values = np.asarray(values)
    size = int(groups.max() + 1)
    if values.ndim == 1:
        return np.bincount(groups, weights=values, minlength=size)
    return np.column_stack([np.bincount(groups, weights=values[:, j], minlength=size) for j in range(values.shape[1])])

def accounting(values):
    values = np.asarray(values, dtype=float)
    positive = float(values[values > 0].sum())
    negative = float(-values[values < 0].sum())
    gross = positive + negative
    return dict(positive_sum=positive, negative_magnitude=negative, net=positive - negative, gross_absolute_sum=gross, cancellation=1 - abs(positive - negative) / gross if gross else None, negative_to_positive=negative / positive if positive else None)

def single(frame, **filters):
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f'Expected one row for {filters}; found {len(selected)}')
    return selected.iloc[0]

def design(frame):
    result = pd.DataFrame(index=frame.index)
    result['solidarity'] = (frame.solidarity_pred != 'Other').astype(float)
    result['hostility'] = (frame.hostility_pred != 'Other').astype(float)
    result['ingroup_mention'] = frame.ingroup.gt(0).astype(float)
    result['outgroup_mention'] = frame.outgroup.gt(0).astype(float)
    for name in ['moral_emotional', 'positive', 'negative']:
        result[name + '_z'] = (frame[name] - frame[name].mean()) / frame[name].std(ddof=1)
    for name in ['has_media', 'has_URL']:
        values = frame[name].astype(str).str.lower().map({'true': 1.0, 'false': 0.0, '1': 1.0, '0': 0.0})
        if values.nunique() > 1:
            result[name] = values
    if result.isna().any().any():
        raise ValueError('Missing/invalid model covariates')
    return result

def checksum(relative_path):
    return hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()

def compute():
    saved_ols = pd.read_csv(ROOT / 'results/cell_OLS.csv')
    saved_ppml = pd.read_csv(ROOT / 'results/cell_PPML.csv')
    saved_support = pd.read_csv(ROOT / 'results/cell_support.csv')
    saved_joint = pd.read_csv(ROOT / 'results/cell_joint.csv')
    saved_bridge = pd.read_csv(ROOT / 'results/scale_bridge.csv')
    ols = saved_ols[(saved_ols.platform == 'facebook') & (saved_ols.design == 'date_all') & (saved_ols.target == 'count') & (saved_ols.block == 14)].set_index('channel').loc[CHANNELS]
    ppml = saved_ppml[(saved_ppml.platform == 'facebook') & (saved_ppml.design == 'date_all') & (saved_ppml.block == 14)].set_index('channel').loc[CHANNELS]
    if len(ols) != 8 or len(ppml) != 8:
        raise ValueError('Canonical design must contain exactly eight channels')
    frame = next(iter(pyreadr.read_r(str(ROOT / 'input/fb_proua_full_osf.RDS')).values()))
    dates = pd.to_datetime(frame.date)
    frame = frame.loc[(dates > pd.Timestamp('2022-02-24')) & (dates < pd.Timestamp('2022-09-13'))].reset_index(drop=True)
    Xframe = design(frame)
    X = Xframe.to_numpy(float)
    Y = frame[CHANNELS].to_numpy(float)
    if (Y < 0).any() or not np.isfinite(Y).all():
        raise ValueError('Invalid raw action counts')
    source = codes(frame['Page.Name'])
    date = codes(pd.to_datetime(frame.date))
    cell = codes(pd.MultiIndex.from_arrays([source, date]))
    n, sources, days, cells = (len(frame), int(source.max() + 1), int(date.max() + 1), int(cell.max() + 1))
    if (n, sources, days, cells) != (535936, 84, 200, 14689):
        raise ValueError(f'Unexpected canonical dimensions: {(n, sources, days, cells)}')
    if not ols.n.eq(n).all() or not ppml.n.eq(n).all():
        raise ValueError('Saved model rows do not use the canonical full sample')
    observed_mean = Y.mean(axis=0)
    observed_zeros = (Y == 0).sum(axis=0)
    observed_maximum = Y.max(axis=0)
    channels = pd.DataFrame(dict(channel=CHANNELS, OLS_count_beta=ols.beta.to_numpy(), PPML_mean0=ppml.mean0.to_numpy(), PPML_mean1=ppml.mean1.to_numpy(), PPML_delta=ppml.delta.to_numpy(), PPML_RR=ppml.RR.to_numpy(), observed_mean=observed_mean, observed_zeros=observed_zeros, observed_maximum=observed_maximum))
    channels['PPML_percent_change'] = 100 * (channels.PPML_RR - 1)
    channels['PPML_beta_H'] = np.log(channels.PPML_RR)
    channels['observed_zero_fraction'] = observed_zeros / n
    channels['maximum_over_observed_mean'] = observed_maximum / observed_mean
    channels['posts'], channels['sources'], channels['source_day_cells'] = (n, sources, cells)
    channels['design'] = 'source_day_full_release'
    population = dict(platform='Facebook', posts=n, sources=sources, dates=days, source_day_cells=cells)
    hostility = X[:, 1].astype(int)
    label_counts = np.zeros((cells, 2))
    np.add.at(label_counts, (cell, hostility), 1)
    both_labels = label_counts.min(axis=1) > 0
    supported = both_labels[cell]
    population.update(both_H_source_day_posts=int(supported.sum()), both_H_source_day_cells=int(both_labels.sum()), both_H_source_day_post_fraction=float(supported.mean()), both_H_subset_sources=int(np.unique(source[supported]).size))
    pa, oa = (accounting(ppml.delta), accounting(ols.beta))
    m0, m1 = (float(ppml.mean0.sum()), float(ppml.mean1.sum()))
    theta = float(ppml.loc['Shares', 'RR'] / ppml.loc['Likes', 'RR'])
    share_log = float(np.log(ppml.loc['Shares', 'RR']))
    like_log = float(-np.log(ppml.loc['Likes', 'RR']))
    log_theta = float(np.log(theta))
    ratio = dict(ratio_of_ratios=theta, log_ratio_of_ratios=log_theta, shares_log_component=share_log, likes_decline_log_component=like_log, shares_fraction_of_log_ratio=share_log / log_theta, likes_fraction_of_log_ratio=like_log / log_theta, ratio_of_standardized_means_H0=float(ppml.loc['Shares', 'mean0'] / ppml.loc['Likes', 'mean0']), ratio_of_standardized_means_H1=float(ppml.loc['Shares', 'mean1'] / ppml.loc['Likes', 'mean1']), Shares_RR=float(ppml.loc['Shares', 'RR']), Likes_RR=float(ppml.loc['Likes', 'RR']), estimand='(m_Shares,1/m_Likes,1)/(m_Shares,0/m_Likes,0); not E[Shares/Likes]', observed_Likes_zero_posts=int(observed_zeros[0]))
    nonlike = float(ols.loc[NONLIKE, 'beta'].sum())
    log_total = float(single(saved_bridge, platform='facebook', design='source_day_cell', target='log1p_total').beta)
    joint_net = float(single(saved_joint, platform='facebook', design='date_all', block=14).net)
    twitter_n = int(single(saved_support, platform='twitter', design='date_all').posts)
    cell_n = np.bincount(cell)
    centered_X = X - aggregate(X, cell)[cell] / cell_n[cell, None]
    focal = np.eye(X.shape[1])[:, 1]
    operator = centered_X @ np.linalg.solve(centered_X.T @ centered_X, focal)
    raw_beta = operator @ Y
    partition = np.zeros((4, 8))
    partition[0, 0] = partition[2, 6] = partition[3, 7] = 1
    partition[1, 1:6] = 1
    direct4 = operator @ (Y @ partition.T)
    indirect4 = partition @ raw_beta
    identities = dict(raw_posts=n, raw_sources=sources, raw_dates=days, raw_source_day_cells=cells, design_columns=list(Xframe.columns), focal_column='hostility', nuisance_columns=[name for name in Xframe.columns if name != 'hostility'], OLS_channel_saved_max_abs_error=float(np.max(abs(raw_beta - ols.beta.to_numpy()))), OLS_total_direct=float(operator @ Y.sum(axis=1)), OLS_total_sum_components=float(raw_beta.sum()), OLS_total_identity_abs_error=float(abs(operator @ Y.sum(axis=1) - raw_beta.sum())), OLS_coarsening_direct_max_abs_error=float(np.max(abs(direct4 - indirect4))), OLS_nonlike_direct=float(direct4[1]), log_total_saved_abs_error=float(abs(operator @ np.log1p(Y.sum(axis=1)) - log_total)), observed_total_equals_sum_components=bool(np.all(frame.engagement.to_numpy() == Y.sum(axis=1))), partition_nonnegative=bool((partition >= 0).all()), partition_column_sums=partition.sum(axis=0).tolist(), projected_four_channels=direct4.tolist())
    reconstructed = []
    with np.load(ROOT / 'results/cell_model_parameters.npz') as cached:
        for k, name in enumerate(CHANNELS):
            beta = cached['facebook_date_all_' + name]
            eta = X @ beta
            maximum = np.full(cells, -np.inf)
            np.maximum.at(maximum, cell, eta)
            exponential = np.exp(eta - maximum[cell])
            denominator = aggregate(exponential, cell)
            totals = aggregate(Y[:, k], cell)
            fitted = totals[cell] * exponential / denominator[cell]
            u0 = fitted * np.exp(-hostility * beta[1])
            u1 = u0 * np.exp(beta[1])
            reconstructed.append(dict(channel=name, mean0=float(u0.mean()), mean1=float(u1.mean()), max_abs_error_saved=max(abs(float(u0.mean()) - ppml.loc[name, 'mean0']), abs(float(u1.mean()) - ppml.loc[name, 'mean1']))))
    identities['PPML_saved_parameters_reconstruction'] = reconstructed
    identities['PPML_reconstructed_max_abs_error'] = max((row['max_abs_error_saved'] for row in reconstructed))
    for key in ['OLS_channel_saved_max_abs_error', 'OLS_total_identity_abs_error', 'OLS_coarsening_direct_max_abs_error', 'log_total_saved_abs_error', 'PPML_reconstructed_max_abs_error']:
        if not np.isfinite(identities[key]) or identities[key] > ATOL:
            raise ValueError(f'Independent reconstruction failed: {key}={identities[key]}')
    if not identities['observed_total_equals_sum_components']:
        raise ValueError('Released engagement total differs from the sum of raw channels')
    canonical = dict(population=population, OLS_count=dict(**oa, nonlike_five_reaction_sum=nonlike, source_day_log1p_total_beta=log_total), PPML_sum_of_separate_channel_models=dict(**pa, mean0=m0, mean1=m1, percent_change=100 * (m1 / m0 - 1), total_RR=m1 / m0, nonlike_five_reaction_delta_sum=float(ppml.loc[NONLIKE, 'delta'].sum()), mean0_weighted_channel_RR=float(ppml.mean0 @ ppml.RR / m0), saved_joint_net=joint_net, status='Sum of eight separately fitted nonnegative channel means; no PPML fit of total outcome'), relative_composition=ratio, coarsening=dict(canonical_nonlike_OLS=nonlike, partition_groups=[['Likes'], NONLIKE, ['Comments'], ['Shares']], original_eight_channel_cancellation=oa['cancellation'], four_channel_cancellation=accounting([ols.loc['Likes', 'beta'], nonlike, ols.loc['Comments', 'beta'], ols.loc['Shares', 'beta']])['cancellation']), platform_row_sum=dict(Facebook_posts=n, Twitter_posts=twitter_n, arithmetic_sum=n + twitter_n, status='Two separately analyzed platform rows, not a pooled cohort'), identity_scope='Accounting and parameter reconstruction only; no optimizer, new uncertainty calculation, or new population estimate.', identity_checks=identities, scientific_metadata=dict(producer='analysis/r7_reproduce_composition.py', absolute_tolerance=ATOL, input_paths_relative_to_project=True, specification='full-release source-day cell models', observed_descriptives='Computed independently from the released raw channel rows', source_files=[dict(path=path, sha256=checksum(path)) for path in SOURCE_FILES]))
    return (channels, canonical)

def compare(expected, actual, path='', rows=None):
    if rows is None:
        rows = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key == 'scientific_metadata':
                continue
            name = f'{path}.{key}' if path else key
            if key not in actual:
                rows.append(dict(field=name, passed=False, reason='missing saved field'))
            else:
                compare(value, actual[key], name, rows)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            rows.append(dict(field=path, passed=False, reason='different list length/type'))
        else:
            for index, (first, second) in enumerate(zip(expected, actual)):
                compare(first, second, f'{path}[{index}]', rows)
    elif isinstance(expected, (float, int, np.number)) and (not isinstance(expected, bool)):
        try:
            error = abs(float(expected) - float(actual))
            passed = bool(np.isfinite(error) and error <= ATOL)
        except (TypeError, ValueError):
            error, passed = (None, False)
        rows.append(dict(field=path, passed=passed, absolute_error=error))
    else:
        rows.append(dict(field=path, passed=bool(expected == actual)))
    return rows

def validate(channels, canonical):
    checks = []
    saved_channels = pd.read_csv(ROOT / 'results/r7_composition_channels.csv').set_index('channel')
    checks.append(dict(field='channels.exact_channel_set', passed=len(saved_channels) == len(CHANNELS) and set(saved_channels.index) == set(CHANNELS)))
    for _, row in channels.iterrows():
        if row.channel not in saved_channels.index:
            checks.append(dict(field=f'channels.{row.channel}', passed=False, reason='missing saved channel'))
            continue
        checks += compare(row.drop('channel').to_dict(), saved_channels.loc[row.channel].to_dict(), f'channels.{row.channel}')
    saved_root = json.loads((ROOT / 'results/r7_composition_canonical.json').read_text(encoding='utf-8'))
    checks += compare(canonical, saved_root, 'canonical')
    identities = canonical['identity_checks']
    for name in ['OLS_channel_saved_max_abs_error', 'OLS_total_identity_abs_error', 'OLS_coarsening_direct_max_abs_error', 'log_total_saved_abs_error', 'PPML_reconstructed_max_abs_error']:
        checks.append(dict(field='identity_tolerance.' + name, passed=identities[name] <= ATOL, absolute_error=identities[name]))
    checks.append(dict(field='identity.observed_total_equals_sum_components', passed=identities['observed_total_equals_sum_components']))
    p = canonical['population']
    support = single(pd.read_csv(ROOT / 'results/cell_support.csv'), platform='facebook', design='date_all')
    for key, column in [('posts', 'posts'), ('sources', 'sources'), ('source_day_cells', 'cells'), ('both_H_source_day_posts', 'both_label_posts'), ('both_H_source_day_cells', 'both_label_cells')]:
        checks.append(dict(field='raw_support_vs_saved.' + key, passed=p[key] == int(support[column])))
    pp = canonical['PPML_sum_of_separate_channel_models']
    checks.append(dict(field='identity.PPML_delta_sum_vs_saved_joint', passed=abs(pp['net'] - pp['saved_joint_net']) <= ATOL, absolute_error=abs(pp['net'] - pp['saved_joint_net'])))
    r = canonical['relative_composition']
    checks.append(dict(field='identity.log_ratio_components', passed=abs(r['shares_log_component'] + r['likes_decline_log_component'] - r['log_ratio_of_ratios']) <= ATOL, absolute_error=abs(r['shares_log_component'] + r['likes_decline_log_component'] - r['log_ratio_of_ratios'])))
    return dict(mode='check', passed=all((row['passed'] for row in checks)), absolute_tolerance=ATOL, comparison_count=len(checks), checks=checks, canonical_summary=dict(population=p, OLS_count=canonical['OLS_count'], PPML_sum_of_separate_channel_models=pp, relative_composition=r), raw_identity_checks=identities, scientific_metadata=canonical['scientific_metadata'])

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=lambda item: item.item() if isinstance(item, np.generic) else str(item)), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Check saved roots; default mode')
    mode.add_argument('--write', action='store_true', help='Regenerate the two scientific R7 root files')
    args = parser.parse_args()
    channels, canonical = compute()
    if args.write:
        channels.to_csv(ROOT / 'results/r7_composition_channels.csv', index=False)
        dump(ROOT / 'results/r7_composition_canonical.json', canonical)
        print('Regenerated the R7 scientific channel CSV and canonical JSON; no figure or model fit changed.')
    else:
        report = validate(channels, canonical)
        dump(ROOT / 'audit/r7_portable_composition_check.json', report)
        print(json.dumps(dict(passed=report['passed'], comparisons=report['comparison_count'], absolute_tolerance=ATOL, largest_absolute_error=max((row.get('absolute_error') or 0 for row in report['checks']))), indent=2))
        if not report['passed']:
            raise SystemExit(1)
if __name__ == '__main__':
    main()
