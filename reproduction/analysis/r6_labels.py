from r6_labels_separation import *
from measurement_sensitivity import prepare
import argparse, itertools, time
PREFIX = 'r6_labels'

def save(rows, name):
    pd.DataFrame(rows).to_csv(ROOT / 'results' / f'{PREFIX}_{name}.csv', index=False)

def validate():
    rng = np.random.default_rng(687125)
    c = np.repeat(np.arange(2), 6)
    n = len(c)
    Z = rng.normal(size=(n, 1))
    H = np.tile([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], 2)
    y = rng.poisson(4, size=n).astype(float) + 1
    null = ppml(Z, y, c)
    r = y - null['mu']
    positive = np.flatnonzero(H == 1)
    checked = []
    for bits in itertools.product([False, True], repeat=len(positive)):
        h = H.copy()
        h[positive[np.array(bits)]] = 0
        if np.all(h == 0):
            continue
        X = np.column_stack([Z, h])
        fit = ppml(X, y, c)
        score = float(h @ r)
        beta = float(fit['beta'][1])
        assert np.sign(score) == np.sign(beta)
        checked.append(dict(correction_class='positive_label_only', mask=''.join((str(int(b)) for b in bits)), null_score=score, beta=beta, sign_agrees=True, no_separation=poisson_separation_certificate(X, y, c)['no_additional_slope_separation_certified']))
    cc = np.repeat(np.arange(2), 4)
    zz = rng.normal(size=(8, 1))
    yy = rng.poisson(5, size=8).astype(float) + 1
    nn = ppml(zz, yy, cc)
    rr = yy - nn['mu']
    either_count = 0
    unidentified = 0
    for bits in itertools.product([0.0, 1.0], repeat=8):
        hh = np.array(bits)
        xx = np.column_stack([zz, hh])
        cert = poisson_separation_certificate(xx, yy, cc)
        if not cert['no_additional_slope_separation_certified']:
            unidentified += 1
            continue
        ff = ppml(xx, yy, cc)
        ss = float(hh @ rr)
        bb = float(ff['beta'][1])
        assert np.sign(ss) == np.sign(bb)
        checked.append(dict(correction_class='either_label_all_binary_candidates', mask=''.join((str(int(b)) for b in bits)), null_score=ss, beta=bb, sign_agrees=True, no_separation=True))
        either_count += 1
    sep = poisson_separation_certificate(np.array([[0.0], [1.0], [0.0], [1.0]]), np.array([1.0, 0.0, 2.0, 0.0]), np.array([0, 0, 1, 1]))
    assert not sep['no_additional_slope_separation_certified']
    save(checked, 'toy_exhaustive')
    out = dict(seed=687125, all_identified_positive_label_correction_subsets_checked=63, unidentified_all_zero_label_subset_excluded=1, all_null_score_signs_match_refitted_slopes=True, all_identified_binary_candidates_in_either_label_toy_checked=either_count, cell_constant_unidentified_candidates_excluded_in_either_label_toy=unidentified, all_finite_fits_have_full_positive_row_rank=all((r['no_separation'] for r in checked)), separating_toy_correctly_inconclusive=sep)
    (ROOT / 'audit' / f'{PREFIX}_validation.json').write_text(json.dumps(out, indent=2))
    return out

def sensitivity():
    started = time.time()
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    H = X[:, 1].copy()
    s, t, c = grouping(d, 'facebook', 'date')
    names = CHANNELS['facebook']
    Y = d[names].to_numpy(float)
    nuisance = np.delete(X, 1, axis=1)
    C = int(c.max() + 1)
    rank = C + X.shape[1]
    block = codes((t - t.min()) // 14)
    df = int(min(s.max(), block.max()))
    crit = stats.t.ppf(1 - 0.05 / 16, df)
    baseline = {}
    nulls = {}
    certs = []
    baseline_rows = []
    null_rows = []
    influence = []
    sharp = []
    boundary = []
    for j, ch in enumerate(names):
        y = Y[:, j]
        cert = poisson_separation_certificate(X, y, c)
        certs.append(dict(channel=ch, design='baseline', **cert))
        if not cert['no_additional_slope_separation_certified']:
            raise RuntimeError(f'Inconclusive baseline separation {ch}')
        f = ppml(X, y, c)
        baseline[ch] = f
        null = ppml(nuisance, y, c)
        r = y - null['mu']
        nulls[ch] = r
        q = f['bread'][:, 1]
        inf = (y - f['mu']) * (X @ q - (f['mean'] @ q)[c])
        influence.append(inf)
        baseline_rows.append(dict(channel=ch, beta=float(f['beta'][1]), RR=float(np.exp(f['beta'][1])), gradient=f['gradient'], condition=f['condition']))
        score = float(H @ r)
        sgn = np.sign(score)
        null_rows.append(dict(channel=ch, null_score=score, null_gradient=null['gradient'], null_condition=null['condition'], score_sign_matches_baseline=bool(sgn == np.sign(f['beta'][1])), null_separation_certified=poisson_separation_certificate(nuisance, y, c)['no_additional_slope_separation_certified']))
        assert sgn == np.sign(f['beta'][1])
        for mode in ['positive_label_corrections_only', 'either_label_corrections']:
            gain = sgn * (2 * H - 1) * r
            if mode == 'positive_label_corrections_only':
                gain = np.where(H == 1, gain, -np.inf)
            ix = np.flatnonzero(gain > 0)
            order = ix[np.argsort(-gain[ix], kind='stable')]
            cs = np.cumsum(gain[order])
            k = int(np.searchsorted(cs, abs(score), side='left') + 1)
            if k > len(order):
                sharp.append(dict(channel=ch, mode=mode, tips=False))
                continue
            br = []
            for kk in [k - 1, k]:
                XX = X.copy()
                XX[order[:kk], 1] = 1 - XX[order[:kk], 1]
                cc = poisson_separation_certificate(XX, y, c)
                certs.append(dict(channel=ch, design=mode, corrections=kk, **cc))
                if not cc['no_additional_slope_separation_certified']:
                    raise RuntimeError(f'Inconclusive boundary separation {ch} {mode} {kk}')
                ff = ppml(XX, y, c, start=f['beta'])
                ss = float(XX[:, 1] @ r)
                assert np.sign(ss) == np.sign(ff['beta'][1])
                br.append(float(ff['beta'][1]))
                boundary.append(dict(channel=ch, mode=mode, corrections=kk, beta=float(ff['beta'][1]), RR=float(np.exp(ff['beta'][1])), null_score=ss, gradient=ff['gradient'], condition=ff['condition'], no_additional_separation=True, finite_unique_slopes=True))
            assert sgn * br[0] > 0 and sgn * br[1] <= 0
            sharp.append(dict(channel=ch, mode=mode, tips=True, minimum_unit_cost_corrections=k, original_beta=float(f['beta'][1]), original_RR=float(np.exp(f['beta'][1])), null_score=score, score_one_fewer=float(score - sgn * cs[k - 2]) if k > 1 else score, score_after=float(score - sgn * cs[k - 1]), beta_one_fewer=br[0], beta_after=br[1], fraction_all_posts=k / len(H), positive_labels_corrected=int((H[order[:k]] == 1).sum()), negative_labels_corrected=int((H[order[:k]] == 0).sum()), scope='Sharp unit-cost point-sign threshold within declared arbitrary binary-label correction class, fixed outcomes/cells/nuisance; no empirical error-rate claim'))
        save(sharp, 'sharp_tipping')
        save(boundary, 'sharp_boundary_refits')
        save(certs, 'separation')
        print('baseline/null/sharp complete', ch, flush=True)
    save(baseline_rows, 'baseline')
    save(null_rows, 'null_fits')
    z, B, yr, ng = prepare(X, Y, c)
    j = names.index('Angry')
    y = Y[:, j]
    r = nulls['Angry']
    sgn = np.sign(float(H @ yr[:, j]))
    gain = np.where(H == 1, sgn * yr[:, j], -np.inf)
    ix = np.flatnonzero(gain > 0)
    order = ix[np.argsort(-gain[ix], kind='stable')]
    scores = float(H @ r) - np.cumsum(r[order])
    cross = np.flatnonzero(np.sign(float(H @ r)) * scores <= 0)
    first_point = int(cross[0] + 1) if len(cross) else None
    path = []
    cache = {}
    last = baseline['Angry']['beta'].copy()

    def pathfit(k):
        nonlocal last
        if k in cache:
            return cache[k]
        XX = X.copy()
        XX[order[:k], 1] = 0.0
        cert = poisson_separation_certificate(XX, y, c)
        if not cert['no_additional_slope_separation_certified']:
            raise RuntimeError(f'Inconclusive path separation {k}')
        ff = ppml(XX, y, c, start=last)
        last = ff['beta'].copy()
        beta = float(last[1])
        q = ff['bread'][:, 1]
        inf = (y - ff['mu']) * (XX @ q - (ff['mean'] @ q)[c])
        var = float(multi_cov(inf[:, None], s, block, rank)['two_way'][0, 0])
        se = np.sqrt(var) if var > 0 else np.nan
        low = beta - crit * se
        high = beta + crit * se
        row = dict(channel='Angry', path='OLS_count_Angry_positive_label_corrections', corrections=k, beta=beta, RR=float(np.exp(beta)), working_logRR_se=float(se), working_logRR_low=float(low), working_logRR_high=float(high), working_RR_low=float(np.exp(low)), working_RR_high=float(np.exp(high)), working_interval_excludes_one=bool(low > 0 or high < 0), working_variance=var, df=df, family=8, block_days=14, null_score=float(XX[:, 1] @ r), gradient=ff['gradient'], condition=ff['condition'], no_additional_separation=True)
        certs.append(dict(channel='Angry', design='OLS_selected_path', corrections=k, **cert))
        path.append(row)
        cache[k] = row
        save(path, 'OLS_path_PPML')
        save(certs, 'separation')
        return row
    loss = None
    for k in range(min(1001, len(order) + 1)):
        row = pathfit(k)
        if not row['working_interval_excludes_one']:
            loss = k
            break
        if k % 10 == 0:
            print('OLS path interval scan', k, flush=True)
    requested = set([0, 1, 38, 100, 203, 204, 205, 500, 1000])
    if first_point is not None:
        requested.update([first_point - 1, first_point, first_point + 1])
    for k in sorted(requested):
        if k <= len(order):
            pathfit(k)
    path.sort(key=lambda r: r['corrections'])
    save(path, 'OLS_path_PPML')
    summary = dict(OLS_selected_path_first_PPML_point_sign_crossing=first_point, OLS_selected_path_first_working_interval_sign_loss=loss, interval_scan_every_integer_through=loss if loss is not None else min(1000, len(order)), interval_interpretation='Path-specific working Bonferroni-eight t14 interval; known calibration failures prevent nominal confidence claim', null_score_identity='S(H)=H.T@(Y-mu_null); nuisance and cell null fit excludes H and does not change with candidate H', sharpness_argument='Concave profile likelihood, finite unique identified fitted slope; its sign equals null profile-score sign. Largest beneficial unit-cost score reductions minimize correction count within declared class. Verified finite fits on both boundary sides.', elapsed_seconds=time.time() - started, n=len(d), positive_labels=int(H.sum()), nuisance_columns=nuisance.shape[1])
    (ROOT / 'audit' / f'{PREFIX}_sensitivity.json').write_text(json.dumps(summary, indent=2))
    a = cell_ols(X, np.column_stack([Y, (Y > 0).astype(float)]), c)
    allinf = np.column_stack([a['obs'], np.column_stack(influence)])
    rms = np.sqrt(np.mean(allinf * allinf, axis=0))
    relative = np.abs(allinf) / np.where(rms > 0, rms, 1.0)
    measure = relative.max(1)
    driver = relative.argmax(1)
    np.savez_compressed(ROOT / 'results' / f'{PREFIX}_annotation_influence.npz', measure=measure, driver=driver)
    print(json.dumps(summary, indent=2), flush=True)

def annotation():
    raw = next(iter(pyreadr.read_r(str(ROOT / 'input/fb_proua_full_osf.RDS')).values()))
    dt = pd.to_datetime(raw.date)
    mask = (dt > pd.Timestamp('2022-02-24')) & (dt < pd.Timestamp('2022-09-13'))
    release_rows = np.flatnonzero(mask.to_numpy()) + 1
    d = raw.loc[mask].reset_index(drop=True)
    H = (d.hostility_pred != 'Other').astype(int).to_numpy()
    inf = np.load(ROOT / 'results' / f'{PREFIX}_annotation_influence.npz')
    measure = inf['measure']
    driver = inf['driver']
    sha = hashlib.sha256((ROOT / 'input/fb_proua_full_osf.RDS').read_bytes()).hexdigest()
    rowkeys = [hashlib.sha256(f'{sha}:RDS-row-1based:{r}'.encode()).hexdigest() for r in release_rows]
    samplehash = [hashlib.sha256(f'r6-annotation-design-687125:{r}'.encode()).hexdigest() for r in rowkeys]
    frame = pd.DataFrame(dict(release_row_1based=release_rows, analysis_row_1based=np.arange(1, len(d) + 1), release_row_sha256=rowkeys, sampling_sha256=samplehash, source=d['Page.Name'], date=d.date, observed_hostility=H, hostility_pred_proba=d.hostility_pred_proba, influence_measure=measure, influence_driver=driver))
    frame = frame.sort_values(['source', 'observed_hostility', 'influence_measure', 'release_row_sha256'])
    g = frame.groupby(['source', 'observed_hostility'], sort=True)
    pct = (g.cumcount() + 1) / g['influence_measure'].transform('size')
    frame['influence_band'] = np.where(pct > 0.99, 'top_1pct', np.where(pct > 0.9, 'next_9pct', 'bottom_90pct'))
    frame['stratum'] = frame.source.astype(str) + '|H' + frame.observed_hostility.astype(str) + '|' + frame.influence_band
    samples = []
    strata = []
    for key, group in frame.groupby('stratum', sort=True):
        N = len(group)
        n = min(3, N)
        sel = group.sort_values('sampling_sha256').head(n).copy()
        sel['stratum_population'] = N
        sel['stratum_sample'] = n
        sel['inclusion_probability'] = n / N
        sel['design_weight'] = N / n
        samples.append(sel)
        strata.append(dict(stratum=key, source=group.source.iloc[0], observed_hostility=int(group.observed_hostility.iloc[0]), influence_band=group.influence_band.iloc[0], population=N, sampled=n, inclusion_probability=n / N, design_weight=N / n))
    sample = pd.concat(samples).sort_values('sampling_sha256').reset_index(drop=True)
    sample['annotation_id'] = ['R6A-' + hashlib.sha256(('blind:' + k).encode()).hexdigest()[:16] for k in sample.release_row_sha256]
    sample['original_post_id'] = ''
    sample['raw_text'] = ''
    sample['text_linkage_status'] = 'not_available_in_public_release'
    sample['raw_text_sha256'] = ''
    sample['content_available_for_annotation'] = False
    sample.to_csv(ROOT / 'review_materials/r6_measurement_sampling_key.csv', index=False)
    pd.DataFrame(strata).to_csv(ROOT / 'review_materials/r6_measurement_sampling_strata.csv', index=False)
    blind = pd.DataFrame(dict(annotation_id=sample.annotation_id, raw_text='', hostility_label='', solidarity_label='', confidence='', ambiguous_or_unavailable='', codebook_rule='', annotator_notes=''))
    for person in ['A', 'B']:
        blind.to_csv(ROOT / f'review_materials/r6_measurement_blinded_annotator_{person}.csv', index=False)
    protocol = {'status': 'Sampling frame and empty annotation forms only; human validation has not occurred', 'release_file': 'input/fb_proua_full_osf.RDS', 'release_sha256': sha, 'release_rows': len(raw), 'analysis_rows': len(d), 'selected_rows': len(sample), 'represented_sources': int(sample.source.nunique()), 'eligible_sources': int(d['Page.Name'].nunique()), 'strata': len(strata), 'target_per_nonempty_stratum': 3, 'sampling': 'Within source x observed-hostility x within-source-label influence band (90/9/1 percent), take 3 smallest salted SHA256 ranks, or census if fewer than 3. Salt 687125 fixed before human annotation.', 'influence': 'Max absolute row influence divided by that coefficient influence RMS across 8 raw-count OLS, 8 occurrence OLS, and 8 PPML log-ratio coefficients in source-day full-release models; driver 0:7 count, 8:15 occurrence, 16:23 PPML in published channel order.', 'probabilities': 'Within every nonempty stratum, inclusion n_h/N_h under the hash pseudorandomization design; analysis weight N_h/n_h. Conditional on the fixed salt actual selection is deterministic. Report this reproducible sampling convention explicitly.', 'weight_sum': float(sample.design_weight.sum()), 'population_recovered': bool(np.isclose(sample.design_weight.sum(), len(d))), 'raw_text_available': False, 'original_post_ids_available': False, 'blinding': 'Annotation forms contain only random annotation_id and empty text/response fields. Keep key inaccessible to coders; populate text only after verified linkage by authorized data custodian. Do not expose labels, outcomes, influence, source or probability.', 'raw_text_requirement': 'No text is present in the public release. Exact original row-to-post-ID linkage and authorized raw post text must be obtained before these forms can be annotated. Never match on engagement alone, infer text, or substitute model outputs for human labels.', 'independence': 'Two Ukrainian/Russian fluent coders independently annotate every available sampled post after a separate training pilot. No discussion of study rows before both annotations are locked. Third adjudicator resolves disagreements, preserving original two labels.', 'rubric': 'Freeze the original supplementary codebook and translation rules before coding. Hostility means expressed hostility/derogation/dislike toward Russia or Russians; solidarity toward Ukraine/Ukrainians is coded separately. Preserve quoted speech, target, context ambiguity and unavailable cases explicitly.', 'analysis': 'Design-weighted joint confusion table, observed-positive correction rate and observed-negative correction rate overall and by influence band, source and calendar phase only where sample size permits; agreement before adjudication. Weighted sensitivity/specificity require a defensible adjudicated reference and nonresponse treatment. Report per-stratum effective n, missing text, agreement uncertainty and cluster dependence; no unweighted pooled accuracy claim from disproportionate sampling.', 'nonresponse': 'Do not silently replace unlinked or unavailable posts. Record reason; retain denominator, report linkage by stratum; report bounds or a declared nonresponse model. A second-stage sample requires its own probabilities.', 'limits': 'Three per source-label-influence stratum is an actionable audit sample, not a precision guarantee for all source-specific errors. Text retrieval may itself be selective. This is conditional validation of released labels and not a new outcome dataset.', 'original_validation': 'Original article DOI 10.1038/s41467-024-52179-8 reports 2000 labels by first author, 1600 training and 400 held-out testing with about .87 accuracy/.80 macro-F1; public host_val schema has no paired news NLI outputs or raw texts. Tuning notebook outputs cannot be reclassified as held-out validation.', 'row_reconciliation': 'Release-row SHA hashes identify rows in this exact release, not original platform IDs or the unknown 139 omitted cases. No reconciliation is claimed.'}
    (ROOT / 'audit' / f'{PREFIX}_annotation.json').write_text(json.dumps(protocol, indent=2))
    lines = ['# Independent, influence-stratified measurement audit (prepared, not completed)', '']
    for k, v in protocol.items():
        lines.extend([f"**{k.replace('_', ' ')}.** {v}", ''])
    lines.extend(['Original study: https://www.nature.com/articles/s41467-024-52179-8', '', 'All hostility/solidarity responses remain empty. The present public-data-only task cannot perform the human validation. No messages, data requests or deposits were sent.'])
    (ROOT / 'review_materials/r6_measurement_annotation_protocol.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: protocol[k] for k in ['selected_rows', 'represented_sources', 'strata', 'weight_sum', 'population_recovered']}, indent=2), flush=True)

def selections():
    raw = next(iter(pyreadr.read_r(str(ROOT / 'input/fb_proua_full_osf.RDS')).values()))
    dt = pd.to_datetime(raw.date)
    mask = (dt > pd.Timestamp('2022-02-24')) & (dt < pd.Timestamp('2022-09-13'))
    released = np.flatnonzero(mask.to_numpy()) + 1
    d = raw.loc[mask].reset_index(drop=True)
    X = ukrainian_design(d).to_numpy()
    H = X[:, 1]
    s, t, c = grouping(d, 'facebook', 'date')
    nuisance = np.delete(X, 1, axis=1)
    sha = hashlib.sha256((ROOT / 'input/fb_proua_full_osf.RDS').read_bytes()).hexdigest()
    rows = []
    thresholds = pd.read_csv(ROOT / 'results' / f'{PREFIX}_sharp_tipping.csv')
    for ch in CHANNELS['facebook']:
        y = d[ch].to_numpy(float)
        r = y - ppml(nuisance, y, c)['mu']
        score = float(H @ r)
        sgn = np.sign(score)
        for mode in ['positive_label_corrections_only', 'either_label_corrections']:
            gain = sgn * (2 * H - 1) * r
            if mode == 'positive_label_corrections_only':
                gain = np.where(H == 1, gain, -np.inf)
            ix = np.flatnonzero(gain > 0)
            order = ix[np.argsort(-gain[ix], kind='stable')]
            k = int(thresholds.loc[(thresholds.channel == ch) & (thresholds['mode'] == mode), 'minimum_unit_cost_corrections'].iloc[0])
            for pos, i in enumerate(order[:k], 1):
                rows.append(dict(channel=ch, path=mode, path_rank=pos, analysis_row_1based=int(i + 1), release_row_1based=int(released[i]), release_row_sha256=hashlib.sha256(f'{sha}:RDS-row-1based:{released[i]}'.encode()).hexdigest(), observed_hostility=int(H[i]), hypothetical_hostility=int(1 - H[i]), beneficial_null_score_reduction=float(gain[i])))
    Y = d[CHANNELS['facebook']].to_numpy(float)
    z, B, yr, ng = prepare(X, Y, c)
    gain = np.where(H == 1, np.sign(float(H @ yr[:, 2])) * yr[:, 2], -np.inf)
    ix = np.flatnonzero(gain > 0)
    order = ix[np.argsort(-gain[ix], kind='stable')]
    original = ix[np.argsort(-gain[ix])]
    assert np.array_equal(original[:1000], order[:1000])
    for pos, i in enumerate(order[:1000], 1):
        rows.append(dict(channel='Angry', path='OLS_count_Angry_positive_label_corrections', path_rank=pos, analysis_row_1based=int(i + 1), release_row_1based=int(released[i]), release_row_sha256=hashlib.sha256(f'{sha}:RDS-row-1based:{released[i]}'.encode()).hexdigest(), observed_hostility=1, hypothetical_hostility=0, OLS_numerator_reduction=float(gain[i])))
    save(rows, 'selected_corrections')
    (ROOT / 'audit' / f'{PREFIX}_selection_identity.json').write_text(json.dumps(dict(release_sha256=sha, selected_records=len(rows), unique_release_rows=len(set((r['release_row_1based'] for r in rows))), original_OLS_order_matches_first_1000=True, identity='SHA256 of release file digest + literal :RDS-row-1based: + original 1-based RDS row. Not a platform ID.'), indent=2))

def output_checks():
    cert = pd.read_csv(ROOT / 'results/r6_labels_separation.csv')
    path = pd.read_csv(ROOT / 'results/r6_labels_OLS_path_PPML.csv')
    sharp = pd.read_csv(ROOT / 'results/r6_labels_sharp_tipping.csv')
    key = pd.read_csv(ROOT / 'review_materials/r6_measurement_sampling_key.csv')
    a = pd.read_csv(ROOT / 'review_materials/r6_measurement_blinded_annotator_A.csv')
    b = pd.read_csv(ROOT / 'review_materials/r6_measurement_blinded_annotator_B.csv')
    sel = pd.read_csv(ROOT / 'results/r6_labels_selected_corrections.csv')
    assert cert.no_additional_slope_separation_certified.all() and len(sharp) == 16
    assert path.loc[path.corrections < 40, 'working_interval_excludes_one'].all()
    assert not path.loc[path.corrections == 40, 'working_interval_excludes_one'].item()
    assert key.annotation_id.is_unique and key.release_row_sha256.is_unique
    assert np.isclose(key.design_weight.sum(), 535936) and a.equals(b)
    assert a.drop(columns='annotation_id').isna().all().all() and key.raw_text.isna().all()
    assert not key.content_available_for_annotation.any() and len(sel) == 6925
    old = pd.read_csv(ROOT / 'results/cell_PPML.csv').query("platform=='facebook' and design=='date_all' and block==14").set_index('channel').RR
    new = pd.read_csv(ROOT / 'results/r6_labels_baseline.csv').set_index('channel').RR
    error = float(np.max(np.abs(new - old)))
    assert error < 1e-07
    out = dict(passed=True, separation_certificates=len(cert), PPML_sharp_thresholds=len(sharp), actual_OLS_path_refits=len(path), annotation_rows=len(key), annotation_sources=int(key.source.nunique()), empty_blinded_forms_verified=True, sampling_weights_recover_population=True, selected_correction_records=len(sel), baseline_matches_original_RR_max_absolute_error=error)
    (ROOT / 'audit/r6_labels_output_checks.json').write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['all', 'validation', 'sensitivity', 'annotation', 'selections', 'checks'], default='all')
    args = parser.parse_args()
    if args.phase in ['all', 'validation']:
        validate()
    if args.phase in ['all', 'sensitivity']:
        sensitivity()
    if args.phase in ['all', 'annotation']:
        annotation()
    if args.phase in ['all', 'selections']:
        selections()
    if args.phase in ['all', 'checks']:
        output_checks()
