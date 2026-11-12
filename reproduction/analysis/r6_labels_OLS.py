from cell_models import *
from measurement_sensitivity import prepare, fit_latent
import time

def run():
    started = time.time()
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    H = X[:, 1]
    names = CHANNELS['facebook']
    Y = d[names].to_numpy(float)
    YY = np.column_stack([Y, (Y > 0).astype(float)])
    s, t, c = grouping(d, 'facebook', 'date')
    z, B, yr, ng = prepare(X, YY, c)
    base = fit_latent(H, z, B, yr, c, ng)
    rows = []
    boundaries = []
    for j, ch in enumerate(names * 2):
        target = 'count' if j < 8 else 'any'
        num = float(H @ yr[:, j])
        sign = np.sign(num)
        for mode in ['positive_label_corrections_only', 'either_label_corrections']:
            gain = sign * (2 * H - 1) * yr[:, j]
            if mode == 'positive_label_corrections_only':
                gain = np.where(H == 1, gain, -np.inf)
            eligible = np.flatnonzero(gain > 0)
            order = eligible[np.argsort(-gain[eligible])]
            cs = np.cumsum(gain[order])
            k = int(np.searchsorted(cs, abs(num)) + 1)
            if k > len(order):
                rows.append(dict(channel=ch, target=target, mode=mode, tips=False))
                continue
            fits = []
            nums = []
            dens = []
            direct = []
            for kk in [k - 1, k]:
                h = H.copy()
                h[order[:kk]] = 1 - h[order[:kk]]
                hw = h - ag(h, c)[c] / ng[c]
                hr = hw - z @ (B @ (z.T @ hw))
                den = float(hr @ hr)
                assert den > 1e-08
                nnum = float(h @ yr[:, j])
                beta = float(nnum / den)
                XX = X.copy()
                XX[:, 1] = h
                full = cell_ols(XX, YY[:, j:j + 1], c)
                dbeta = float(full['beta'][1, 0])
                error = abs(dbeta - beta)
                assert error < 2e-08 and np.sign(dbeta) == np.sign(nnum)
                fits.append(beta)
                nums.append(nnum)
                dens.append(den)
                direct.append(dbeta)
                boundaries.append(dict(channel=ch, target=target, mode=mode, corrections=kk, fixed_null_numerator=nnum, hostility_residual_sum_squares=den, FWL_beta=beta, direct_cell_OLS_beta=dbeta, absolute_refit_error=error, full_cell_OLS_condition=float(np.linalg.cond(full['bread'])), identified=True))
            assert sign * fits[0] > 0 and sign * fits[1] <= 0
            rows.append(dict(channel=ch, target=target, mode=mode, tips=True, original_beta=float(base[j]), minimum_flips=k, minimum_unit_cost_corrections=k, fraction_all_posts=k / len(H), fraction_positive_labels=k / (H == 1).sum(), positive_labels_corrected=int((H[order[:k]] == 1).sum()), negative_labels_corrected=int((H[order[:k]] == 0).sum()), beta_after=fits[1], beta_one_fewer=fits[0], null_numerator=num, null_numerator_one_fewer=nums[0], null_numerator_after=nums[1], direct_cell_OLS_beta_one_fewer=direct[0], direct_cell_OLS_beta_after=direct[1], boundary_direct_refit_max_error=max(abs(direct[0] - fits[0]), abs(direct[1] - fits[1])), selected_analysis_rows_1based_ordered_SHA256=hashlib.sha256(','.join((str(i + 1) for i in order[:k])).encode()).hexdigest(), interpretation='Sharp unit-cost point-sign threshold within declared hypothetical correction class, fixed outcomes/cells/nuisance; no estimate of actual label errors or practical/interval threshold'))
        print('OLS all-channel boundaries', ch, target, flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(ROOT / 'results/r6_labels_OLS_tipping_all.csv', index=False)
    refs = pd.DataFrame(boundaries)
    refs.to_csv(ROOT / 'results/r6_labels_OLS_boundary_refits_all.csv', index=False)
    old = pd.read_csv(ROOT / 'results/adversarial_label_tipping.csv')
    keys = ['channel', 'target', 'mode']
    merged = old.merge(result, on=keys, suffixes=('_old', '_new'), validate='one_to_one')
    assert len(merged) == 16 and len(result) == 32
    check = []
    for _, r in merged.iterrows():
        assert r.minimum_flips_old == r.minimum_flips_new
        err = max(abs(r.beta_after_old - r.beta_after_new), abs(r.beta_one_fewer_old - r.beta_one_fewer_new), abs(r.original_beta_old - r.original_beta_new))
        assert err < 2e-08
        check.append(dict(channel=r.channel, target=r.target, mode=r['mode'], old_minimum=int(r.minimum_flips_old), new_minimum=int(r.minimum_flips_new), minimum_matches=True, beta_max_absolute_error=err))
    pd.DataFrame(check).to_csv(ROOT / 'results/r6_labels_OLS_previous_match.csv', index=False)
    audit = dict(passed=True, n=len(d), source_day_cells=int(c.max() + 1), covariate_columns=X.shape[1], threshold_rows=len(result), channels=names, targets=['count', 'any'], correction_classes=['positive_label_corrections_only', 'either_label_corrections'], direct_cell_OLS_boundary_refits=len(refs), all_boundaries_identified=bool(refs.identified.all()), direct_cell_OLS_refit_max_error=float(refs.absolute_refit_error.max()), matched_old_focal_rows=len(merged), all_old_focal_thresholds_identical=True, old_focal_beta_max_error=float(max((r['beta_max_absolute_error'] for r in check))), finite_denominator_minimum=float(refs.hostility_residual_sum_squares.min()), fixed_null_nuisance_argument='Residualize each outcome against source-day cells and all non-H covariates once. The numerator Hstar.T@Yres is linear in corrected Hstar; its sign determines OLS slope whenever Hstar residual sum of squares is positive. Ranking beneficial signed numerator reductions yields the minimum unit-cost point-sign crossing in each declared correction class.', limitations='Arbitrary outcome-informed hypothetical corrections; not actual annotation errors, realistic-error constraints, interval-sign thresholds or behavioral validity. No new raw texts or labels.', elapsed_seconds=time.time() - started)
    (ROOT / 'audit/r6_labels_OLS_checks.json').write_text(json.dumps(audit, indent=2))
    pivot = result.pivot(index='channel', columns=['target', 'mode'], values='minimum_flips').reindex(names)
    lines = ['# Complete OLS label-tipping table for all eight channels', '', 'The full-release source-day design and nuisance covariates are fixed. The numerator is H* times the outcome residual from the nuisance-only projection. Under an identified residualized H*, its sign equals the OLS slope sign. Sorting the largest beneficial, unit-cost numerator reductions gives the sharp point-sign threshold in the declared class. Every reported k−1 and k boundary was independently refitted with the full cell-OLS routine; all residualized-label denominators are positive.', '', '| Channel | Count, positive only | Count, either class | Occurrence, positive only | Occurrence, either class |', '|---|---:|---:|---:|---:|']
    for ch in names:
        v = pivot.loc[ch]
        lines.append(f"| {ch} | {int(v['count', 'positive_label_corrections_only'])} | {int(v['count', 'either_label_corrections'])} | {int(v['any', 'positive_label_corrections_only'])} | {int(v['any', 'either_label_corrections'])} |")
    lines.extend(['', f"All 16 rows from the older four-channel table reproduce exactly in correction count (maximum boundary/baseline coefficient difference {audit['old_focal_beta_max_error']:.3g}). All 64 full cell-OLS boundary refits agree with the fixed-null residual calculation (maximum absolute difference {audit['direct_cell_OLS_refit_max_error']:.3g}). The smallest residualized-label sum of squares is {audit['finite_denominator_minimum']:.6g}.", '', 'Files: results/r6_labels_OLS_tipping_all.csv (32 rows); results/r6_labels_OLS_boundary_refits_all.csv (64 actual refits); results/r6_labels_OLS_previous_match.csv (16 old-table comparisons); audit/r6_labels_OLS_checks.json. Exact ranking is reproducible from analysis/r6_labels_OLS.py; the table includes a SHA256 digest of each ordered selected analysis-row list.', '', 'These thresholds concern unconstrained, outcome-informed hypothetical label corrections within their declared class. They do not establish actual errors, misclassification rates, calibrated intervals, practical equivalence or superiority of occurrence outcomes. PPML has its own separately reported thresholds and refits.', '', 'Reproduce with bundled Python: python -X utf8 analysis/r6_labels_OLS.py. No manuscript sources, legacy files or PDFs were changed.'])
    (ROOT / 'audit/r6_labels_OLS_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(audit, indent=2), flush=True)
if __name__ == '__main__':
    run()
