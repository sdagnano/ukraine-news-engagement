from cell_models import *
PREFIX = 'r6_margins'

def read(name):
    return pd.read_csv(ROOT / 'results' / f'{PREFIX}_{name}.csv')

def markdown(df, digits=6):

    def fmt(x):
        if isinstance(x, (int, np.integer)):
            return str(x)
        if isinstance(x, (float, np.floating)):
            return f'{x:.{digits}g}'
        return str(x)
    rows = ['| ' + ' | '.join(map(str, df.columns)) + ' |', '| ' + ' | '.join(['---'] * len(df.columns)) + ' |']
    return '\n'.join(rows + ['| ' + ' | '.join((fmt(x) for x in row)) + ' |' for row in df.itertuples(index=False, name=None)])

def main():
    support = read('support')
    two = read('adjusted_two_part')
    boundary = read('two_part_boundaries')
    identity = read('first_action_identity')
    weights = read('source_weight_summary')
    time = read('time_slopes')
    tails = read('tail_action_concentration')
    refits = read('support_refits')
    ja = json.loads((ROOT / 'audit' / f'{PREFIX}_two_part.json').read_text())
    jt = json.loads((ROOT / 'audit' / f'{PREFIX}_time.json').read_text())
    jr = json.loads((ROOT / 'audit' / f'{PREFIX}_support_refits.json').read_text())
    validation = json.loads((ROOT / 'audit' / f'{PREFIX}_validation.json').read_text())
    expected = {'support': 64, 'adjusted_two_part': 16, 'two_part_boundaries': 8, 'first_action_identity': 8, 'source_weight_summary': 4, 'time_slopes': 32, 'tail_action_concentration': 27, 'support_refits': 32}
    frames = locals().copy()
    actual = {'support': len(support), 'adjusted_two_part': len(two), 'two_part_boundaries': len(boundary), 'first_action_identity': len(identity), 'source_weight_summary': len(weights), 'time_slopes': len(time), 'tail_action_concentration': len(tails), 'support_refits': len(refits)}
    assert expected == actual
    numeric = [support, two, boundary, identity, weights, time, tails, refits]
    assert all((np.isfinite(z.select_dtypes(include='number').to_numpy()).all() for z in numeric))
    assert validation['passed'] and validation['full_dummy_optimizer_success']
    mean_identity = float(abs(two.adjusted_difference - two.occurrence_component - two.positive_count_component).max())
    means_difference = float(abs(two.adjusted_mean1 - two.adjusted_mean0 - two.adjusted_difference).max())
    max_post_identity = float(two.max_identity_error.max())
    assert max_post_identity < 1e-08 and mean_identity < 1e-10 and (means_difference < 1e-10)
    first_identity = float(abs(identity.identity_error).max())
    assert first_identity < 1e-10
    assert (boundary.included_posts + boundary.excluded_posts == 535936).all()
    assert (boundary.included_cells + boundary.excluded_cells == 14689).all()
    assert boundary.min_fitted_p.ge(0).all() and boundary.max_fitted_p.le(1).all() and boundary.min_fitted_positive_mean.ge(1).all()
    certificate = [f['positive_excess_ppml']['separation'] for f in ja['fits']]
    certificate += [f['separation'] for f in jt['fits']] + [f['separation'] for f in jr['fits']]
    assert len(certificate) == 48 and all((f['no_additional_slope_separation_certified'] for f in certificate))
    assert time.groupby('channel').n.sum().eq(535936).all()
    assert time.groupby('phase').n.nunique().eq(1).all()
    source_weight_error = float(abs(weights.identity_error).max())
    assert source_weight_error < 1e-12
    assert support.reference_count_weight_retained.between(0, 1).all()
    assert support.label1_count_weight_retained.between(0, 1).all()
    full = refits.query("support_rule=='full_release'").set_index('channel')
    old_ols = pd.read_csv(ROOT / 'results/cell_OLS.csv').query("platform=='facebook' and design=='date_all' and block==14")
    ols_error = max(float(abs(full.count_OLS - old_ols.query("target=='count'").set_index('channel').beta).max()), float(abs(full.any_OLS - old_ols.query("target=='any'").set_index('channel').beta).max()))
    old_ppml = pd.read_csv(ROOT / 'results/cell_PPML.csv').query("platform=='facebook' and design=='date_all' and block==14").set_index('channel')
    ppml_error = float(abs(full.PPML_RR - old_ppml.RR).max())
    assert ols_error < 1e-09 and ppml_error < 1e-09
    full_tails = tails.query('quantile==0.99 and channel!="Total"').set_index('channel')
    tail_coefficient_error = float(abs(full_tails.coefficient - full.count_OLS).max())
    assert tail_coefficient_error < 1e-09
    scores = [f['logit']['gradient'] for f in ja['fits']]
    excess_scores = [f['positive_excess_ppml']['gradient'] for f in ja['fits']]
    checks = dict(passed=True, table_rows=actual, max_post_symmetric_identity_error=max_post_identity, max_mean_symmetric_identity_error=mean_identity, max_model_means_difference_error=means_difference, max_first_action_linear_identity_error=first_identity, max_source_weight_identity_error=source_weight_error, max_full_release_OLS_reproduction_error=ols_error, max_full_release_PPML_RR_reproduction_error=ppml_error, max_tail_vs_full_OLS_coefficient_error=tail_coefficient_error, no_additional_Poisson_slope_separation_certificates=48, all_Poisson_slope_separation_certificates_passed=True, maximum_two_part_logit_normalized_slope_score=max(scores), maximum_two_part_excess_PPML_normalized_score=max(excess_scores), maximum_support_refit_PPML_normalized_score=float(refits.PPML_gradient.max()), logistic_validation=validation)
    (ROOT / 'audit' / f'{PREFIX}_checks.json').write_text(json.dumps(checks, indent=2))
    common = two.query("population=='common_positive_support'")
    refonly = two.query("population=='common_positive_and_binary_NN05'")
    t1 = tails.query('quantile==0.99')
    support_summary = support.groupby('support').agg(posts=('retained_posts', 'first'), post_fraction=('post_fraction', 'first'), minimum_reference_weight=('reference_count_weight_retained', 'min'), maximum_reference_weight=('reference_count_weight_retained', 'max')).reset_index()
    rr = refits.pivot(index='channel', columns='support_rule', values='PPML_RR').reindex(CHANNELS['facebook'])
    rr = rr[['full_release', 'same_binary_profile', 'same_binary_NN05', 'same_binary_NN1']].reset_index()
    report = "# R6 margins, support, temporal slopes, source weights, and action concentration\n\nCompleted from the public released numerical data. All additions are exploratory after the review; the original specifications and inputs were retained. Point summaries below concern the released automatic hostility label. They establish neither causal attribution nor calibrated population uncertainty.\n\n## Adjusted two-part target\n\nFor each channel separately, retain source-day cells containing at least one positive action count under each observed hostility label. Fit occurrence with a profiled source-day logistic mean p_h and fit positive excess Y−1 among Y>0 with profiled source-day PPML, so m_h=1+mu_excess,h. Both models adjust for the full released design: solidarity, H, two group mentions, three continuous language variables, media, and URL. The continuous-variable SD scaling is fixed from the full release.\n\nStandardize p_h*m_h over every retained post in that channel's support. The occurrence component averages (m_1+m_0)(p_1−p_0)/2; the positive-count component averages (p_1+p_0)(m_1−m_0)/2. They sum exactly to this separately defined joint model's standardized contrast. They do not decompose the full-release linear count coefficient or the single PPML coefficient. The two components are model descriptions rather than independently identified behavioral mechanisms. Support is outcome-selected and differs by channel.\n\n"
    report += markdown(common[['channel', 'n', 'fraction_full_posts', 'adjusted_mean0', 'adjusted_mean1', 'adjusted_difference', 'occurrence_component', 'positive_count_component']])
    report += '\n\nThe following NN≤0.5 rows restrict only the standardization/reference population of those same fitted models. They do not refit slopes or intercepts. The NN rule requires the same source-day and exact five-binary covariate profile plus an opposite-label post within Euclidean distance 0.5 in the three full-release SD-scaled continuous variables.\n\n'
    report += markdown(refonly[['channel', 'n', 'fraction_full_posts', 'adjusted_difference', 'occurrence_component', 'positive_count_component', 'fitted_reference_weight_retained_within_common']])
    report += '\n\n## Exclusions and fitted boundaries\n\nAll full-release cells number 14,689 and all posts number 535,936. Occurrence all-one cells use the fitted boundary p_h=1 for both toggles. All-zero positive-excess cells use fitted excess zero and hence m_h=1. These are likelihood boundaries in this model, not evidence of structural population constants. No occurrence all-zero cells remain because both H groups must contain positive outcomes. Every model retains all nine slope columns.\n\n'
    report += markdown(boundary[['channel', 'excluded_posts', 'excluded_cells', 'positive_fit_rows', 'occurrence_boundary_one_cells', 'occurrence_boundary_one_posts', 'positive_excess_boundary_zero_cells', 'positive_excess_boundary_zero_standardization_posts']])
    report += '\n\n## Joint measured-covariate support\n\nSame-binary strata match source-day, solidarity, two binary group mentions, media, and URL. Distances and bounding boxes refer to the three continuous z variables within these strata. A bounding box is not a convex hull. These outcome-independent descriptive rules are not positivity proofs; failing a rule is not proof of impossible support. Reference count weights come from full-release single-PPML toggle-H=0 predictions. Since that model has a constant H ratio, H=1 prediction-weight shares are identical. The share outside a declared region is one minus the retained reference weight shown below.\n\nThe observed-H propensity model is an in-sample source-day logit. All-H cells retain fitted 0/1 boundaries. Its bands are neither a validated assignment probability nor a causal adjustment criterion.\n\n'
    report += markdown(support_summary)
    report += '\n\n## Actual support-subset refits\n\nHere the full-count/occurrence linear projections and single PPML are actually refitted after each measured-X rule. This changes the estimation sample, source-day cells, coefficients, and standardization population. All nine controls remain identified. The same-binary, NN≤0.5, and NN≤1 samples contain respectively 395,086, 103,187, and 218,201 posts, covering 81, 78, and 79 sources. Full-release refits reproduce the previous estimates. Complete counts, occurrence coefficients, standardized means and absolute differences are in results/r6_margins_support_refits.csv.\n\n'
    report += markdown(rr)
    report += '\n\nThese subset refits are distinct from the two-part reference-only rows above. Pointwise changes across these targets cannot establish absence of extrapolation bias.\n\n## Outcome-independent calendar interactions\n\nFour consecutive 50-day calendar windows begin on 25 February 2022; endpoints are 15 April, 4 June, 24 July and 12 September. Window sizes are 158,904, 133,139, 123,661 and 120,232. Only H slopes interact with windows; nuisance slopes remain common and source-day intercepts remain absorbed. No outcome or event selected the windows. These descriptive refits show changing magnitudes; no calibrated interval or persistence claim is inferred.\n\n'
    report += markdown(time.query("channel in ['Angry','Haha','Shares']")[['channel', 'phase', 'count_OLS', 'any_OLS', 'PPML_RR', 'PPML_delta']])
    report += '\n\n## Source weights\n\nThe unchanged common finite-source population comprises 71 sources and 516,758 posts. Every source has its own H and nuisance slopes. These sources are selected through fit availability across the four outcomes. The aggregate standardized ratio equals the mean of source ratios weighted by n_s*mean0_s. Equal-source mean ratios, medians, and ratios of equal-source marginal means are different summaries; none is a population homogeneity check.\n\n'
    report += markdown(weights[['channel', 'aggregate_baseline_weighted_ratio', 'mean_source_ratio', 'median_source_ratio', 'equal_source_marginal_mean_ratio', 'fraction_sources_ratio_above1', 'top5_source_weight', 'effective_sources_inverse_HHI']])
    report += '\n\n## Top 1% action concentration versus coefficient contributions\n\nThe same full-release total-action 99th-percentile cutoff defines the tail for all channels. Coefficient terms use the unchanged full-design FWL operator, so tail plus remainder equals the original count OLS coefficient. The absolute-component fraction is |tail|/(|tail|+|remainder|), prioritizing stable magnitudes when signed contributions cancel. It is not a share of actions.\n\n'
    report += markdown(t1[['channel', 'tail_action_fraction', 'tail_coefficient_contribution', 'remainder_coefficient_contribution', 'tail_abs_component_fraction']])
    report += '\n\n## Computational checks and commands\n\nAll checks in audit/r6_margins_checks.json passed. The profiled logistic fit agrees with independent full-dummy SciPy trust-exact optimization on the fixed synthetic validation dataset. Independent Poisson positive-response within-cell rank certifies no additional slope separation for all eight positive-excess models, all eight calendar-interaction models, and all 32 support refits. Known all-zero-cell boundary intercepts remain explicit. A full-rank certificate is sufficient here because any Poisson separating slope must be constant on positive rows within a cell, which full positive-row within-cell rank excludes.\n\n'
    report += f"- Maximum synthetic full-dummy logistic coefficient difference: {validation['logit_vs_full_dummy_coefficient_max_error']:.12g}.\n- Maximum synthetic full-dummy logistic prediction difference: {validation['logit_vs_full_dummy_prediction_max_error']:.12g}.\n- Maximum empirical post-level symmetric identity error: {max_post_identity:.12g}.\n- Maximum empirical mean identity error: {mean_identity:.12g}.\n- Maximum first-action/excess linear accounting identity error: {first_identity:.12g}.\n- Maximum baseline-weighted source-ratio identity error: {source_weight_error:.12g}.\n- Maximum full-release OLS reproduction error: {ols_error:.12g}.\n- Maximum full-release single-PPML ratio reproduction error: {ppml_error:.12g}.\n\nThe first-action linear identity is an accounting identity and is not a conditional-positive-count behavioral margin. All outputs above are numerical point summaries; the prior coverage diagnostics prevent interpreting displayed working intervals as calibrated evidence for population signs or equivalence.\n\nRun with Python at python, UTF-8, and OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1:\n\n```text\npython -X utf8 analysis/r6_margins.py --part all\npython -X utf8 analysis/r6_margins_report.py\n```\n\nThe main script imports the independent certificate helper analysis/r6_labels_separation.py. Original released RDS data, original analyses, and core manuscript files were not edited by these scripts.\n"
    (ROOT / 'audit' / f'{PREFIX}_report.md').write_text(report, encoding='utf-8')
    print(json.dumps(checks, indent=2))
if __name__ == '__main__':
    main()
