from pathlib import Path, PurePosixPath, PureWindowsPath
import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd
from pypdf import PdfReader
ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['Likes', 'Love', 'Angry', 'Haha', 'Sad', 'Wow', 'Comments', 'Shares']
NONLIKE = ['Love', 'Angry', 'Haha', 'Sad', 'Wow']
CHECKS = []
SKIPPED = []
ATOL = 1e-10

def check(name, condition, details=None):
    row = dict(name=name, passed=bool(condition))
    if details is not None:
        row['details'] = details
    CHECKS.append(row)
    return row['passed']

def close(name, first, second, atol=ATOL, rtol=0):
    a, b = (np.asarray(first, dtype=float), np.asarray(second, dtype=float))
    same_shape = a.shape == b.shape
    error = float(np.max(np.abs(a - b))) if same_shape and a.size else 0.0 if same_shape else None
    return check(name, same_shape and np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False), dict(max_absolute_error=error, atol=atol, rtol=rtol))

def read(name):
    return pd.read_csv(ROOT / 'results' / name)

def load_json(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def project_path(relative):
    text = str(relative).replace('\\', '/')
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute() or '..' in PurePosixPath(text).parts:
        raise ValueError(f'Non-project-relative path: {text}')
    result = (ROOT / text).resolve()
    if not result.is_relative_to(ROOT.resolve()):
        raise ValueError(f'Path leaves project: {text}')
    return result

def verify_hash_map(mapping, prefix):
    for relative, expected in mapping.items():
        try:
            path = project_path(relative)
            check(prefix + '.' + str(relative), path.is_file() and digest(path) == expected)
        except (OSError, ValueError) as error:
            check(prefix + '.' + str(relative), False, str(error))

def row(frame, **filters):
    selected = frame
    for key, value in filters.items():
        selected = selected[selected[key] == value]
    if len(selected) != 1:
        raise ValueError(f'Expected one row for {filters}, found {len(selected)}')
    return selected.iloc[0]

def truths(series):
    return series.astype(str).str.lower().eq('true')

def accounting(values):
    x = np.asarray(values, dtype=float)
    p, n = (float(x[x > 0].sum()), float(-x[x < 0].sum()))
    return dict(positive_sum=p, negative_magnitude=n, net=p - n, gross_absolute_sum=p + n, cancellation=1 - abs(p - n) / (p + n), negative_to_positive=n / p)

def compare_tree(expected, actual, prefix):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                check(prefix + '.' + key, False, 'Missing field')
            else:
                compare_tree(value, actual[key], prefix + '.' + key)
    elif isinstance(expected, list):
        check(prefix + '.length', isinstance(actual, list) and len(expected) == len(actual))
        if isinstance(actual, list) and len(expected) == len(actual):
            for index, (a, b) in enumerate(zip(expected, actual)):
                compare_tree(a, b, prefix + f'[{index}]')
    elif isinstance(expected, (int, float)) and (not isinstance(expected, bool)):
        close(prefix, expected, actual)
    else:
        check(prefix, expected == actual)

def verify_composition():
    C = read('r7_composition_channels.csv').set_index('channel')
    J = load_json('results/r7_composition_canonical.json')
    P = read('cell_PPML.csv')
    P = P[(P.platform == 'facebook') & (P.design == 'date_all') & (P.block == 14)].set_index('channel').loc[CHANNELS]
    O = read('cell_OLS.csv')
    O = O[(O.platform == 'facebook') & (O.design == 'date_all') & (O.target == 'count') & (O.block == 14)].set_index('channel').loc[CHANNELS]
    check('composition.exact_eight_channels', len(C) == 8 and set(C.index) == set(CHANNELS))
    C = C.loc[CHANNELS]
    for target, source in [('OLS_count_beta', O.beta), ('PPML_mean0', P.mean0), ('PPML_mean1', P.mean1), ('PPML_delta', P.delta), ('PPML_RR', P.RR)]:
        close('composition.channels.' + target, C[target], source)
    close('composition.channel_delta_identity', C.PPML_mean1 - C.PPML_mean0, C.PPML_delta)
    close('composition.channel_RR_identity', C.PPML_mean1 / C.PPML_mean0, C.PPML_RR)
    close('composition.channel_log_RR', np.log(C.PPML_RR), C.PPML_beta_H)
    close('composition.channel_percent_change', 100 * (C.PPML_RR - 1), C.PPML_percent_change)
    close('composition.zero_fractions', C.observed_zeros / 535936, C.observed_zero_fraction)
    close('composition.maximum_over_mean', C.observed_maximum / C.observed_mean, C.maximum_over_observed_mean)
    check('composition.channel_population', C.posts.eq(535936).all() and C.sources.eq(84).all() and C.source_day_cells.eq(14689).all())
    for section, values in [('OLS_count', O.beta), ('PPML_sum_of_separate_channel_models', P.delta)]:
        compare_tree(accounting(values), J[section], 'composition.' + section)
    close('composition.OLS_nonlike', O.loc[NONLIKE, 'beta'].sum(), J['OLS_count']['nonlike_five_reaction_sum'])
    p = J['PPML_sum_of_separate_channel_models']
    m0, m1 = (float(P.mean0.sum()), float(P.mean1.sum()))
    for key, value in dict(mean0=m0, mean1=m1, percent_change=100 * (m1 / m0 - 1), total_RR=m1 / m0, nonlike_five_reaction_delta_sum=float(P.loc[NONLIKE, 'delta'].sum()), mean0_weighted_channel_RR=float(P.mean0 @ P.RR / m0)).items():
        close('composition.PPML_total.' + key, value, p[key])
    joint = row(read('cell_joint.csv'), platform='facebook', design='date_all', block=14)
    close('composition.saved_joint_net', p['net'], joint.net)
    q = J['relative_composition']
    share_log, like_log = (float(np.log(P.loc['Shares', 'RR'])), float(-np.log(P.loc['Likes', 'RR'])))
    theta = float(P.loc['Shares', 'RR'] / P.loc['Likes', 'RR'])
    log_theta = math.log(theta)
    expected = dict(ratio_of_ratios=theta, log_ratio_of_ratios=log_theta, shares_log_component=share_log, likes_decline_log_component=like_log, shares_fraction_of_log_ratio=share_log / log_theta, likes_fraction_of_log_ratio=like_log / log_theta, ratio_of_standardized_means_H0=float(P.loc['Shares', 'mean0'] / P.loc['Likes', 'mean0']), ratio_of_standardized_means_H1=float(P.loc['Shares', 'mean1'] / P.loc['Likes', 'mean1']), Shares_RR=float(P.loc['Shares', 'RR']), Likes_RR=float(P.loc['Likes', 'RR']), observed_Likes_zero_posts=int(C.loc['Likes', 'observed_zeros']))
    compare_tree(expected, q, 'composition.relative')
    close('composition.log_ratio_decomposition', share_log + like_log, log_theta)
    close('composition.saved_joint_ratio', theta, joint.ratio)
    bridge = row(read('scale_bridge.csv'), platform='facebook', design='source_day_cell', target='log1p_total')
    close('composition.log_total', J['OLS_count']['source_day_log1p_total_beta'], bridge.beta)
    receipt = load_json('audit/r7_portable_composition_check.json')
    check('composition.raw_reconstruction_receipt_passed', receipt['passed'] and all((r['passed'] for r in receipt['checks'])))
    check('composition.raw_reconstruction_tolerance', receipt['absolute_tolerance'] <= ATOL)
    compare_tree(receipt['canonical_summary'], J, 'composition.cached_raw_summary')
    compare_tree(receipt['raw_identity_checks'], J['identity_checks'], 'composition.cached_raw_identities')
    for entry in receipt['scientific_metadata']['source_files']:
        verify_hash_map({entry['path']: entry['sha256']}, 'composition.current_input_hash')
    for key in ['OLS_channel_saved_max_abs_error', 'OLS_total_identity_abs_error', 'OLS_coarsening_direct_max_abs_error', 'log_total_saved_abs_error', 'PPML_reconstructed_max_abs_error']:
        check('composition.raw_identity_tolerance.' + key, J['identity_checks'][key] <= ATOL)
    return J

def verify_r6_receipts():
    inference = load_json('audit/r6_inference_independent_checks.json')
    check('r6.independent_inference_receipt', inference['all_checks_passed'])
    verify_hash_map(inference['input_hashes'], 'r6.current_independent_input_hash')
    margins = load_json('audit/r6_margins_checks.json')
    check('r6.margins_receipt', margins['passed'] and margins['all_Poisson_slope_separation_certificates_passed'])
    check('r6.margins_independent_logistic_validation', margins['logistic_validation']['passed'])
    for key in ['max_post_symmetric_identity_error', 'max_mean_symmetric_identity_error', 'max_first_action_linear_identity_error', 'max_source_weight_identity_error', 'max_full_release_OLS_reproduction_error', 'max_full_release_PPML_RR_reproduction_error']:
        check('r6.margins_receipt.' + key, margins[key] < 1e-08)
    labels = load_json('audit/r6_labels_output_checks.json')
    check('r6.labels_output_receipt', labels['passed'] and labels['empty_blinded_forms_verified'] and labels['sampling_weights_recover_population'])
    close('r6.labels_receipt_row_counts', [labels['PPML_sharp_thresholds'], labels['actual_OLS_path_refits'], labels['annotation_rows'], labels['annotation_sources']], [16, 49, 1431, 84], atol=0)
    validation = load_json('audit/r6_labels_validation.json')
    check('r6.label_sharp_threshold_toy', validation['all_null_score_signs_match_refitted_slopes'] and validation['all_finite_fits_have_full_positive_row_rank'])
    close('r6.label_exhaustive_toy_counts', [validation['all_identified_positive_label_correction_subsets_checked'], validation['all_identified_binary_candidates_in_either_label_toy_checked']], [63, 252], atol=0)
    ols = load_json('audit/r6_labels_OLS_checks.json')
    check('r6.OLS_threshold_receipt', ols['passed'] and ols['all_boundaries_identified'] and ols['all_old_focal_thresholds_identical'])
    close('r6.OLS_threshold_receipt_counts', [ols['threshold_rows'], ols['direct_cell_OLS_boundary_refits']], [32, 64], atol=0)
    check('r6.OLS_threshold_refit_error', ols['direct_cell_OLS_refit_max_error'] < 1e-08 and ols['finite_denominator_minimum'] > 0)

def verify_support_and_margins(J):
    full = row(read('cell_support.csv'), platform='facebook', design='date_all')
    population = J['population']
    compare_tree(dict(posts=int(full.posts), sources=int(full.sources), source_day_cells=int(full.cells), both_H_source_day_posts=int(full.both_label_posts), both_H_source_day_cells=int(full.both_label_cells)), population, 'support.full')
    S = read('r6_margins_support.csv')
    check('support.eight_channels_per_rule', S.groupby('support').size().eq(8).all())
    close('support.post_fractions', S.post_fraction, S.retained_posts / 535936)
    close('support.reference_weight_H_invariance', S.reference_count_weight_retained, S.label1_count_weight_retained)
    for rule, n in [('both_H_source_day', 523246), ('both_H_same_binary_profile', 395086), ('same_binary_and_NN_distance_le_05', 103187), ('same_binary_and_NN_distance_le_1', 218201)]:
        selected = S[S.support == rule]
        check('support.population.' + rule, len(selected) == 8 and selected.retained_posts.eq(n).all())
    first = read('r6_margins_first_action_identity.csv')
    close('margins.first_action_identity', first.full_count_coefficient, first.first_action_coefficient + first.excess_beyond_first_coefficient)
    two = read('r6_margins_adjusted_two_part.csv')
    close('margins.adjusted_two_part_identity', two.adjusted_difference, two.occurrence_component + two.positive_count_component)
    close('margins.adjusted_mean_difference', two.adjusted_difference, two.adjusted_mean1 - two.adjusted_mean0)
    refits = read('r6_margins_support_refits.csv')
    check('support.refit_separation', truths(refits.no_additional_slope_separation_certified).all())
    close('support.refit_mean_difference', refits.PPML_delta, refits.PPML_mean1 - refits.PPML_mean0)
    C = read('r7_composition_channels.csv').set_index('channel').loc[CHANNELS]
    R = refits[refits.support_rule == 'full_release'].set_index('channel').loc[CHANNELS]
    close('support.full_refit_count', R.count_OLS, C.OLS_count_beta)
    close('support.full_refit_PPML_RR', R.PPML_RR, C.PPML_RR)
    close('support.full_refit_PPML_means', R[['PPML_mean0', 'PPML_mean1']], C[['PPML_mean0', 'PPML_mean1']])

def verify_thresholds():
    sharp = read('r6_labels_sharp_tipping.csv')
    boundaries = read('r6_labels_sharp_boundary_refits.csv')
    check('threshold.PPML_row_structure', len(sharp) == 16 and (not sharp.duplicated(['channel', 'mode']).any()) and (len(boundaries) == 32))
    check('threshold.PPML_finite_certificates', truths(boundaries.no_additional_separation).all() and truths(boundaries.finite_unique_slopes).all())
    for r in sharp.to_dict('records'):
        tag = 'threshold.PPML.' + r['channel'] + '.' + r['mode']
        k = int(r['minimum_unit_cost_corrections'])
        sign = np.sign(r['original_beta'])
        check(tag + '.crossing', k >= 1 and sign * r['beta_one_fewer'] > 0 and (sign * r['beta_after'] <= 0) and (sign * r['score_one_fewer'] > 0) and (sign * r['score_after'] <= 0))
        check(tag + '.correction_count', r['positive_labels_corrected'] + r['negative_labels_corrected'] == k)
        if r['mode'] == 'positive_label_corrections_only':
            check(tag + '.class', r['negative_labels_corrected'] == 0)
        for count, key in [(k - 1, 'beta_one_fewer'), (k, 'beta_after')]:
            fitted = row(boundaries, channel=r['channel'], mode=r['mode'], corrections=count)
            close(tag + '.refit_' + str(count), fitted.beta, r[key], atol=1e-08)
    baseline = read('r6_labels_baseline.csv').set_index('channel').loc[CHANNELS]
    C = read('r7_composition_channels.csv').set_index('channel').loc[CHANNELS]
    close('threshold.PPML_baseline_RR', baseline.RR, C.PPML_RR)
    close('threshold.PPML_refit_RR', np.exp(boundaries.beta), boundaries.RR)
    all_ols = read('r6_labels_OLS_tipping_all.csv')
    direct = read('r6_labels_OLS_boundary_refits_all.csv')
    check('threshold.OLS_row_structure', len(all_ols) == 32 and (not all_ols.duplicated(['channel', 'target', 'mode']).any()) and (len(direct) == 64))
    check('threshold.OLS_identified_boundaries', truths(direct.identified).all() and direct.hostility_residual_sum_squares.gt(0).all())
    close('threshold.OLS_FWL_identity', direct.fixed_null_numerator / direct.hostility_residual_sum_squares, direct.FWL_beta)
    close('threshold.OLS_independent_direct_refits', direct.FWL_beta, direct.direct_cell_OLS_beta, atol=1e-08)
    for r in all_ols.to_dict('records'):
        tag = 'threshold.OLS.' + r['channel'] + '.' + r['target'] + '.' + r['mode']
        k, sign = (int(r['minimum_flips']), np.sign(r['original_beta']))
        check(tag + '.crossing', sign * r['beta_one_fewer'] > 0 and sign * r['beta_after'] <= 0 and (sign * r['null_numerator_one_fewer'] > 0) and (sign * r['null_numerator_after'] <= 0))
        check(tag + '.correction_count', k == r['minimum_unit_cost_corrections'] == r['positive_labels_corrected'] + r['negative_labels_corrected'])
        for count, key in [(k - 1, 'beta_one_fewer'), (k, 'beta_after')]:
            fitted = row(direct, channel=r['channel'], target=r['target'], mode=r['mode'], corrections=count)
            close(tag + '.refit_' + str(count), fitted.direct_cell_OLS_beta, r[key], atol=1e-08)
    for target, expected in [('count', 204), ('any', 14297)]:
        actual = row(all_ols, channel='Angry', target=target, mode='positive_label_corrections_only')
        close('threshold.Angry_primary_' + target, actual.minimum_flips, expected, atol=0)
    path = read('r6_labels_OLS_path_PPML.csv')
    check('threshold.OLS_path_working_interval', truths(path.loc[path.corrections < 40, 'working_interval_excludes_one']).all() and (not truths(path.loc[path.corrections == 40, 'working_interval_excludes_one']).any()))

def verify_calibration():
    summary = read('r6_inference_operational_summary.csv')
    for filename in ['r6_inference_full_source_and_block_slopes_trials.csv', 'r6_inference_subsample_independent_NB_trials.csv']:
        trials = read(filename)
        check('calibration.unique_trials.' + filename, not trials.duplicated(['replicate', 'method']).any())
        for method, selected in trials.groupby('method', sort=False):
            design = selected.design.iloc[0]
            saved = row(summary, design=design, method=method)
            n = len(selected)
            valid = int(truths(selected.valid).sum())
            joint = int(truths(selected.joint_noncoverage).sum())
            failure = int(truths(selected.operational_failure).sum())
            close('calibration.summary.' + design + '.' + method, [n, valid, n - valid, joint, failure], [saved.replicates, saved.valid, saved.invalid, saved.joint_noncoverage_count, saved.operational_failure_count], atol=0)
            check('calibration.failure_union.' + design + '.' + method, np.array_equal(truths(selected.operational_failure), ~truths(selected.valid) | truths(selected.joint_noncoverage)))
    full = row(summary, design='full', method='PPML')
    sub = row(summary, design='subsample', method='PPML')
    close('calibration.primary_30_of_100', [full.joint_noncoverage_count, full.replicates], [30, 100], atol=0)
    close('calibration.primary_166_of_500', [sub.operational_failure_count, sub.replicates], [166, 500], atol=0)
    check('calibration.full_and_subsample_targets_distinct', full.scenario == 'source_and_block_slopes' and sub.scenario == 'independent_NB')

def normalize(text):
    text = unicodedata.normalize('NFKD', text).lower().replace('’', "'")
    return re.sub('\\s+', ' ', text)

def verify_documents():
    names = ['main', 'main_anonymous', 'supplementary', 'title_page']
    source_present = any(((ROOT / (name + '.tex')).exists() for name in names))
    pdf_present = any(((ROOT / (name + '.pdf')).exists() for name in names))
    if not source_present and (not pdf_present):
        SKIPPED.append('PDFs and compilation: analysis-only package; no manuscript source or submission PDFs are present')
        return
    if source_present:
        for name in names:
            check('source.complete_document_set.' + name, (ROOT / (name + '.tex')).is_file())
    macro_source = (ROOT / 'author_metadata.tex').read_text(encoding='utf-8') if (ROOT / 'author_metadata.tex').exists() else ''
    macros = dict(re.findall('\\\\newcommand\\{\\\\(Author\\w+)\\}\\{([^{}]*)\\}', macro_source))
    identifiers = [macros.get(key, '') for key in ['AuthorName', 'AuthorEmail', 'AuthorORCID']]
    if macros.get('AuthorName', '').strip():
        identifiers.append(macros['AuthorName'].strip().split()[-1])
    if (ROOT / 'main.pdf').exists():
        author = str((PdfReader(ROOT / 'main.pdf').metadata or {}).get('/Author', '')).strip()
        if author:
            identifiers.append(author)
    identifiers = sorted(set((normalize(value) for value in identifiers if value.strip())))
    receipt = load_json('audit/r7_compilation.json') if (ROOT / 'audit/r7_compilation.json').exists() else None
    for name in names:
        path = ROOT / (name + '.pdf')
        if not check('PDF.exists.' + name, path.is_file()):
            continue
        pdf = PdfReader(path)
        text = '\n'.join((page.extract_text() or '' for page in pdf.pages))
        metadata = dict(pdf.metadata or {})
        check('PDF.nonempty.' + name, len(pdf.pages) > 0 and len(text.strip()) > 50)
        check('PDF.resolved_references.' + name, '??' not in text and (not re.search('\\[\\s*\\?\\s*\\]', text)))
        check('PDF.title_metadata.' + name, bool(str(metadata.get('/Title', '')).strip()))
        if name in ['main_anonymous', 'supplementary']:
            check('PDF.anonymous_author_metadata.' + name, not str(metadata.get('/Author', '')).strip())
            all_content = normalize(text + ' ' + json.dumps(metadata, default=str))
            leaks = [identifier for identifier in identifiers if identifier in all_content]
            check('PDF.anonymous_identifiers.' + name, not leaks, dict(identifier_leaks=leaks))
        else:
            check('PDF.named_author_metadata.' + name, bool(str(metadata.get('/Author', '')).strip()))
        if receipt:
            document = receipt.get('documents', {}).get(name)
            check('compile.document_receipt.' + name, document is not None)
            if document:
                check('compile.PDF_digest.' + name, digest(path) == document['PDF_sha256'])
                check('compile.exit_codes.' + name, all((step['exit_code'] == 0 for step in document['steps'])))
        log = ROOT / 'audit' / ('r7_' + name + '_compilation.log')
        if log.exists():
            content = log.read_text(encoding='utf-8', errors='replace')
            check('compile.no_undefined_references.' + name, not re.search('(?:undefined references|Citation .+ undefined|Reference .+ undefined)', content, re.I))
    if receipt and (ROOT / 'main.tex').exists():
        verify_hash_map(receipt['input_sha256'], 'compile.current_input_hash')
    elif (ROOT / 'main.tex').exists():
        check('compile.receipt_required_for_source_and_PDF', False)

def verify_manifest(skip):
    if skip:
        SKIPPED.append('Current submission manifest: --skip-manifest')
        return
    path = ROOT / 'MANIFEST_SUBMISSION_SHA256.json'
    if not path.exists():
        SKIPPED.append('Current submission manifest: not present; historical manifests are not requirements')
        return
    raw = json.loads(path.read_text(encoding='utf-8'))
    mapping = raw.get('sha256', raw)
    if not check('manifest.path_digest_mapping', isinstance(mapping, dict) and bool(mapping) and all((isinstance(v, str) and re.fullmatch('[0-9a-fA-F]{64}', v) for v in mapping.values()))):
        return
    verify_hash_map(mapping, 'manifest.current_file_hash')

def guarded(label, function, *args):
    try:
        return function(*args)
    except Exception as error:
        check(label + '.completed', False, f'{type(error).__name__}: {error}')
        return None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-receipt', action='store_true')
    parser.add_argument('--analysis-only', action='store_true')
    parser.add_argument('--skip-manifest', action='store_true')
    args = parser.parse_args()
    canonical = guarded('composition', verify_composition)
    guarded('independent_receipts', verify_r6_receipts)
    if canonical is not None:
        guarded('support_margins', verify_support_and_margins, canonical)
    guarded('correction_thresholds', verify_thresholds)
    guarded('calibration', verify_calibration)
    automatic_analysis_only = not any(((ROOT / (name + extension)).exists() for name in ['main', 'main_anonymous', 'supplementary', 'title_page'] for extension in ['.tex', '.pdf']))
    if args.analysis_only:
        SKIPPED.append('PDFs and compilation: --analysis-only')
    else:
        guarded('documents', verify_documents)
    guarded('manifest', verify_manifest, args.skip_manifest)
    failed = [entry for entry in CHECKS if not entry['passed']]
    report = dict(passed=not failed, checks_passed=len(CHECKS) - len(failed), checks_total=len(CHECKS), mode='analysis_only' if args.analysis_only else 'analysis_only_auto' if automatic_analysis_only else 'scientific_results_and_submission_documents', default_is_read_only=True, skipped=SKIPPED, checks=CHECKS, dependencies=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, pypdf=__import__('pypdf').__version__), limitations='Verifies saved numerical identities, current-input fingerprints and prior independent reconstruction receipts; does not repeat fits, simulate outcomes, recalibrate coverage or validate labels.')
    if args.write_receipt:
        (ROOT / 'audit/r7_submission_verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['passed', 'checks_passed', 'checks_total', 'skipped', 'dependencies']}, indent=2))
    if failed:
        print(json.dumps(dict(failed_checks=failed), indent=2))
        raise SystemExit(1)
if __name__ == '__main__':
    main()
