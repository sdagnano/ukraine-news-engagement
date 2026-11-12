from cell_models import *
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
BLUE = HexColor('#205B8F')
ORANGE = HexColor('#B85423')
GRAY = HexColor('#66717B')
LIGHT = HexColor('#E0E5E9')

def fmt(x, n=2):
    return f'{float(x):.{n}f}'

def ci(a, b, n=2):
    return '[' + fmt(a, n) + ', ' + fmt(b, n) + ']'

def tab(name, header, rows, align):
    out = '\\begin{tabular}{' + align + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n'
    out += '\n'.join((' & '.join(map(str, row)) + ' \\\\' for row in rows))
    out += '\n\\bottomrule\n\\end{tabular}\n'
    (ROOT / 'tables' / ('r6_' + name + '.tex')).write_text(out, encoding='utf-8')

def label(c, x, y, s, size=8, color=GRAY):
    c.setFillColor(color)
    c.setFont('Helvetica', size)
    c.drawString(x, y, str(s))

def line(c, x, y, xx, yy, color=LIGHT, width=0.6):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x, y, xx, yy)

def dot(c, x, y, color):
    c.setFillColor(color)
    c.circle(x, y, 2.5, stroke=0, fill=1)

def base_assets():
    O = pd.read_csv(ROOT / 'results/cell_OLS.csv')
    P = pd.read_csv(ROOT / 'results/cell_PPML.csv')
    a = O.query("platform=='facebook' and design=='date_all' and target=='any' and block==14").set_index('channel')
    b = O.query("platform=='facebook' and design=='date_all' and target=='count' and block==14").set_index('channel')
    p = P.query("platform=='facebook' and design=='date_all' and block==14").set_index('channel')
    rows = []
    for ch in CHANNELS['facebook']:
        rows.append([ch, fmt(a.loc[ch, 'beta'] * 100), fmt(b.loc[ch, 'beta']), fmt(p.loc[ch, 'RR'], 3), fmt(p.loc[ch, 'delta'])])
    tab('primary', 'Action & Occurrence (pp) & Count OLS & PPML ratio & PPML difference', rows, 'lrrrr')
    c = canvas.Canvas(str(ROOT / 'figures/r6_common_design.pdf'), pagesize=(530, 300))
    c.setTitle('Same-design linear projections of occurrence and action counts')
    x1 = lambda z: 91 + (z + 8) / 22 * 162
    x2 = lambda z: 335 + (z + 105) / 125 * 157
    for v in [-5, 0, 5, 10]:
        line(c, x1(v), 42, x1(v), 255)
        label(c, x1(v) - 6, 27, v)
    for v in [-100, -50, 0]:
        line(c, x2(v), 42, x2(v), 255)
        label(c, x2(v) - 9, 27, v)
    line(c, x1(0), 42, x1(0), 255, GRAY, 1)
    line(c, x2(0), 42, x2(0), 255, GRAY, 1)
    for j, ch in enumerate(CHANNELS['facebook']):
        y = 245 - j * 27
        av = a.loc[ch, 'beta'] * 100
        bv = b.loc[ch, 'beta']
        label(c, 7, y - 3, ch, 9)
        dot(c, x1(av), y, BLUE)
        dot(c, x2(bv), y, ORANGE)
        label(c, 263, y - 3, fmt(av), 8, BLUE)
        label(c, 499, y - 3, fmt(bv), 7.5, ORANGE)
    label(c, 88, 278, 'Occurrence: percentage points', 9, BLUE)
    label(c, 334, 278, 'Volume: actions per post', 9, ORANGE)
    label(c, 88, 7, 'Identical posts, covariates and source-day linear projection. Descriptive point estimates.', 8)
    c.showPage()
    c.save()
    a = O.query("platform=='twitter' and design=='date_all' and target=='any' and block==14").set_index('channel')
    b = O.query("platform=='twitter' and design=='date_all' and target=='count' and block==14").set_index('channel')
    p = P.query("platform=='twitter' and design=='date_all' and block==14").set_index('channel')
    tab('twitter', 'Action & Occurrence (pp) & Count OLS & PPML ratio & PPML difference', [[{'favorite_count': 'Favorites', 'retweet_count': 'Retweets'}.get(ch, ch), fmt(a.loc[ch, 'beta'] * 100), fmt(b.loc[ch, 'beta']), fmt(p.loc[ch, 'RR'], 3), fmt(p.loc[ch, 'delta'])] for ch in CHANNELS['twitter']], 'lrrrr')
    rows = []
    p = P.query("platform=='facebook' and design=='date_all'")
    for ch in CHANNELS['facebook']:
        v = p[p.channel == ch].set_index('block')
        rows.append([ch] + [ci(np.exp(v.loc[w, 'logRR_low']), np.exp(v.loc[w, 'logRR_high']), 3) for w in [1, 7, 14, 28]])
    tab('working_intervals', 'Action & Day & 7 days & 14 days & 28 days', rows, 'lrrrr')
    h = pd.read_csv(ROOT / 'results/adversarial_label_tipping.csv')
    rows = []
    for ch in CHANNELS['facebook']:
        z = h[h.channel == ch]
        vals = []
        for mode in ['positive_label_corrections_only', 'either_label_corrections']:
            q = z[z['mode'] == mode].set_index('target')
            if len(q):
                vals += [f"{int(q.loc[k, 'minimum_flips']):,}" for k in ['count', 'any']]
        if len(vals) == 4:
            rows.append([ch] + vals)
    if not rows:
        modes = list(h['mode'].drop_duplicates())
        assert len(modes) == 2, modes
        for ch in CHANNELS['facebook']:
            vals = []
            for mode in modes:
                q = h[(h.channel == ch) & (h['mode'] == mode)].set_index('target')
                vals += [f"{int(q.loc[k, 'minimum_flips']):,}" for k in ['count', 'any']]
            rows.append([ch] + vals)
    tab('OLS_tipping', 'Action & Positive: count & Positive: any & Either: count & Either: any', rows, 'lrrrr')

def inference_assets():
    root = ROOT / 'results'
    p = root / 'r6_inference_channel_summary.csv'
    if not p.exists():
        return
    z = pd.read_csv(p)
    rows = []
    for ch in CHANNELS['facebook']:
        q = z.query("design=='full' and method=='PPML' and channel==@ch and conditioning=='channel_valid'").set_index('interval')
        v = q.loc['pointwise95']
        r = q.loc['Bonferroni8']
        rows.append([ch, fmt(v.bias, 3), fmt(v.RMSE, 3), fmt(v.empirical_SD, 3), fmt(v.mean_SE, 3), fmt(v.noncoverage * 100, 0) + '\\%', fmt(r.noncoverage * 100, 0) + '\\%'])
    tab('channel_calibration', 'Action & Bias & RMSE & Empirical SD & Mean SE & Pointwise miss & Family-bar miss', rows, 'lrrrrrr')
    z = pd.read_csv(root / 'r6_inference_operational_summary.csv')
    rows = []
    for _, v in z.iterrows():
        rows.append(['Full' if v.design == 'full' else 'Sparse', 'PPML' if v.method == 'PPML' else 'Occurrence OLS', int(v.replicates), int(v.invalid), int(v.joint_noncoverage_count), fmt(v.operational_failure_rate * 100, 1) + '\\%'])
    tab('operational', 'Design & Procedure & Draws & Invalid & Valid but miss & Invalid or miss', rows, 'llrrrr')
    z = pd.read_csv(root / 'r6_inference_block_origin_summary.csv')
    rows = []
    for method in ['count_OLS', 'any_OLS', 'PPML']:
        for ch in ['Angry', 'Sad', 'Shares']:
            for width in [7, 14, 28]:
                v = z.query('method==@method and channel==@ch and width==@width').iloc[0]
                rows.append([{'count_OLS': 'Count OLS', 'any_OLS': 'Occurrence OLS', 'PPML': 'PPML'}[method], ch, width, int(v.origins), fmt(v.min_SE, 3) + '--' + fmt(v.max_SE, 3), int(v.positive_intervals), int(v.negative_intervals)])
    tab('block_origins', 'Procedure & Action & Days & Origins & SE range & Positive & Negative', rows, 'llrrrrr')

def margin_assets():
    rr = ROOT / 'results'
    S = pd.read_csv(rr / 'r6_margins_support.csv')
    T = pd.read_csv(rr / 'r6_margins_adjusted_two_part.csv')
    rows = []
    for rule in S.support.drop_duplicates():
        z = S[S.support == rule].set_index('channel')
        v = z.iloc[0]
        names = {'both_H_source_day': 'Both labels in source-day', 'both_H_same_binary_profile': 'Same five binary covariates', 'opposite_H_continuous_bounding_box': 'Opposite-label bounding box', 'same_binary_and_NN_distance_le_025': 'Same binary; distance $\\leq0.25$', 'same_binary_and_NN_distance_le_05': 'Same binary; distance $\\leq0.5$', 'same_binary_and_NN_distance_le_1': 'Same binary; distance $\\leq1$', 'fitted_H_propensity_005_095': 'Fitted propensity 0.05--0.95', 'fitted_H_propensity_01_09': 'Fitted propensity 0.1--0.9'}
        rows.append([names.get(rule, rule), f'{int(v.retained_posts):,}', fmt(v.post_fraction * 100, 2) + '\\%', fmt(z.loc['Likes', 'reference_count_weight_retained'] * 100, 2) + '\\%', fmt(z.loc['Angry', 'reference_count_weight_retained'] * 100, 2) + '\\%'])
    tab('support', 'Rule & Posts & Post share & Likes baseline mass & Angry baseline mass', rows, 'lrrrr')
    rows = []
    for ch in CHANNELS['facebook']:
        v = T.query("channel==@ch and population=='common_positive_support'").iloc[0]
        rows.append([ch, f'{int(v.n):,}', fmt(v.adjusted_mean0), fmt(v.adjusted_mean1), fmt(v.occurrence_component), fmt(v.positive_count_component), fmt(v.adjusted_difference)])
    tab('two_part', 'Action & Posts & $\\bar\\mu_0$ & $\\bar\\mu_1$ & Occurrence term & Positive-count term & Sum', rows, 'lrrrrrr')
    rows = []
    for ch in CHANNELS['facebook']:
        v = T.query("channel==@ch and population=='common_positive_and_binary_NN05'").iloc[0]
        rows.append([ch, f'{int(v.n):,}', fmt(v.occurrence_component), fmt(v.positive_count_component), fmt(v.adjusted_difference)])
    tab('two_part_restricted', 'Action & Reference posts & Occurrence term & Positive-count term & Sum', rows, 'lrrrr')
    B = pd.read_csv(rr / 'r6_margins_two_part_boundaries.csv')
    tab('two_part_boundaries', 'Action & Excluded posts & Occurrence-one cells & Occurrence-one posts & Zero-excess cells', [[v.channel, f'{int(v.excluded_posts):,}', int(v.occurrence_boundary_one_cells), f'{int(v.occurrence_boundary_one_posts):,}', int(v.positive_excess_boundary_zero_cells)] for _, v in B.iterrows()], 'lrrrr')
    tails = pd.read_csv(rr / 'r6_margins_tail_action_concentration.csv').query('quantile==0.99')
    rows = []
    for _, v in tails.iterrows():
        rows.append([v.channel, fmt(v.tail_action_fraction * 100, 1) + '\\%', fmt(v.coefficient), fmt(v.tail_coefficient_contribution), fmt(v.remainder_coefficient_contribution), fmt(v.tail_abs_component_fraction * 100, 1) + '\\%'])
    tab('tail_concentration', 'Action & Tail action share & Coefficient & Tail component & Remainder & Absolute-component share', rows, 'lrrrrr')
    W = pd.read_csv(rr / 'r6_margins_source_weight_summary.csv')
    rows = []
    for _, v in W.iterrows():
        rows.append([v.channel, fmt(v.aggregate_baseline_weighted_ratio, 3), fmt(v.mean_source_ratio, 3), fmt(v.median_source_ratio, 3), fmt(v.top5_source_weight * 100, 1) + '\\%', fmt(v.effective_sources_inverse_HHI, 1)])
    tab('source_weights', 'Action & Aggregate ratio & Mean source ratio & Median ratio & Top-five mass & Effective sources', rows, 'lrrrrr')
    time = pd.read_csv(rr / 'r6_margins_time_slopes.csv')
    rows = []
    for ch in CHANNELS['facebook']:
        v = time[time.channel == ch].set_index('phase')
        rows.append([ch] + [fmt(v.loc[j, 'any_OLS'] * 100) + ' / ' + fmt(v.loc[j, 'count_OLS']) for j in [1, 2, 3, 4]])
    tab('time_slopes', 'Action & Window 1 & Window 2 & Window 3 & Window 4', rows, 'lrrrr')
    refits = pd.read_csv(rr / 'r6_margins_support_refits.csv')
    rows = []
    for _, v in refits.iterrows():
        if v.channel in ['Angry', 'Sad', 'Shares']:
            rule = {'full_release': 'Full release', 'same_binary_profile': 'Same binary', 'same_binary_NN05': 'NN $\\leq0.5$', 'same_binary_NN1': 'NN $\\leq1$'}.get(str(v['support_rule']), str(v['support_rule']).replace('_', ' '))
            rows.append([v.channel, rule, f'{int(v.n):,}', fmt(v.any_OLS * 100), fmt(v.count_OLS), fmt(v.PPML_RR, 3)])
    tab('support_refits', 'Action & Population & Posts & Occurrence (pp) & Count OLS & PPML ratio', rows, 'llrrrr')
    r = S.query("channel=='Likes' and support=='both_H_same_binary_profile'").iloc[0]
    q = S.query("channel=='Likes' and support=='same_binary_and_NN_distance_le_05'").iloc[0]
    a = T.query("channel=='Angry' and population=='common_positive_support'").iloc[0]
    sd = T.query("channel=='Sad' and population=='common_positive_support'").iloc[0]
    tt = tails[tails.channel == 'Total'].iloc[0]
    macros = {'RBinarySupport': fmt(r.post_fraction * 100) + '\\%', 'RNNPostSupport': fmt(q.post_fraction * 100) + '\\%', 'RNNLikeMass': fmt(q.reference_count_weight_retained * 100) + '\\%', 'RAngryTwoDelta': fmt(a.adjusted_difference), 'RAngryTwoOcc': fmt(a.occurrence_component), 'RAngryTwoPositive': fmt(a.positive_count_component), 'RAngryTwoN': f'{int(a.n):,}', 'RSadTwoDelta': fmt(sd.adjusted_difference), 'RSadTwoOcc': fmt(sd.occurrence_component), 'RSadTwoPositive': fmt(sd.positive_count_component), 'RSadTwoN': f'{int(sd.n):,}', 'RTailActionShare': fmt(tt.tail_action_fraction * 100) + '\\%'}
    (rr / 'r6_numbers.tex').write_text('\n'.join(('\\newcommand{\\' + k + '}{' + v + '}' for k, v in macros.items())) + '\n', encoding='utf-8')

def label_assets():
    rr = ROOT / 'results'
    P = pd.read_csv(rr / 'r6_labels_sharp_tipping.csv')
    op = rr / 'r6_labels_OLS_tipping_all.csv'
    O = pd.read_csv(op if op.exists() else rr / 'adversarial_label_tipping.csv')
    rows = []
    for ch in CHANNELS['facebook']:
        p = P[P.channel == ch].set_index('mode')
        o = O[(O.channel == ch) & (O['mode'] == 'positive_label_corrections_only')].set_index('target')
        ov = [f"{int(o.loc[k, 'minimum_flips']):,}" if k in o.index else '---' for k in ['count', 'any']]
        rows.append([ch] + ov + [f"{int(p.loc['positive_label_corrections_only', 'minimum_unit_cost_corrections']):,}", f"{int(p.loc['either_label_corrections', 'minimum_unit_cost_corrections']):,}"])
    tab('label_thresholds', 'Action & OLS count, positive & OLS any, positive & PPML, positive & PPML, either', rows, 'lrrrr')
    path = pd.read_csv(rr / 'r6_labels_OLS_path_PPML.csv')
    rows = []
    for k in [0, 1, 38, 39, 40, 100, 201, 202, 203, 204, 205, 500, 1000]:
        v = path[path.corrections == k].iloc[0]
        rows.append([k, fmt(v.RR, 6), ci(v.working_RR_low, v.working_RR_high, 6)])
    tab('label_path', 'Positive-label corrections & Refitted PPML ratio & Nominal working interval', rows, 'rrr')
    q = pd.read_csv(rr / 'r6_labels_separation.csv')
    assert q.no_additional_slope_separation_certified.all() if 'no_additional_slope_separation_certified' in q else True

def compact_auxiliary():
    q = pd.read_csv(ROOT / 'results/scale_bridge.csv').query("platform=='facebook'")
    rows = []
    names = {'random_intercept': 'Random source intercept', 'source_only': 'Source FE', 'source_month': 'Source + month', 'source_date': 'Source + date', 'source_week_cell': 'Source $\\times$ week', 'source_day_cell': 'Source $\\times$ day'}
    for design, name in names.items():
        a = q[(q.design == design) & (q.target == 'log1p_total')].iloc[0]
        b = q[(q.design == design) & (q.target == 'count_total')].iloc[0]
        rows.append([name, fmt(a.beta, 4), ci(a.low, a.high, 4), fmt(b.beta)])
    tab('scale_bridge', 'Design & Log coefficient & Nominal log interval & Count coefficient', rows, 'lrrr')

def main():
    base_assets()
    inference_assets()
    margin_assets()
    label_assets()
    compact_auxiliary()
    print('Core same-design and inference assets generated.')
if __name__ == '__main__':
    main()
