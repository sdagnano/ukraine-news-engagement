from cell_models import *
from r6_assets import tab as oldtab, fmt, ci, label, line, dot, BLUE, ORANGE, GRAY, LIGHT
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from r7_figures import render_figures

def table(name, header, rows, align):
    s = '\\begin{tabular}{' + align + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n'
    s += '\n'.join((' & '.join(map(str, row)) + ' \\\\' for row in rows)) + '\n\\bottomrule\n\\end{tabular}\n'
    (ROOT / 'tables' / ('r7_' + name + '.tex')).write_text(s, encoding='utf-8')

def main():
    C = pd.read_csv(ROOT / 'results/r7_composition_channels.csv')
    J = json.loads((ROOT / 'results/r7_composition_canonical.json').read_text())
    O = pd.read_csv(ROOT / 'results/cell_OLS.csv').query("platform=='facebook' and design=='date_all' and target=='any' and block==14").set_index('channel')
    rows = []
    for _, v in C.iterrows():
        rows.append([v.channel, fmt(O.loc[v.channel, 'beta'] * 100), fmt(v.OLS_count_beta), fmt(v.PPML_mean0), fmt(v.PPML_mean1), fmt(v.PPML_delta), fmt(v.PPML_RR, 3)])
    s = J['PPML_sum_of_separate_channel_models']
    rows.append(['Sum', '---', fmt(J['OLS_count']['net']), fmt(s['mean0']), fmt(s['mean1']), fmt(s['net']), fmt(s['total_RR'], 3)])
    table('primary', 'Action & Occurrence (pp) & Count OLS & $m_0$ & $m_1$ & $\\Delta^{P}$ & $R^{P}$', rows, 'lrrrrrr')
    S = pd.read_csv(ROOT / 'results/r6_margins_support.csv')
    rows = []
    rules = {'both_H_source_day': 'Both labels in source-day', 'both_H_same_binary_profile': 'Same binary profile', 'same_binary_and_NN_distance_le_05': 'Same binary; NN $\\leq0.5$', 'same_binary_and_NN_distance_le_1': 'Same binary; NN $\\leq1$'}
    for k, name in rules.items():
        q = S[S.support == k].set_index('channel')
        v = q.loc['Likes']
        rows.append([name, f'{int(v.retained_posts):,}', fmt(v.post_fraction * 100) + '\\%', fmt(v.reference_count_weight_retained * 100) + '\\%', fmt(q.loc['Angry', 'reference_count_weight_retained'] * 100) + '\\%'])
    table('support', 'Measured comparison rule & Posts & Post share & Likes mass & Angry mass', rows, 'lrrrr')
    OLS = pd.read_csv(ROOT / 'results/r6_labels_OLS_tipping_all.csv')
    P = pd.read_csv(ROOT / 'results/r6_labels_sharp_tipping.csv')
    rows = []
    for ch in CHANNELS['facebook']:
        z = OLS[(OLS.channel == ch) & (OLS['mode'] == 'positive_label_corrections_only')].set_index('target')
        p = P[P.channel == ch].set_index('mode')
        rows.append([ch] + [f"{int(z.loc[k, 'minimum_flips']):,}" for k in ['any', 'count']] + [f"{int(p.loc[k, 'minimum_unit_cost_corrections']):,}" for k in ['positive_label_corrections_only', 'either_label_corrections']])
    table('tipping', 'Action & OLS occurrence & OLS count & PPML positive only & PPML either class', rows, 'lrrrr')
    rows = []
    for ch in CHANNELS['facebook']:
        vals = []
        for mode in ['positive_label_corrections_only', 'either_label_corrections']:
            z = OLS[(OLS.channel == ch) & (OLS['mode'] == mode)].set_index('target')
            vals.extend((f"{int(z.loc[k, 'minimum_flips']):,}" for k in ['any', 'count']))
        rows.append([ch] + vals)
    table('OLS_classes', 'Action & Positive: occurrence & Positive: count & Either: occurrence & Either: count', rows, 'lrrrr')
    table('observed', 'Action & Observed mean & Zero posts & Zero share & Maximum', [[v.channel, fmt(v.observed_mean), f'{int(v.observed_zeros):,}', fmt(v.observed_zero_fraction * 100, 1) + '\\%', f'{int(v.observed_maximum):,}'] for _, v in C.iterrows()], 'lrrrr')
    B = pd.read_csv(ROOT / 'results/scale_bridge.csv').query("platform=='facebook'")
    rows = []
    for k, name in [('source_only', 'Source'), ('source_date', 'Source + date'), ('source_week_cell', 'Source $\\times$ week'), ('source_day_cell', 'Source $\\times$ day')]:
        a = B[(B.design == k) & (B.target == 'log1p_total')].iloc[0]
        b = B[(B.design == k) & (B.target == 'count_total')].iloc[0]
        rows.append([name, fmt(a.beta, 5), ci(a.low, a.high, 5), fmt(b.beta)])
    table('scale', 'Design & Log-total coefficient & Nominal log interval & Count-total coefficient', rows, 'lrrr')
    render_figures()
    q = J['relative_composition']
    macros = {'RSevenTotalOLS': fmt(J['OLS_count']['net']), 'RSevenPositiveOLS': fmt(J['OLS_count']['positive_sum']), 'RSevenNegativeOLS': fmt(J['OLS_count']['negative_magnitude']), 'RSevenNonlikeOLS': fmt(J['OLS_count']['nonlike_five_reaction_sum']), 'RSevenPPMLNet': fmt(s['net']), 'RSevenPPMLZero': fmt(s['mean0']), 'RSevenPPMLOne': fmt(s['mean1']), 'RSevenPPMLPercent': fmt(s['percent_change']), 'RSevenTheta': fmt(q['ratio_of_ratios'], 3), 'RSevenLogNumerator': fmt(q['shares_log_component'], 5), 'RSevenLogDenominator': fmt(q['likes_decline_log_component'], 5)}
    (ROOT / 'results/r7_numbers.tex').write_text('\n'.join(('\\newcommand{\\' + k + '}{' + v + '}' for k, v in macros.items())) + '\n', encoding='utf-8')
    print('R7 canonical displays and macros generated; no fitted model changed.')
if __name__ == '__main__':
    main()
