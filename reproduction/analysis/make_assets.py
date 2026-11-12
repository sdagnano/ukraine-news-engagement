from cell_models import *
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
BLUE = HexColor('#205B8F')
ORANGE = HexColor('#B85423')
GRAY = HexColor('#737D86')
LIGHT = HexColor('#DCE1E5')

def label(c, x, y, s, size=8, color=GRAY):
    c.setFillColor(color)
    c.setFont('Helvetica', size)
    c.drawString(x, y, str(s))

def line(c, x, y, xx, yy, color=LIGHT, width=0.6):
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x, y, xx, yy)

def dot(c, x, y, color=BLUE, r=2):
    c.setFillColor(color)
    c.circle(x, y, r, stroke=0, fill=1)

def fmt(x, d=2):
    return f'{float(x):.{d}f}'

def ci(a, b, d=2):
    return '[' + fmt(a, d) + ', ' + fmt(b, d) + ']'

def tab(name, header, rows, align):
    s = '\\begin{tabular}{' + align + '}\n\\toprule\n' + header + ' \\\\\n\\midrule\n' + '\n'.join((' & '.join(map(str, r)) + ' \\\\' for r in rows)) + '\n\\bottomrule\n\\end{tabular}\n'
    (ROOT / 'tables' / f'{name}.tex').write_text(s, encoding='utf-8')

def figures():
    O = pd.read_csv(ROOT / 'results/cell_OLS.csv').query("platform=='facebook' and design=='date_all' and target=='any' and block==14").set_index('channel')
    P = pd.read_csv(ROOT / 'results/cell_PPML.csv').query("platform=='facebook' and design=='date_all' and block==14").set_index('channel')
    c = canvas.Canvas(str(ROOT / 'figures/margins.pdf'), pagesize=(510, 290))
    c.setTitle('Post-level occurrence and count-volume associations')
    x1 = lambda v: 90 + (v + 11) / 30 * 165
    x2 = lambda v: 320 + (np.log(v) - np.log(0.4)) / (np.log(2.8) - np.log(0.4)) * 160
    for v in [-10, 0, 10, 20]:
        if v <= 19:
            line(c, x1(v), 38, x1(v), 252)
            label(c, x1(v) - 7, 24, v)
    for v in [0.5, 1, 2]:
        line(c, x2(v), 38, x2(v), 252)
        label(c, x2(v) - 6, 24, v)
    line(c, x1(0), 38, x1(0), 252, GRAY, 1)
    line(c, x2(1), 38, x2(1), 252, GRAY, 1)
    for i, ch in enumerate(CHANNELS['facebook']):
        y = 240 - i * 27
        v = O.loc[ch]
        p = P.loc[ch]
        label(c, 12, y - 3, ch, 9)
        line(c, x1(v.low * 100), y, x1(v.high * 100), y, BLUE, 1.2)
        dot(c, x1(v.beta * 100), y)
        line(c, x2(np.exp(p.logRR_low)), y, x2(np.exp(p.logRR_high)), y, ORANGE, 1.2)
        dot(c, x2(p.RR), y, ORANGE)
    label(c, 90, 271, 'Occurrence difference (percentage points)', 9, BLUE)
    label(c, 320, 271, 'Count mean ratio (log axis)', 9, ORANGE)
    label(c, 90, 7, 'Source-day effects; source / 14-day-block covariance.', 8)
    c.showPage()
    c.save()
    H = pd.read_csv(ROOT / 'results/hierarchical_sources.csv').query("model=='block14'")
    ids = sorted(H.source.unique())
    names = ['Likes', 'Love', 'Angry', 'Haha']
    c = canvas.Canvas(str(ROOT / 'figures/aligned_sources.pdf'), pagesize=(540, 700))
    c.setTitle('Aligned source estimates and conditional empirical Bayes summaries')
    bottom = 54
    top = 645
    step = (top - bottom) / (len(ids) - 1)
    for i, sg in enumerate(ids):
        yy = top - i * step
        label(c, 5, yy - 2, f'S{sg + 1:02}', 6.6)
        if i % 2 == 0:
            line(c, 32, yy, 528, yy, LIGHT, 0.3)
    for k, ch in enumerate(names):
        z = H[H.channel == ch].set_index('source').loc[ids]
        ox = 39 + k * 124
        W = 111
        lo = np.arcsinh(min(z.low.min(), 0))
        hi = np.arcsinh(max(z.high.max(), 0))
        pad = (hi - lo) * 0.025
        lo -= pad
        hi += pad
        fx = lambda v: ox + W * (np.arcsinh(v) - lo) / (hi - lo)
        line(c, fx(0), bottom - 5, fx(0), top + 5, GRAY, 0.7)
        label(c, ox + 28, 668, ch, 10, BLUE)
        ticks = [-1000, -10, 0, 10, 1000] if ch == 'Likes' else [-100, -1, 0, 1, 100]
        for v in ticks:
            if lo < np.arcsinh(v) < hi:
                label(c, fx(v) - 8, 35, v, 6)
        for i, (_, v) in enumerate(z.iterrows()):
            yy = top - i * step
            line(c, fx(v.low), yy, fx(v.high), yy, GRAY, 0.5)
            dot(c, fx(v.beta), yy, GRAY, 1.1)
            dot(c, fx(v.posterior_mean), yy, BLUE, 1.7)
    label(c, 34, 17, 'Gray: source estimate / normal working interval. Blue: conditional EB mean (14-day covariance).', 7)
    label(c, 34, 6, 'Asinh axes, ticks in actions/post. IDs align across all outcomes; intervals are not simultaneous.', 7)
    c.showPage()
    c.save()

def build():
    O = pd.read_csv(ROOT / 'results/cell_OLS.csv')
    P = pd.read_csv(ROOT / 'results/cell_PPML.csv')
    J = pd.read_csv(ROOT / 'results/cell_joint.csv')
    b = pd.read_csv(ROOT / 'results/scale_bridge.csv').query("platform=='facebook'")
    rows = []
    labels = {'random_intercept': 'Random source intercept', 'source_only': 'Source FE', 'source_month': 'Source + month', 'source_date': 'Source + date', 'source_week_cell': 'Source $\\times$ week', 'source_day_cell': 'Source $\\times$ day'}
    for design in labels:
        a = b[(b.design == design) & (b.target == 'log1p_total')].iloc[0]
        z = b[(b.design == design) & (b.target == 'count_total')].iloc[0]
        rows.append([labels[design], fmt(a.beta, 4) + ' ' + ci(a.low, a.high, 4), fmt(z.beta) + ' ' + ci(z.low, z.high)])
    tab('scale_bridge', 'Design & Log-total coefficient (interval) & Count coefficient (interval)', rows, 'lrr')
    primary = P.query("platform=='facebook' and design=='date_all' and block==14").set_index('channel')
    rows = []
    for ch in CHANNELS['facebook']:
        a = primary.loc[ch]
        rows.append([ch, fmt(a.mean0), fmt(a.mean1), fmt(a.delta), fmt(a.RR, 3)])
    tab('count_means', 'Action & $m_0$ & $m_1$ & Difference & Ratio', rows, 'lrrrr')
    tail = pd.read_csv(ROOT / 'results/tail_contributions.csv').query("platform=='facebook' and quantile==0.99")
    counts = tail[tail.channel != 'log1p_total']
    total = counts[['beta', 'tail_contribution', 'body_contribution']].sum()
    rows = []
    for _, v in counts.iterrows():
        rows.append([v.channel, fmt(v.beta), fmt(v.tail_contribution), fmt(v.body_contribution), fmt(v.signed_tail_fraction * 100, 1) + '\\%'])
    rows.append(['Total', fmt(total.beta), fmt(total.tail_contribution), fmt(total.body_contribution), fmt(total.tail_contribution / total.beta * 100, 1) + '\\%'])
    tab('tails', 'Action & Full coefficient & Top 1\\% & Remaining 99\\% & Signed share', rows, 'lrrrr')
    q = pd.read_csv(ROOT / 'results/all_source_pair_deletions.csv').query("platform=='facebook'")
    rows = []
    base = O.query("platform=='facebook' and design=='date_all' and target=='count' and block==14").set_index('channel')
    for ch in CHANNELS['facebook']:
        a = q[q.channel == ch]
        rows.append([ch, fmt(base.loc[ch, 'beta']), ci(a.beta.min(), a.beta.max()), int(a.shape[0])])
    tab('pairs', 'Action & Full coefficient & All source-pair range & Pairs', rows, 'lrrr')
    a = pd.read_csv(ROOT / 'results/adversarial_label_tipping.csv')
    a = a[a['mode'] == 'positive_label_corrections_only']
    rows = []
    for ch in ['Likes', 'Love', 'Angry', 'Haha']:
        z = a[a.channel == ch].set_index('target')
        count = z.loc['count']
        anyv = z.loc['any']
        rows.append([ch, f'{int(count.minimum_flips):,}', fmt(count.fraction_positive_labels * 100, 3) + '\\%', f'{int(anyv.minimum_flips):,}', fmt(anyv.fraction_positive_labels * 100, 3) + '\\%'])
    tab('tipping', 'Action & Count flips & Of positive labels & Occurrence flips & Of positive labels', rows, 'lrrrr')
    g = pd.read_csv(ROOT / 'results/differential_label_grid.csv')
    rows = []
    for high in [0, 0.1, 0.2, 0.4, 0.6]:
        z = g[(g.channel == 'Angry') & (g.positive_low_correction == 0) & (g.negative_correction == 0) & (g.positive_high_Angry_correction == high)].set_index('target')
        a = z.loc['count']
        v = z.loc['any']
        rows.append([f'{high * 100:.0f}\\%', fmt(a.median_beta), ci(a.assumption_q05, a.assumption_q95), fmt(v.median_beta * 100), ci(v.assumption_q05 * 100, v.assumption_q95 * 100)])
    tab('label_grid', 'High-Angry corrections & Count median & Assumed-label range & Occurrence pp & Assumed-label range', rows, 'lrrrr')
    rows = []
    for plat in ['facebook', 'twitter']:
        for design in ['week_all', 'date_all', 'date_both']:
            v = J.query('platform==@plat and design==@design and block==14').iloc[0]
            s = pd.read_csv(ROOT / 'results/cell_support.csv').query('platform==@plat and design==@design').iloc[0]
            rows.append(['FB' if plat == 'facebook' else 'TW', design.replace('_', ' '), f'{int(s.posts):,}', int(s.cells), fmt(v.net), ci(v.net_low, v.net_high), fmt(v.ratio, 3)])
    tab('cells', 'Platform & Design & Posts & Cells & PPML net & Net interval & $\\Theta$', rows, 'llrrrrr')
    rows = []
    for ch in CHANNELS['facebook']:
        row = [ch]
        for width in [1, 14, 28]:
            v = P.query("platform=='facebook' and design=='date_all' and channel==@ch and block==@width").iloc[0]
            row.append(ci(np.exp(v.logRR_low), np.exp(v.logRR_high), 3))
        rows.append(row)
    tab('blocks', 'Action & Daily interval & 14-day interval & 28-day interval', rows, 'lrrr')
    rows = []
    for ch in CHANNELS['facebook']:
        v = O.query("platform=='facebook' and design=='date_all' and target=='any' and channel==@ch and block==14").iloc[0]
        p = primary.loc[ch]
        q16 = stats.t.ppf(1 - 0.05 / 32, v.df)
        rows.append([ch, ci((v.beta - q16 * v.se) * 100, (v.beta + q16 * v.se) * 100), ci(np.exp(np.log(p.RR) - q16 * p.logRR_se), np.exp(np.log(p.RR) + q16 * p.logRR_se), 3)])
    tab('combined', 'Action & Occurrence interval (pp) & Count-ratio interval', rows, 'lrr')
    I = pd.read_csv(ROOT / 'results/intensive_projections.csv').query("platform=='facebook' and block==14")
    rows = []
    for _, v in I.iterrows():
        rows.append([v.channel, f'{int(v.n):,}', fmt(v.beta), ci(v.low, v.high)])
    tab('intensive', 'Action & Positive posts & Count coefficient & Simultaneous interval', rows, 'lrrr')
    A = pd.read_csv(ROOT / 'results/two_part_accounting.csv').query("platform=='facebook'")
    rows = []
    for _, v in A.iterrows():
        rows.append([v.channel, fmt(v.delta), fmt(v.activation), fmt(v.intensity), fmt(v.unsupported_positive_margin), fmt(v.both_positive_weight * 100, 1) + '\\%'])
    tab('accounting', 'Action & Mean difference & Occurrence term & Positive-count term & Unsupported & Supported weight', rows, 'lrrrrr')
    H = pd.read_csv(ROOT / 'results/ppml_heterogeneity.csv')
    rows = []
    for ch in ['Likes', 'Love', 'Angry', 'Haha']:
        row = [ch]
        for model in ['constant_H', 'quartile_H_slopes', 'all_slopes_source_specific']:
            z = H[(H.channel == ch) & (H.population == 'common_source_finite') & (H.model == model)].iloc[0]
            row.append(fmt(z.RR, 3) + ' / ' + fmt(z.delta))
        rows.append(row)
    tab('heterogeneity', 'Action & Constant ratio / difference & Quartile ratio / difference & Source-specific ratio / difference', rows, 'lrrr')
    CV = pd.read_csv(ROOT / 'results/ppml_date_holdout.csv')
    rows = []
    for ch in ['Likes', 'Love', 'Angry', 'Haha']:
        a = CV[(CV.channel == ch) & (CV.model == 'constant_H')]
        b = CV[(CV.channel == ch) & (CV.model == 'quartile_H')]
        mse0 = np.average(a.MSE, weights=a.heldout_posts)
        mse1 = np.average(b.MSE, weights=b.heldout_posts)
        dev0 = np.average(a.deviance_positive_training_cells, weights=a.positive_training_cell_posts)
        dev1 = np.average(b.deviance_positive_training_cells, weights=b.positive_training_cell_posts)
        rows.append([ch, fmt((mse1 / mse0 - 1) * 100, 3) + '\\%', fmt((dev1 / dev0 - 1) * 100, 3) + '\\%', int(a.zero_prediction_positive_outcome.sum()), int(a.unseen_cell_posts.sum())])
    tab('prediction', 'Action & MSE change & Finite-cell deviance change & Positive outcomes predicted 0 & Unseen-cell posts', rows, 'lrrrr')
    E = json.loads((ROOT / 'audit/hierarchical_models.json').read_text())
    rows = []
    hn = {'day': 'Daily', 'block14': '14-day', 'block14_volume': '14-day + volume'}
    for z in E:
        rows.append([hn[z['model']], z['sources'], z['raw_pattern'], z['posterior_mean_pattern'], z['posterior_joint_over95'], fmt(z['mean_joint_probability'], 3)])
    tab('hierarchy', 'Working model & Sources & Raw profile & EB mean profile & Joint $p>0.95$ & Mean joint $p$', rows, 'lrrrrr')
    W = pd.read_csv(ROOT / 'results/working_covariance_sensitivity.csv').query("target=='count' and channel in ['Likes','Love','Angry','Haha']")
    rows = []
    for _, v in W.iterrows():
        rows.append([v.channel, 'Identity' if v.working_covariance == 'identity' else 'Source variances', fmt(v.df, 2), ci(v.low, v.high)])
    tab('working_covariance', 'Action & Working covariance & df & Eight-channel interval', rows, 'llrr')
    d = load('facebook')
    s, t, c = grouping(d, 'facebook', 'date')
    rows = []
    for ch in CHANNELS['facebook']:
        totals = ag(d[ch].to_numpy(float), c)
        zero = totals == 0
        rows.append([ch, int(zero.sum()), f'{int(zero[c].sum()):,}', int((totals > 0).sum())])
    tab('boundaries', 'Action & Zero-total cells & Posts in zero cells & Positive-total cells', rows, 'lrrr')
    S = pd.read_csv(ROOT / 'results/unknown139_stress.csv')
    rows = []
    for _, v in S.iterrows():
        rows.append([v.target_channel, v.selection.replace('_', ' '), fmt(v.original_beta), fmt(v.refitted_beta), fmt(v.refitted_total)])
    tab('unknown139', 'Target action & Stress selection & Full slope & Refitted slope & Refitted total', rows, 'llrrr')
    macros = dict(AngryAny=fmt(O.query("platform=='facebook' and design=='date_all' and target=='any' and block==14 and channel=='Angry'").iloc[0].beta * 100), AngryRR=fmt(primary.loc['Angry', 'RR'], 3), TailNetPercent=fmt(total.tail_contribution / total.beta * 100, 2) + '\\%', TailNetValue=fmt(total.tail_contribution), BodyNetValue=fmt(total.body_contribution))
    for ch, key in [('Angry', 'Angry'), ('Shares', 'Shares'), ('Sad', 'Sad')]:
        z = O.query("platform=='facebook' and design=='date_all' and target=='any' and block==14 and channel==@ch").iloc[0]
        macros[key + 'Any'] = fmt(z.beta * 100)
        macros[key + 'AnyCI'] = ci(z.low * 100, z.high * 100)
        macros[key + 'RR'] = fmt(primary.loc[ch, 'RR'], 3)
        macros[key + 'RRCI'] = ci(np.exp(primary.loc[ch, 'logRR_low']), np.exp(primary.loc[ch, 'logRR_high']), 3)
    v = J.query("platform=='facebook' and design=='date_all' and block==14").iloc[0]
    macros.update(CellNet=fmt(v.net), CellNetCI=ci(v.net_low, v.net_high), CellTheta=fmt(v.ratio, 3))
    if (ROOT / 'results/calibration_full.csv').exists():
        C = pd.concat([pd.read_csv(ROOT / 'results/calibration_subsample.csv'), pd.read_csv(ROOT / 'results/calibration_full.csv')])
        rows = []
        scen = {'independent_NB': 'A: independent', 'source_slopes': 'B: source', 'source_and_block_slopes': 'C: source + block'}
        for _, v in C.iterrows():
            rows.append(['Full' if v.design == 'full' else '10\\%', scen[v.scenario], 'PPML' if v.method == 'PPML' else 'Any OLS', int(v.replicates), fmt(v.family_noncoverage, 3), ci(v.MC_low, v.MC_high, 3), fmt(v.false_opposition, 3) if pd.notna(v.false_opposition) else '--', fmt(v.invalid_fraction, 3)])
        tab('calibration', 'Design & DGP & Method & Reps & Noncoverage & MC interval & Opposite & Invalid', rows, 'lllrrrrr')
        f = C.query("design=='full'").set_index('method')
        macros['CalibrationResults'] = 'In the full-size design-anchored source-plus-block simulation, family noncoverage is ' + fmt(f.loc['PPML', 'family_noncoverage'] * 100, 1) + '\\% for PPML and ' + fmt(f.loc['any_OLS', 'family_noncoverage'] * 100, 1) + '\\% for occurrence projections, with 100 replicates and wide Monte Carlo intervals. No simultaneous positive/negative count declaration occurs in those 100 realizations. Table~\\ref{tab:calibration} reports invalid cases and the smaller-design scenarios. The reference parameters marginalize over re-drawn source/block slopes; these are not error bounds for a finite-source conditional parameter or estimates of how often the observed four-sign profile is false. The failure of nominal count-interval coverage limits their inferential interpretation even when fitted mean profiles remain stable.'
        F = pd.read_csv(ROOT / 'results/calibration_full_fit.csv').query("scenario=='source_and_block_slopes'")
        rows = []
        for ch in CHANNELS['facebook']:
            a = F[(F.channel == ch) & F.alpha.notna()].iloc[0]
            b = F[(F.channel == ch) & F.simulated_zero.notna()].iloc[0]
            rows.append([ch, fmt(a.observed_zero * 100, 1) + '\\%', fmt(b.simulated_zero * 100, 1) + '\\%', fmt(b.observed_q99, 1), fmt(b.mean_simulated_q99, 1)])
        tab('dgp_fit', 'Action & Observed zeros & Simulated zeros & Observed 99th & Simulated 99th', rows, 'lrrrr')
        if (ROOT / 'results/full_DGP_tail_fit.csv').exists():
            z = pd.read_csv(ROOT / 'results/full_DGP_tail_fit.csv')
            rows = []
            for _, v in z.iterrows():
                rows.append([v.channel, fmt(v.observed_q999, 0), fmt(v.simulated_q999_median, 0), fmt(v.observed_maximum, 0), fmt(v.simulated_maximum_median, 0)])
            tab('dgp_tail', 'Action & Observed99.9th & Simulated99.9th & Observed max & Median simulated max', rows, 'lrrrr')
    else:
        macros['CalibrationResults'] = 'Calibration results are unavailable for this run.'
    (ROOT / 'results/numbers.tex').write_text('\n'.join(('\\newcommand{\\' + k + '}{' + v + '}' for k, v in macros.items())) + '\n', encoding='utf-8')
    print(json.dumps(macros, indent=2))
if __name__ == '__main__':
    figures()
    build()
