from cell_models import *
from scipy.special import expit, logsumexp
from scipy.spatial import cKDTree
from r6_labels_separation import poisson_separation_certificate
import datetime, time, argparse
PREFIX = 'r6_margins'
CHANNEL = CHANNELS['facebook']

def save(rows, name):
    pd.DataFrame(rows).to_csv(ROOT / 'results' / f'{PREFIX}_{name}.csv', index=False)

def independent_columns(X, c, priority=1):
    ng = np.bincount(c)
    xc = X - ag(X, c)[c] / ng[c, None]
    order = [priority] + [j for j in range(X.shape[1]) if j != priority]
    keep = []
    rank = 0
    for j in order:
        nr = np.linalg.matrix_rank(xc[:, keep + [j]])
        if nr > rank:
            keep.append(j)
            rank = nr
    if priority not in keep:
        raise RuntimeError('Hostility unidentified after cell absorption')
    return sorted(keep)

def logistic_profile(X, y, c, tol=2e-10, maxiter=60):
    n, p = X.shape
    C = int(c.max() + 1)
    ng = np.bincount(c)
    yt = ag(y, c, C)
    mixed = (yt > 0) & (yt < ng)
    active = mixed[c]
    if not np.any(active):
        raise RuntimeError('No mixed-outcome cells: logit slopes unidentified')
    keep = independent_columns(X[active], codes(c[active]), priority=1)
    XX = X[:, keep]
    xx = XX[active]
    yy = y[active]
    cc = codes(c[active])
    nn = np.bincount(cc)
    sy = ag(yy, cc)

    def ev(b, need_hess=False):
        eta = xx @ b
        center = np.log(sy / (nn - sy))
        aa = center - ag(eta, cc) / nn
        emin = np.full(len(nn), np.inf)
        emax = np.full(len(nn), -np.inf)
        np.minimum.at(emin, cc, eta)
        np.maximum.at(emax, cc, eta)
        lower = center - emax
        upper = center - emin
        intercept_tol = np.maximum(2e-10, nn * 2e-13)
        for inner in range(100):
            q = expit(eta + aa[cc])
            v = q * (1 - q)
            g = ag(q, cc) - sy
            den = ag(v, cc)
            if np.all(abs(g) <= intercept_tol):
                break
            upper = np.where(g > 0, aa, upper)
            lower = np.where(g < 0, aa, lower)
            step = np.divide(g, den, out=np.zeros_like(g), where=den > 1e-16)
            proposed = aa - step
            interior = (proposed >= lower) & (proposed <= upper) & np.isfinite(proposed) & (den > 1e-16)
            aa = np.where(abs(g) <= intercept_tol, aa, np.where(interior, proposed, 0.5 * (lower + upper)))
        else:
            raise RuntimeError('Logit intercept profiling not converged')
        lin = eta + aa[cc]
        q = expit(lin)
        v = q * (1 - q)
        objective = np.sum(np.logaddexp(0, lin) - yy * lin) / len(yy)
        gradient = xx.T @ (q - yy) / len(yy)
        H = None
        if need_hess:
            sv = ag(v, cc)
            sx = ag(v[:, None] * xx, cc)
            H = xx.T @ (v[:, None] * xx) - sx.T @ (sx / sv[:, None])
        return (objective, gradient, H, aa, q)
    b = np.zeros(len(keep))
    for it in range(maxiter):
        obj, gr, H, alpha, q = ev(b, True)
        if abs(gr).max() < tol:
            break
        eig = np.linalg.eigvalsh(H)
        if eig.min() <= eig.max() * 1e-13:
            raise RuntimeError('Logit profiled information nonidentified')
        step = linalg.solve(H, gr * len(yy), assume_a='pos')
        scale = 1.0
        for _ in range(35):
            no, ngrad, _, _, _ = ev(b - scale * step)
            if no < obj - 0.0001 * scale * (gr @ step) or (no <= obj + 1e-13 and abs(ngrad).max() < abs(gr).max()):
                break
            scale *= 0.5
        else:
            raise RuntimeError('Logit profile line search failed')
        b -= scale * step
        if abs(b).max() > 40:
            raise RuntimeError('Logit slope diverged: possible separation')
    obj, gr, H, alpha, q = ev(b, True)
    if abs(gr).max() > 2e-08:
        raise RuntimeError('Logit final score did not converge')
    hcol = keep.index(1)
    bfull = np.zeros(p)
    bfull[keep] = b
    base = np.zeros(n)
    p0 = np.zeros(n)
    p1 = np.zeros(n)
    ones = (yt == ng)[c]
    p0[ones] = 1
    p1[ones] = 1
    base[active] = alpha[cc] + xx @ b - XX[active, hcol] * b[hcol]
    p0[active] = expit(base[active])
    p1[active] = expit(base[active] + b[hcol])
    return dict(beta=bfull, p0=p0, p1=p1, keep=keep, gradient=float(abs(gr).max()), mixed_cells=int(mixed.sum()), zero_cells=int((yt == 0).sum()), one_cells=int((yt == ng).sum()), max_intercept_score=float(np.max(abs(ag(q, cc) - sy))), iterations=it, min_information_eigenvalue=float(np.linalg.eigvalsh(H).min()), condition=float(np.linalg.cond(H)))

def validate():
    rng = np.random.default_rng(503172)
    G = 13
    n = 260
    c = np.repeat(np.arange(G), 20)
    X = np.column_stack([rng.integers(0, 2, n), rng.integers(0, 2, n), rng.normal(size=n)])
    D = np.eye(G)[c]
    W = np.column_stack([D, X])
    true = np.r_[rng.normal(0, 0.5, G), 0.2, 0.6, -0.1]
    y = rng.binomial(1, expit(W @ true)).astype(float)
    fit = logistic_profile(X, y, c)

    def full_hess(b):
        q = expit(W @ b)
        return W.T @ ((q * (1 - q))[:, None] * W)
    res = optimize.minimize(lambda b: (np.sum(np.logaddexp(0, W @ b) - y * (W @ b)), W.T @ (expit(W @ b) - y)), np.zeros(W.shape[1]), jac=True, hess=full_hess, method='trust-exact', options={'gtol': 1e-08, 'maxiter': 1000})
    err = float(np.max(abs(res.x[-3:] - fit['beta'])))
    p0 = expit(W @ res.x - X[:, 1] * res.x[-2])
    p1 = expit(W @ res.x + (1 - X[:, 1]) * res.x[-2])
    pred = max(float(abs(p0 - fit['p0']).max()), float(abs(p1 - fit['p1']).max()))
    assert res.success and err < 2e-06 and (pred < 2e-06) and (abs(res.jac).max() < 1e-07)
    p0 = rng.uniform(0, 1, n)
    p1 = rng.uniform(0, 1, n)
    m0 = 1 + rng.exponential(20, n)
    m1 = 1 + rng.exponential(30, n)
    exact = p1 * m1 - p0 * m0 - (0.5 * (m1 + m0) * (p1 - p0) + 0.5 * (p1 + p0) * (m1 - m0))
    identity = float(abs(exact).max())
    assert identity < 1e-12
    out = dict(passed=True, seed=503172, logit_vs_full_dummy_coefficient_max_error=err, logit_vs_full_dummy_prediction_max_error=pred, symmetric_decomposition_max_error=identity, full_dummy_optimizer_method='scipy trust-exact, analytic full-dummy Hessian', full_dummy_optimizer_success=bool(res.success), full_dummy_score=float(abs(res.jac).max()))
    (ROOT / 'audit' / f'{PREFIX}_validation.json').write_text(json.dumps(out, indent=2))
    return out

def fitted_mean(X, y, c, b):
    C = int(c.max() + 1)
    eta = X @ b
    mx = np.full(C, -np.inf)
    np.maximum.at(mx, c, eta)
    ex = np.exp(eta - mx[c])
    mu = ag(y, c, C)[c] * ex / ag(ex, c, C)[c]
    return mu

def support(d, X, Y, s, t, c):
    names = list(ukrainian_design(d).columns)
    binary = [names.index(k) for k in ['solidarity', 'ingroup_mention', 'outgroup_mention', 'has_media', 'has_URL']]
    cont = [names.index(k) for k in ['moral_emotional_z', 'positive_z', 'negative_z']]
    pattern = (X[:, binary] @ 2 ** np.arange(len(binary))).astype(int)
    H = X[:, 1].astype(int)
    joint = codes(pd.MultiIndex.from_arrays([c, pattern]))
    J = joint.max() + 1
    counts = ag(np.ones(len(X)), joint * 2 + H, 2 * J).reshape(J, 2)
    bothbinary = (counts.min(1) > 0)[joint]
    cellcounts = ag(np.ones(len(X)), c * 2 + H, 2 * (c.max() + 1)).reshape(-1, 2)
    bothcell = (cellcounts.min(1) > 0)[c]
    z = X[:, cont]
    distance = np.full(len(X), np.inf)
    box = np.zeros(len(X), bool)
    order = np.argsort(joint, kind='stable')
    edges = np.r_[0, np.flatnonzero(np.diff(joint[order])) + 1, len(order)]
    for a, b in zip(edges[:-1], edges[1:]):
        ix = order[a:b]
        neg = ix[H[ix] == 0]
        pos = ix[H[ix] == 1]
        if len(neg) == 0 or len(pos) == 0:
            continue
        for target, opposite in [(neg, pos), (pos, neg)]:
            distance[target] = cKDTree(z[opposite]).query(z[target], k=1)[0]
            lo = z[opposite].min(0)
            hi = z[opposite].max(0)
            box[target] = np.all((z[target] >= lo - 1e-12) & (z[target] <= hi + 1e-12), axis=1)
    nuisance = [j for j in range(X.shape[1]) if j != 1]
    Xp = X[:, nuisance]
    fp = logistic_profile(Xp, H.astype(float), c)
    prop = np.where(Xp[:, 1] > 0, fp['p1'], fp['p0'])
    assert names[nuisance[1]] == 'ingroup_mention'
    masks = {'both_H_source_day': bothcell, 'both_H_same_binary_profile': bothbinary, 'opposite_H_continuous_bounding_box': box, 'same_binary_and_NN_distance_le_025': distance <= 0.25, 'same_binary_and_NN_distance_le_05': distance <= 0.5, 'same_binary_and_NN_distance_le_1': distance <= 1, 'fitted_H_propensity_005_095': (prop >= 0.05) & (prop <= 0.95), 'fitted_H_propensity_01_09': (prop >= 0.1) & (prop <= 0.9)}
    pars = np.load(ROOT / 'results/cell_model_parameters.npz')
    rows = []
    for ch, y in zip(CHANNEL, Y.T):
        beta = pars['facebook_date_all_' + ch]
        mu = fitted_mean(X, y, c, beta)
        m0 = mu * np.exp(-H * beta[1])
        m1 = m0 * np.exp(beta[1])
        for name, mask in masks.items():
            rows.append(dict(channel=ch, support=name, retained_posts=int(mask.sum()), post_fraction=float(mask.mean()), reference_count_weight_retained=float(m0[mask].sum() / m0.sum()), label1_count_weight_retained=float(m1[mask].sum() / m1.sum()), observed_action_fraction_retained=float(y[mask].sum() / y.sum()), retained_sources=int(np.unique(s[mask]).size), retained_cells=int(np.unique(c[mask]).size)))
    save(rows, 'support')
    quantiles = []
    for h in [0, 1]:
        for q in [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]:
            quantiles.append(dict(H=h, quantile=q, propensity=float(np.quantile(prop[H == h], q)), opposite_NN_distance=float(np.quantile(distance[(H == h) & np.isfinite(distance)], q))))
    save(quantiles, 'support_distribution')
    np.savez_compressed(ROOT / 'results' / f'{PREFIX}_support_arrays.npz', same_binary=bothbinary, opposite_box=box, distance=distance, propensity=prop)
    (ROOT / 'audit' / f'{PREFIX}_support.json').write_text(json.dumps(dict(binary_covariates=[names[k] for k in binary], continuous_distance_covariates=[names[k] for k in cont], same_binary_strata=int(J), propensity_fit={k: v for k, v in fp.items() if k not in ['beta', 'p0', 'p1']}, note='Distances are Euclidean in full-release SD units. Thresholds are diagnostic choices, not positivity proofs. Bounding boxes are not convex-hull guarantees. Propensity model is in-sample, with observed all-H cells at boundary0/1; it is neither a causal assignment model nor validated overlap probability.'), indent=2))
    print('support complete', flush=True)
    return masks

def tails(Y):
    total = Y.sum(1)
    old = pd.read_csv(ROOT / 'results/tail_contributions.csv').query("platform=='facebook'")
    rows = []
    for q in [0.95, 0.99, 0.999]:
        cut = np.quantile(total, q)
        mask = total > cut
        for k, ch in enumerate(CHANNEL + ['Total']):
            y = total if ch == 'Total' else Y[:, k]
            if ch == 'Total':
                z = old[(old['quantile'] == q) & old.channel.isin(CHANNEL)]
                beta = z.beta.sum()
                a = z.tail_contribution.sum()
                b = z.body_contribution.sum()
            else:
                z = old[(old['quantile'] == q) & (old.channel == ch)].iloc[0]
                beta = z.beta
                a = z.tail_contribution
                b = z.body_contribution
            rows.append(dict(channel=ch, quantile=q, cutoff=cut, tail_posts=int(mask.sum()), tail_post_fraction=mask.mean(), observed_actions=y.sum(), tail_action_sum=y[mask].sum(), tail_action_fraction=y[mask].sum() / y.sum(), coefficient=beta, tail_coefficient_contribution=a, remainder_coefficient_contribution=b, tail_abs_component_fraction=abs(a) / (abs(a) + abs(b)) if abs(a) + abs(b) > 0 else np.nan, signed_fraction_not_headline=a / beta if abs(beta) > 1e-08 else np.nan))
    save(rows, 'tail_action_concentration')
    print('tail action concentration complete', flush=True)

def source_weights():
    sr = pd.read_csv(ROOT / 'results/ppml_source_slopes.csv')
    audit = json.loads((ROOT / 'audit/ppml_heterogeneity.json').read_text())
    common = audit['common_finite_sources']
    rows = []
    detail = []
    for ch in ['Likes', 'Love', 'Angry', 'Haha']:
        z = sr[(sr.channel == ch) & sr.source.isin(common)].copy()
        z['ratio'] = np.exp(z.beta)
        z['baseline_mass'] = z.n * z.mean0
        w = z.baseline_mass / z.baseline_mass.sum()
        rw = z.n / z.n.sum()
        sortedw = np.sort(w)[::-1]
        R = float(np.sum(w * z.ratio))
        M = float(np.sum(rw * z.mean1) / np.sum(rw * z.mean0))
        assert abs(R - M) < 1e-12
        rows.append(dict(channel=ch, sources=len(z), posts=int(z.n.sum()), aggregate_baseline_weighted_ratio=R, mean_source_ratio=float(z.ratio.mean()), median_source_ratio=float(z.ratio.median()), post_weighted_mean_source_ratio=float(np.sum(rw * z.ratio)), equal_source_marginal_mean_ratio=float(z.mean1.mean() / z.mean0.mean()), source_ratio_min=float(z.ratio.min()), source_ratio_max=float(z.ratio.max()), source_ratio_q10=float(z.ratio.quantile(0.1)), source_ratio_q90=float(z.ratio.quantile(0.9)), fraction_sources_ratio_above1=float((z.ratio > 1).mean()), baseline_mass_ratio_above1=float(w[z.ratio > 1].sum()), largest_source_weight=float(sortedw[0]), top5_source_weight=float(sortedw[:5].sum()), top10_source_weight=float(sortedw[:10].sum()), effective_sources_inverse_HHI=float(1 / np.sum(w * w)), identity_error=R - M))
        for (_, r), weight in zip(z.iterrows(), w):
            detail.append(dict(channel=ch, source=int(r.source), n=int(r.n), RR=r.ratio, mean0=r.mean0, mean1=r.mean1, baseline_weight=float(weight)))
    save(rows, 'source_weight_summary')
    save(detail, 'source_weights')
    print('source weights complete', flush=True)

def time_slopes(X, Y, s, t, c, d):
    phase = t // 50
    assert phase.max() == 3
    P = X.shape[1]
    Z = np.column_stack([X] + [X[:, 1] * (phase == j) for j in [1, 2, 3]])
    rows = []
    audit = []
    ols = cell_ols(Z, np.column_stack([Y, (Y > 0).astype(float)]), c)
    K = Y.shape[1]
    for k, ch in enumerate(CHANNEL):
        fit = ppml(Z, Y[:, k], c)
        b = fit['beta']
        slopes = b[1] + np.r_[0, b[P:]]
        slope = slopes[phase]
        m0 = fit['mu'] * np.exp(-X[:, 1] * slope)
        m1 = m0 * np.exp(slope)
        for j in range(4):
            ix = phase == j
            count_b = ols['beta'][1, k] + (ols['beta'][P + j - 1, k] if j else 0)
            any_b = ols['beta'][1, K + k] + (ols['beta'][P + j - 1, K + k] if j else 0)
            rows.append(dict(channel=ch, phase=j + 1, start=str(pd.to_datetime(d.date[ix]).min().date()), end=str(pd.to_datetime(d.date[ix]).max().date()), n=int(ix.sum()), sources=int(np.unique(s[ix]).size), cells=int(np.unique(c[ix]).size), count_OLS=count_b, any_OLS=any_b, PPML_RR=float(np.exp(slopes[j])), PPML_mean0=float(m0[ix].mean()), PPML_mean1=float(m1[ix].mean()), PPML_delta=float((m1[ix] - m0[ix]).mean())))
        audit.append(dict(channel=ch, score=fit['gradient'], condition=fit['condition'], zero_cells=fit['zero_cells'], separation=poisson_separation_certificate(Z, Y[:, k], c)))
    save(rows, 'time_slopes')
    (ROOT / 'audit' / f'{PREFIX}_time.json').write_text(json.dumps(dict(calendar_rule='Four consecutive 50-day windows from the first included day, fixed before estimating these interactions; no outcome/event selection.', nuisance_slopes='Common across windows; only hostility varies. Same source-day effects and full sample.', fits=audit), indent=2))
    print('time slopes complete', flush=True)

def two_part(X, Y, s, t, c, masks):
    H = X[:, 1]
    C = c.max() + 1
    rows = []
    audits = []
    identityrows = []
    boundaryrows = []
    full = cell_ols(X, np.column_stack([Y, (Y > 0).astype(float), Y - (Y > 0)]), c)
    K = Y.shape[1]
    for k, ch in enumerate(CHANNEL):
        counts = ag((Y[:, k] > 0).astype(float), c * 2 + H.astype(int), 2 * C).reshape(C, 2)
        valid = (counts.min(1) > 0)[c]
        Xv = X[valid]
        yv = Y[valid, k]
        cv = codes(c[valid])
        hv = Xv[:, 1]
        f = logistic_profile(Xv, (yv > 0).astype(float), cv)
        pos = yv > 0
        cp = codes(cv[pos])
        assert len(np.unique(cp)) == len(np.unique(cv))
        keep = independent_columns(Xv[pos], cp, priority=1)
        Xp = Xv[pos][:, keep]
        excess = yv[pos] - 1
        g = ppml(Xp, excess, cp)
        separation = poisson_separation_certificate(Xp, excess, cp)
        bfull = np.zeros(X.shape[1])
        bfull[keep] = g['beta']
        eta = Xv @ bfull
        eta_pos = eta[pos]
        mx = np.full(cv.max() + 1, -np.inf)
        np.maximum.at(mx, cp, eta_pos)
        den = ag(np.exp(eta_pos - mx[cp]), cp)
        tot = ag(excess, cp)
        alpha = np.divide(tot, den, out=np.zeros_like(tot), where=den > 0)
        e0 = alpha[cv] * np.exp(eta - mx[cv] - hv * bfull[1])
        e1 = e0 * np.exp(bfull[1])
        m0 = 1 + e0
        m1 = 1 + e1
        p0 = f['p0']
        p1 = f['p1']
        assert np.all(np.isfinite(e0)) and np.all(np.isfinite(e1))
        assert m0.min() >= 1 and m1.min() >= 1 and (p0.min() >= 0) and (p1.min() >= 0) and (p0.max() <= 1) and (p1.max() <= 1)
        pred0 = p0 * m0
        pred1 = p1 * m1
        occ = 0.5 * (m0 + m1) * (p1 - p0)
        positive = 0.5 * (p0 + p1) * (m1 - m0)
        delta = pred1 - pred0
        err = float(abs(delta - occ - positive).max())
        assert err < 1e-08
        for label, sub in [('common_positive_support', np.ones(len(yv), bool)), ('common_positive_and_binary_NN05', masks['same_binary_and_NN_distance_le_05'][valid])]:
            rows.append(dict(channel=ch, population=label, n=int(sub.sum()), fraction_full_posts=float(sub.sum() / len(X)), sources=int(np.unique(s[valid][sub]).size), cells=int(np.unique(cv[sub]).size), adjusted_mean0=float(pred0[sub].mean()), adjusted_mean1=float(pred1[sub].mean()), adjusted_difference=float(delta[sub].mean()), occurrence_component=float(occ[sub].mean()), positive_count_component=float(positive[sub].mean()), average_occurrence0=float(p0[sub].mean()), average_occurrence1=float(p1[sub].mean()), average_positive_mean0=float(m0[sub].mean()), average_positive_mean1=float(m1[sub].mean()), original_observed_action_fraction_retained=float(yv[sub].sum() / Y[:, k].sum()), fitted_reference_weight_retained_within_common=float(pred0[sub].sum() / pred0.sum()), fit_rows=len(yv), support_restriction_refitted=False, max_identity_error=err, mean_identity_error=float(abs(delta[sub].mean() - occ[sub].mean() - positive[sub].mean()))))
        identityrows.append(dict(channel=ch, full_count_coefficient=float(full['beta'][1, k]), first_action_coefficient=float(full['beta'][1, K + k]), excess_beyond_first_coefficient=float(full['beta'][1, 2 * K + k]), identity_error=float(full['beta'][1, k] - full['beta'][1, K + k] - full['beta'][1, 2 * K + k])))
        nt = np.bincount(cv)
        positive_by_cell = ag(pos.astype(float), cv)
        one_cells = positive_by_cell == nt
        boundaryrows.append(dict(channel=ch, full_posts=len(X), full_cells=int(C), included_posts=len(yv), excluded_posts=int((~valid).sum()), included_cells=int(len(nt)), excluded_cells=int(C - len(nt)), included_sources=int(np.unique(s[valid]).size), positive_fit_rows=int(pos.sum()), occurrence_boundary_one_cells=int(one_cells.sum()), occurrence_boundary_one_posts=int(nt[one_cells].sum()), occurrence_boundary_zero_cells=int((positive_by_cell == 0).sum()), positive_excess_boundary_zero_cells=int((tot == 0).sum()), positive_excess_boundary_zero_standardization_posts=int(nt[tot == 0].sum()), positive_excess_boundary_zero_positive_fit_rows=int(np.sum((tot == 0)[cp])), observed_action_fraction_retained=float(yv.sum() / Y[:, k].sum()), max_post_identity_error=err, min_fitted_p=float(min(p0.min(), p1.min())), max_fitted_p=float(max(p0.max(), p1.max())), min_fitted_positive_mean=float(min(m0.min(), m1.min()))))
        audits.append(dict(channel=ch, fit_posts=len(yv), excluded_posts=int((~valid).sum()), logit={key: value for key, value in f.items() if key not in ['beta', 'p0', 'p1']}, positive_excess_ppml=dict(beta_H=float(bfull[1]), gradient=g['gradient'], condition=g['condition'], zero_excess_cells=g['zero_cells'], dropped_controls=X.shape[1] - len(keep), separation=separation), identity_max_error=err))
        print('two part', ch, 'n', len(yv), 'delta', float(delta.mean()), 'occ', float(occ.mean()), 'positive', float(positive.mean()), flush=True)
        save(rows, 'adjusted_two_part')
        save(identityrows, 'first_action_identity')
        save(boundaryrows, 'two_part_boundaries')
        (ROOT / 'audit' / f'{PREFIX}_two_part.json').write_text(json.dumps(dict(fits=audits, interpretation='Point summaries of a separately defined joint adjusted model, not decomposition of the full-release OLS or single PPML parameter. Positive-count support selects outcomes; populations differ across channels. Boundary occurrence0/1 and zero positive-excess cells are fitted boundary assumptions, not known structural population zeros. NN0.5 row restriction changes standardization only and does not refit; no causal attribution or validated confidence intervals.'), indent=2))
    return rows

def support_refits(X, Y, s, t, c):
    z = np.load(ROOT / 'results' / f'{PREFIX}_support_arrays.npz')
    masks = {'full_release': np.ones(len(X), bool), 'same_binary_profile': z['same_binary'], 'same_binary_NN05': z['distance'] <= 0.5, 'same_binary_NN1': z['distance'] <= 1}
    rows = []
    audits = []
    for rule, mask in masks.items():
        xx = X[mask]
        yy = Y[mask]
        cc = codes(c[mask])
        ss = s[mask]
        K = yy.shape[1]
        keep = independent_columns(xx, cc, priority=1)
        assert len(keep) == X.shape[1], 'Support refit design lost nuisance rank'
        fit = cell_ols(xx, np.column_stack([yy, (yy > 0).astype(float)]), cc)
        for k, ch in enumerate(CHANNEL):
            g = ppml(xx, yy[:, k], cc)
            h = xx[:, 1]
            m0 = g['mu'] * np.exp(-h * g['beta'][1])
            m1 = m0 * np.exp(g['beta'][1])
            cert = poisson_separation_certificate(xx, yy[:, k], cc)
            rows.append(dict(support_rule=rule, channel=ch, n=len(xx), fraction_full_posts=float(mask.mean()), sources=int(np.unique(ss).size), cells=int(cc.max() + 1), retained_observed_action_fraction=float(yy[:, k].sum() / Y[:, k].sum()), any_OLS=float(fit['beta'][1, K + k]), count_OLS=float(fit['beta'][1, k]), PPML_beta_H=float(g['beta'][1]), PPML_RR=float(np.exp(g['beta'][1])), PPML_mean0=float(m0.mean()), PPML_mean1=float(m1.mean()), PPML_delta=float((m1 - m0).mean()), PPML_zero_cells=g['zero_cells'], PPML_gradient=g['gradient'], no_additional_slope_separation_certified=cert['no_additional_slope_separation_certified']))
            audits.append(dict(support_rule=rule, channel=ch, condition=g['condition'], iterations=g['iterations'], separation=cert))
            save(rows, 'support_refits')
            (ROOT / 'audit' / f'{PREFIX}_support_refits.json').write_text(json.dumps(dict(fits=audits, interpretation='Each row refits both linear projections and PPML on the indicated outcome-independent measured-X support subset. Covariate SD scaling is fixed from the original full release; sample, cells, coefficients, and standardization population change. These refits are separate from the two-part NN05 rows, which restrict the reference population without refitting. No validated confidence intervals or causal interpretation.'), indent=2))
        print('support refits', rule, 'n', len(xx), 'complete', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['all', 'core', 'two_part', 'support_refits', 'validate'], default='all')
    args = parser.parse_args()
    spec = ROOT / 'audit' / f'{PREFIX}_specification.json'
    if not spec.exists():
        spec.write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='Exploratory after review; choices written before these new fits.', choices=['Binary support exactly source-day, solidarity, group mentions, media and URL; continuous opposite-label nearest distance thresholds0.25/0.5/1 full-release SD units; H propensity bands0.05–0.95 and0.1–0.9 are diagnostics.', 'Four consecutive50-day windows from first date; common nuisance slopes with H interactions.', 'Positive both-H source-day support separately per channel; source-day logit of occurrence and PPML of positive excessY−1, m=1+mu. Symmetric standardized decomposition; point summaries only.', 'Tail action concentrations use unchanged total-action cutoffs0.95/0.99/0.999. Source weights use unchanged operationally finite common71sources.']), indent=2))
    validation = validate()
    if args.part == 'validate':
        print(json.dumps(validation))
        return
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    Y = d[CHANNEL].to_numpy(float)
    s, t, c = grouping(d, 'facebook', 'date')
    if args.part in ['all', 'core']:
        masks = support(d, X, Y, s, t, c)
        tails(Y)
        source_weights()
        time_slopes(X, Y, s, t, c, d)
    else:
        z = np.load(ROOT / 'results' / f'{PREFIX}_support_arrays.npz')
        masks = {'same_binary_and_NN_distance_le_05': z['distance'] <= 0.5}
    if args.part in ['all', 'two_part']:
        two_part(X, Y, s, t, c, masks)
    if args.part in ['all', 'support_refits']:
        support_refits(X, Y, s, t, c)
    print('R6 margins work complete', flush=True)
if __name__ == '__main__':
    main()
