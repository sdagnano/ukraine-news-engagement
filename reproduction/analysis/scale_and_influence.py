from cell_models import *

def reml(M, Y, g):
    n, p = M.shape
    G = g.max() + 1
    ng = np.bincount(g)
    sx = ag(M, g)
    sy = ag(Y, g)
    XX = M.T @ M
    XY = M.T @ Y
    YY = np.sum(Y * Y, axis=0)
    bs = []
    weights = []
    aud = []
    for k in range(Y.shape[1]):

        def calculate(loglam, full=False):
            lam = np.exp(loglam)
            v = lam / (1 + lam * ng)
            S = XX - sx.T @ (v[:, None] * sx)
            R = XY[:, k] - sx.T @ (v * sy[:, k])
            B = linalg.inv(S)
            b = B @ R
            q = YY[k] - np.sum(v * sy[:, k] ** 2) - R @ b
            obj = (n - p) * np.log(q / (n - p)) + np.log1p(lam * ng).sum() + np.linalg.slogdet(S)[1]
            return (obj, b, B, v, q / (n - p)) if full else obj
        result = optimize.minimize_scalar(calculate, bounds=(-20, 20), method='bounded', options={'xatol': 1e-08})
        obj, b, B, v, sigma = calculate(result.x, True)
        w = (M - v[g, None] * sx[g]) @ B[:, 2]
        bs.append(b[2])
        weights.append(w * (Y[:, k] - M @ b))
        aud.append(dict(lambda_ratio=float(np.exp(result.x)), residual_variance=float(sigma), optimizer_success=bool(result.success)))
    return (np.array(bs), np.array(weights).T, aud)

def main():
    bridge = []
    tail = []
    ext = []
    sources = []
    posts = []
    pairs = []
    account = []
    audits = []
    for platform in ['facebook', 'twitter']:
        d = load(platform)
        X = ukrainian_design(d).to_numpy()
        names = CHANNELS[platform]
        Y = d[names].to_numpy(float)
        tot = Y.sum(1)
        s, t, c = grouping(d, platform, 'date')
        G = s.max() + 1
        K = len(names)
        Z = np.column_stack([np.log1p(tot), tot])
        b, obs, audit = reml(np.column_stack([np.ones(len(d)), X]), Z, s)
        for k, target in enumerate(['log1p_total', 'count_total']):
            V = covariance(obs[:, k, None], s, X.shape[1] + 1)
            se = np.sqrt(V[0, 0])
            q = stats.t.ppf(0.975, G - 1)
            bridge.append(dict(platform=platform, design='random_intercept', target=target, beta=b[k], se=se, low=b[k] - q * se, high=b[k] + q * se, n=len(d)))
        for design in ['source_only', 'source_month', 'source_date', 'source_week_cell', 'source_day_cell']:
            if design in ['source_month', 'source_date']:
                tm = codes(pd.to_datetime(d.date).dt.to_period('M')) if design == 'source_month' else t
                a = sufficient(X, Z, s, tm)
                bb = a['b'][1]
                oi = obs_influence(a, 1)
                rank = a['rank']
            else:
                cc = s if design == 'source_only' else grouping(d, platform, 'week')[2] if design == 'source_week_cell' else c
                a = cell_ols(X, Z, cc)
                bb = a['beta'][1]
                oi = a['obs']
                rank = a['rank']
            V = multi_cov(oi, s, t, rank)['two_way']
            se = np.sqrt(np.diag(V))
            q = stats.t.ppf(0.975, G - 1)
            for k, target in enumerate(['log1p_total', 'count_total']):
                bridge.append(dict(platform=platform, design=design, target=target, beta=bb[k], se=se[k], low=bb[k] - q * se[k], high=bb[k] + q * se[k], n=len(d)))
        a = cell_ols(X, np.column_stack([Y, np.log1p(tot)]), c)
        w = a['w']
        bb = a['beta'][1]
        p = X.shape[1]
        for quantile in [0.95, 0.99, 0.999]:
            cutoff = np.quantile(tot, quantile)
            ix = tot > cutoff
            for k, ch in enumerate(names + ['log1p_total']):
                response = np.log1p(tot) if k == K else Y[:, k]
                upper = float(w[ix] @ response[ix])
                lower = float(w[~ix] @ response[~ix])
                b = float(w @ response)
                assert abs(upper + lower - b) < 1e-08
                tail.append(dict(platform=platform, channel=ch, quantile=quantile, cutoff=cutoff, tail_posts=int(ix.sum()), beta=b, tail_contribution=upper, body_contribution=lower, signed_tail_fraction=upper / b if abs(b) > 1e-08 else np.nan))
        for k, ch in enumerate(names):
            ix = Y[:, k] > 0
            ci = codes(c[ix])
            ni = np.bincount(ci)
            xi = X[ix]
            xw = xi - ag(xi, ci)[ci] / ni[ci, None]
            nuisance = [j for j in range(X.shape[1]) if j != 1]
            Q, R, piv = linalg.qr(xw[:, nuisance], mode='economic', pivoting=True)
            r = np.linalg.matrix_rank(xw[:, nuisance])
            keep = sorted([1] + [nuisance[j] for j in piv[:r]])
            if np.linalg.matrix_rank(xw[:, keep]) <= r:
                raise RuntimeError('Positive-only hostility not identified')
            jh = keep.index(1)
            aa = cell_ols(xi[:, keep], Y[ix, k, None], ci, j=jh)
            ss = codes(s[ix])
            tt = t[ix]
            for width, V, se, df, q in inference(aa['obs'], ss, tt, aa['rank'], K):
                ext.append(dict(platform=platform, channel=ch, design='source_day_positive_only', target='conditional_positive_count', n=int(ix.sum()), cells=int(np.unique(c[ix]).size), dropped_nuisance=X.shape[1] - len(keep), block=width, beta=aa['beta'][jh, 0], se=se[0], low=aa['beta'][jh, 0] - q * se[0], high=aa['beta'][jh, 0] + q * se[0], df=df))
        contrib = w[:, None] * Y
        influence = a['obs'][:, :K]
        source_contrib = ag(contrib, s)
        source_score = ag(influence, s)
        for sg in range(G):
            for k, ch in enumerate(names):
                sources.append(dict(platform=platform, source=sg, channel=ch, n=int((s == sg).sum()), coefficient_contribution=source_contrib[sg, k], influence_contribution=source_score[sg, k]))
        leverage = 1 / a['cell_counts'][c] + np.einsum('ij,ij->i', a['x'] @ a['bread'], a['x'])
        dff = np.divide(-influence, 1 - leverage[:, None], out=np.zeros_like(influence), where=1 - leverage[:, None] > 1e-10)
        for k, ch in enumerate(names):
            for idx in np.argsort(-abs(dff[:, k]))[:30]:
                posts.append(dict(platform=platform, channel=ch, row=int(idx), source=int(s[idx]), date=str(d.date.iloc[idx]), H=int(X[idx, 1]), outcome=Y[idx, k], coefficient_contribution=contrib[idx, k], DFBETA=dff[idx, k], leverage=leverage[idx]))
        Sg = []
        Rg = []
        for sg in range(G):
            ix = s == sg
            Sg.append(a['x'][ix].T @ a['x'][ix])
            Rg.append(a['x'][ix].T @ a['y'][ix, :K])
        Sg = np.array(Sg)
        Rg = np.array(Rg)
        S = Sg.sum(0)
        R = Rg.sum(0)
        for sg in range(G):
            for hh in range(sg):
                bp = linalg.solve(S - Sg[sg] - Sg[hh], R - Rg[sg] - Rg[hh], assume_a='pos')[1]
                for k, ch in enumerate(names):
                    pairs.append(dict(platform=platform, source1=hh, source2=sg, channel=ch, beta=bp[k]))
        hc = c * 2 + X[:, 1].astype(int)
        N = ag(np.ones(len(d)), hc, 2 * (c.max() + 1)).reshape(-1, 2)
        sums = ag(Y, hc, 2 * len(N)).reshape(-1, 2, K)
        positive = ag((Y > 0).astype(float), hc, 2 * len(N)).reshape(-1, 2, K)
        both = N.min(1) > 0
        weights = N.sum(1)[both]
        weights /= weights.sum()
        for k, ch in enumerate(names):
            n = N[both]
            pos = positive[both, :, k]
            sm = sums[both, :, k]
            prob = pos / n
            mean = sm / n
            valid = pos.min(1) > 0
            aa = np.divide(sm, pos, out=np.zeros_like(sm), where=pos > 0)
            e = 0.5 * (aa[:, 1] + aa[:, 0]) * (prob[:, 1] - prob[:, 0])
            ii = 0.5 * (prob[:, 1] + prob[:, 0]) * (aa[:, 1] - aa[:, 0])
            diff = mean[:, 1] - mean[:, 0]
            activation = float(weights[valid] @ e[valid])
            intensity = float(weights[valid] @ ii[valid])
            unresolved = float(weights[~valid] @ diff[~valid])
            delta = float(weights @ diff)
            assert abs(activation + intensity + unresolved - delta) < 1e-09
            account.append(dict(platform=platform, channel=ch, target='pooled_post_weighted_common_source_day_raw_means', posts=int(N[both].sum()), delta=delta, activation=activation, intensity=intensity, unsupported_positive_margin=unresolved, both_positive_weight=float(weights[valid].sum()), unsupported_weight=float(weights[~valid].sum())))
        dates = pd.to_datetime(d.date)
        audits.append(dict(platform=platform, released_window=len(d), deposited_script_strict_before_12Sep=int((dates < pd.Timestamp('2022-09-12')).sum()), Sep12_posts=int((dates == pd.Timestamp('2022-09-12')).sum()), max_released_log_difference=float(abs(d.engagement_log - np.log1p(tot)).max()), missing_original_controls=[z for z in ['total_tokens', 'Followers.at.Posting', 'followers_count'] if z not in d], REML=audit))
        pd.DataFrame(bridge).to_csv(ROOT / 'results/scale_bridge.csv', index=False)
        pd.DataFrame(tail).to_csv(ROOT / 'results/tail_contributions.csv', index=False)
        pd.DataFrame(ext).to_csv(ROOT / 'results/intensive_projections.csv', index=False)
        pd.DataFrame(sources).to_csv(ROOT / 'results/source_contributions.csv', index=False)
        pd.DataFrame(posts).to_csv(ROOT / 'results/post_influence.csv', index=False)
        pd.DataFrame(pairs).to_csv(ROOT / 'results/all_source_pair_deletions.csv', index=False)
        pd.DataFrame(account).to_csv(ROOT / 'results/two_part_accounting.csv', index=False)
        (ROOT / 'audit/scale_and_sample.json').write_text(json.dumps(audits, indent=2))
        print(platform, 'scale/margins/influence complete', flush=True)
if __name__ == '__main__':
    main()
