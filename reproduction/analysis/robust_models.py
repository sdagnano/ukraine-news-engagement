from legacy_common import *
import pyreadr
CHANNELS = {'facebook': ['Likes', 'Love', 'Angry', 'Haha', 'Sad', 'Wow', 'Comments', 'Shares'], 'twitter': ['favorite_count', 'retweet_count']}

def load(platform):
    f = 'fb_proua_full_osf.RDS' if platform == 'facebook' else 'tw_proua_full_osf.RDS'
    d = next(iter(pyreadr.read_r(str(ROOT / 'input' / f)).values()))
    t = pd.to_datetime(d.date)
    return d.loc[(t > pd.Timestamp('2022-02-24')) & (t < pd.Timestamp('2022-09-13'))].reset_index(drop=True)

def agg(a, g, G):
    a = np.asarray(a)
    one = a.ndim == 1
    if one:
        a = a[:, None]
    z = np.zeros((G, a.shape[1]))
    np.add.at(z, g, a)
    return z[:, 0] if one else z

def sufficient(X, Y, g, t):
    n, p = X.shape
    G = g.max() + 1
    T = t.max() + 1
    K = Y.shape[1]
    ng = np.bincount(g, minlength=G)
    xm = agg(X, g, G) / ng[:, None]
    ym = agg(Y, g, G) / ng[:, None]
    x = X - xm[g]
    y = Y - ym[g]
    P = p + T - 1
    C = np.zeros((G, T))
    np.add.at(C, (g, t), 1)
    Sg = np.zeros((G, P, P))
    Rg = np.zeros((G, P, K))
    for s in range(G):
        ix = g == s
        xs = x[ix]
        ys = y[ix]
        tt = t[ix]
        c = C[s]
        cross = agg(xs, tt, T).T[:, 1:]
        Sg[s, :p, :p] = xs.T @ xs
        Sg[s, :p, p:] = cross
        Sg[s, p:, :p] = cross.T
        Sg[s, p:, p:] = (np.diag(c) - np.outer(c, c) / ng[s])[1:, 1:]
        Rg[s, :p] = xs.T @ ys
        Rg[s, p:] = agg(ys, tt, T)[1:]
    S = Sg.sum(0)
    R = Rg.sum(0)
    B = linalg.inv(S)
    b = B @ R
    return dict(Sg=Sg, Rg=Rg, S=S, R=R, B=B, b=b, x=x, y=y, C=C, ng=ng, g=g, t=t, p=p, n=n, G=G, T=T, rank=G + P)

def obs_influence(a, j):
    p = a['p']
    t = a['t']
    g = a['g']
    C = a['C']
    ng = a['ng']
    b = a['b']
    B = a['B']
    dt = np.vstack([np.zeros((1, b.shape[1])), b[p:]])
    res = a['y'] - a['x'] @ b[:p] - dt[t] + (C @ dt / ng[:, None])[g]
    db = np.r_[0.0, B[p:, j]]
    w = a['x'] @ B[:p, j] + db[t] - (C @ db / ng)[g]
    return res * w[:, None]

def robust(a, j=1, loo=True):
    Sg, Rg, S, R, B, b = [a[k] for k in ['Sg', 'Rg', 'S', 'R', 'B', 'b']]
    G = a['G']
    K = b.shape[1]
    scores = Rg - Sg @ b
    deleted = np.array([linalg.solve(S - Sg[s], R - Rg[s], assume_a='pos')[j] for s in range(G)])
    diff = deleted - b[j]
    cv3 = (G - 1) / G * (diff.T @ diff)
    centered = deleted - deleted.mean(0)
    cv3j = (G - 1) / G * (centered.T @ centered)
    ev, U = linalg.eigh(B)
    C = U * np.sqrt(ev) @ U.T
    infl = []
    theta = []
    mineig = 1.0
    for s in range(G):
        ev, Q = linalg.eigh(np.eye(len(B)) - C @ Sg[s] @ C)
        mineig = min(mineig, float(ev.min()))
        if ev.min() < 1e-09:
            raise RuntimeError('CR2 singular deletion leverage')
        A = Q * (1 / np.sqrt(ev)) @ Q.T
        F = C @ A @ C
        infl.append(F[j] @ scores[s])
        theta.append(F[:, j])
    infl = np.array(infl)
    theta = np.array(theta)
    V2 = infl.T @ infl
    aa = np.einsum('gij,gj->gi', Sg, theta)
    dd = np.einsum('gi,gi->g', theta, aa)
    Gram = np.diag(dd) - aa @ B @ aa.T
    df2 = float(np.trace(Gram) ** 2 / np.sum(Gram ** 2))
    oi = obs_influence(a, j)
    cov = multi_cov(oi, a['g'], a.get('cluster_time', a['t']), a['rank'])
    result = dict(beta=b[j], CR2=V2, CV3=cv3, CV3J=cv3j, CR1=cov['source'], two_way=cov['two_way'], df_CR2=df2, deleted=deleted, CR2_min_IminusH_eigenvalue=mineig, condition=float(np.linalg.cond(S)))
    if loo:
        double = np.full((G, G, K), np.nan)
        for s in range(G):
            for h in range(s):
                v = linalg.solve(S - Sg[s] - Sg[h], R - Rg[s] - Rg[h], assume_a='pos')[j]
                double[s, h] = v
                double[h, s] = v
        vv = []
        for s in range(G):
            v = double[s, np.arange(G) != s]
            v -= v.mean(0)
            vv.append((G - 2) / (G - 1) * np.sum(v * v, axis=0))
        result['deleted_CV3J_se'] = np.sqrt(vv)
    return result

def wild(a, j=1, draws=9999, seed=873214, weights='rademacher', batch=250):
    Sg, Rg, S, R, B, b = [a[k] for k in ['Sg', 'Rg', 'S', 'R', 'B', 'b']]
    G = a['G']
    K = b.shape[1]
    P = len(B)
    keep = np.arange(P) != j
    restricted = np.zeros_like(b)
    restricted[keep] = linalg.solve(S[np.ix_(keep, keep)], R[keep], assume_a='pos')
    sn = Rg - Sg @ restricted
    Q = np.array([linalg.solve(S - Sg[s], np.eye(P)[:, j], assume_a='pos') for s in range(G)])
    QS = np.einsum('gi,gij->gj', Q, Sg)
    qs = np.einsum('gi,gik->gk', Q, sn)
    deletion = Q @ R - np.einsum('gi,gik->gk', Q, Rg)
    se = np.sqrt((G - 1) / G * np.sum((deletion - b[j]) ** 2, axis=0))
    observed = np.abs(b[j] / se)
    rng = np.random.default_rng(seed)
    Ts = []
    points = np.array([-np.sqrt(1.5), -1, -np.sqrt(0.5), np.sqrt(0.5), 1, np.sqrt(1.5)])
    for start in range(0, draws, batch):
        m = min(batch, draws - start)
        w = rng.choice([-1.0, 1.0] if weights == 'rademacher' else points, size=(m, G))
        delta = np.einsum('ab,mg,bgk->mak', B, w, sn.transpose(1, 0, 2), optimize=True)
        r = w[:, :, None] * qs[None, :, :] - np.einsum('gp,mpk->mgk', QS, delta, optimize=True)
        sest = np.sqrt((G - 1) / G * np.sum(r * r, axis=1))
        Ts.append(np.abs(delta[:, j] / sest))
    T = np.concatenate(Ts)
    single = (1 + (T >= observed).sum(0)) / (draws + 1)
    maximum = (1 + (T.max(1)[:, None] >= observed).sum(0)) / (draws + 1)
    order = np.argsort(-observed)
    rw = np.empty(K)
    last = 0.0
    for ix, k in enumerate(order):
        v = (1 + np.sum(T[:, order[ix:]].max(1) >= observed[k])) / (draws + 1)
        last = max(last, float(v))
        rw[k] = last
    return dict(p_single=single, p_maxT=maximum, p_stepdown=rw, draws=draws, weights=weights, seed=seed, T_observed=observed)

def run_ols():
    rows = []
    deletions = []
    aud = []
    for platform in ['facebook', 'twitter']:
        d = load(platform)
        X = ukrainian_design(d)
        names = CHANNELS[platform]
        Y = d[names].to_numpy(float)
        g = codes(d['Page.Name' if platform == 'facebook' else 'user_id'])
        for time in ['month', 'date']:
            t = codes(pd.to_datetime(d.date).dt.to_period('M') if time == 'month' else d.date)
            print(platform, time, 'sufficient', flush=True)
            a = sufficient(X.to_numpy(), Y, g, t)
            a['cluster_time'] = codes(d.date)
            r = robust(a)
            K = len(names)
            G = a['G']
            boot = {w: wild(a, weights=w) for w in ['rademacher', 'webb']}
            for k, ch in enumerate(names):
                for method in ['CR1', 'two_way', 'CR2', 'CV3', 'CV3J']:
                    df = r['df_CR2'] if method == 'CR2' else min(G - 1, 199) if method == 'two_way' else G - 1
                    se = np.sqrt(r[method][k, k])
                    crit = stats.t.ppf(1 - 0.05 / (2 * K), df)
                    b = r['beta'][k]
                    rows.append(dict(platform=platform, time=time, channel=ch, method=method, beta=b, se=se, df=df, low=b - crit * se, high=b + crit * se, p_single=2 * stats.t.sf(abs(b / se), df), family=K))
                for w, v in boot.items():
                    rows.append(dict(platform=platform, time=time, channel=ch, method='wild_' + w, beta=r['beta'][k], p_single=v['p_single'][k], p_maxT=v['p_maxT'][k], p_stepdown=v['p_stepdown'][k], family=K))
                for s in range(G):
                    se = r['deleted_CV3J_se'][s, k]
                    crit = stats.t.ppf(1 - 0.05 / (2 * K), G - 2)
                    b = r['deleted'][s, k]
                    deletions.append(dict(platform=platform, time=time, channel=ch, source=s, beta=b, se_CV3J=se, low=b - crit * se, high=b + crit * se, family=K, df=G - 2))
            aud.append(dict(platform=platform, time=time, n=len(d), sources=G, dates=a['T'], design_columns=list(X.columns), df_CR2=r['df_CR2'], condition=r['condition'], CR2_min_IminusH_eigenvalue=r['CR2_min_IminusH_eigenvalue'], seed=873214, draws=9999))
            np.savez_compressed(ROOT / 'results' / f'ols_{platform}_{time}_covariances.npz', **{k: r[k] for k in ['beta', 'CR1', 'two_way', 'CR2', 'CV3', 'CV3J']})
            pd.DataFrame(rows).to_csv(ROOT / 'results/ols_inference.csv', index=False)
            pd.DataFrame(deletions).to_csv(ROOT / 'results/ols_deletion_inference.csv', index=False)
            (ROOT / 'audit/ols_models.json').write_text(json.dumps(aud, indent=2, default=lambda v: v.item()))
            print(platform, time, 'completed', flush=True)
    return rows
if __name__ == '__main__':
    run_ols()
