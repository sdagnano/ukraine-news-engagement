from robust_models import *

def ag(a, g, G=None):
    a = np.asarray(a)
    G = int(g.max() + 1) if G is None else int(G)
    if a.ndim == 1:
        return np.bincount(g, weights=a, minlength=G)
    return np.column_stack([np.bincount(g, weights=a[:, j], minlength=G) for j in range(a.shape[1])])

def grouping(d, platform, scale):
    s = codes(d['Page.Name' if platform == 'facebook' else 'user_id'])
    date = pd.to_datetime(d.date)
    t = codes(date.dt.to_period('W').astype(str)) if scale == 'week' else codes(date)
    cell = codes(pd.MultiIndex.from_arrays([s, t]))
    return (s, codes(date), cell)

def cell_ols(X, Y, cell, j=1):
    n, p = X.shape
    ng = np.bincount(cell)
    G = len(ng)
    x = X - ag(X, cell, G)[cell] / ng[cell, None]
    y = Y - ag(Y, cell, G)[cell] / ng[cell, None]
    B = linalg.inv(x.T @ x)
    b = B @ x.T @ y
    res = y - x @ b
    w = x @ B[:, j]
    return dict(beta=b, bread=B, x=x, y=y, res=res, w=w, obs=w[:, None] * res, rank=G + p, cell_counts=ng)

def ppml(X, y, cell, start=None, omit_source=None, source=None, tol=2e-10, maxiter=60):
    n, p = X.shape
    C = int(cell.max() + 1)
    active = np.ones(n) if omit_source is None else (source != omit_source).astype(float)
    yy = y * active
    yt = ag(yy, cell, C)
    total = yy.sum()
    yx = X.T @ yy
    b = np.zeros(p) if start is None else start.copy()
    if total <= 0:
        raise RuntimeError('No positive response')

    def ev(beta, hess=False):
        eta = X @ beta
        maximum = np.full(C, -np.inf)
        np.maximum.at(maximum, cell, eta)
        ee = np.exp(eta - maximum[cell])
        den = ag(ee, cell, C)
        mu = yt[cell] * ee / den[cell]
        obj = (yt @ (maximum + np.log(den)) - yx @ beta) / total
        grad = (X.T @ mu - yx) / total
        if not hess:
            return (obj, grad, mu, None, None)
        WX = ag(mu[:, None] * X, cell, C)
        mean = np.divide(WX, yt[:, None], out=np.zeros_like(WX), where=yt[:, None] > 0)
        H = X.T @ (mu[:, None] * X) - mean.T @ (yt[:, None] * mean)
        return (obj, grad, mu, H, mean)
    for it in range(maxiter):
        obj, grad, mu, H, mean = ev(b, True)
        if np.max(abs(grad)) < tol:
            break
        try:
            step = linalg.solve(H, grad * total, assume_a='pos')
        except linalg.LinAlgError:
            raise RuntimeError('Singular information / possible separation')
        fac = 1.0
        for ls in range(35):
            ob, gr, _, _, _ = ev(b - fac * step)
            if np.isfinite(ob) and (ob < obj - 0.0001 * fac * (grad @ step) or (ob <= obj + 1e-12 and abs(gr).max() < abs(grad).max())):
                break
            fac *= 0.5
        else:
            raise RuntimeError('Poisson line search failure')
        b -= fac * step
        if np.max(abs(b)) > 40:
            raise RuntimeError('Divergent coefficient / possible separation')
    obj, grad, mu, H, mean = ev(b, True)
    if abs(grad).max() > max(2e-08, tol * 10):
        raise RuntimeError('Poisson score did not converge')
    eig = np.linalg.eigvalsh(H)
    if eig.min() <= eig.max() * 1e-13:
        raise RuntimeError('Nonidentified information / possible separation')
    return dict(beta=b, mu=mu, bread=linalg.inv(H), mean=mean, yt=yt, active=active, n=int(active.sum()), gradient=float(abs(grad).max()), iterations=it, zero_cells=int((yt == 0).sum()), positive_cells=int((yt > 0).sum()), condition=float(eig.max() / eig.min()))

def functional(f, X, y, cell, A=None):
    n = f['n']
    p = X.shape[1]
    h = X[:, 1]
    b = f['beta']
    mu = f['mu']
    C = len(f['yt'])
    if A is None:
        A = np.zeros((len(X), p))
        A[:, 1] = 1
    slope = A @ b
    rr = np.exp(slope)
    m0i = mu * np.exp(-h * slope)
    m1i = m0i * rr
    delta_i = m1i - m0i
    vals = np.array([m0i.sum() / n, m1i.sum() / n, delta_i.sum() / n])
    gradients = []
    cellgrads = []
    for u, v in [(m0i, -h * m0i), (m1i, (1 - h) * m1i), (delta_i, m1i - h * delta_i)]:
        gradients.append((X.T @ u + A.T @ v) / n)
        cellgrads.append(ag(u, cell, C) / n)
    obs = []
    for gb, ga in zip(gradients, cellgrads):
        eff = gb - f['mean'].T @ ga
        q = f['bread'] @ eff
        intercept = np.divide(ga, f['yt'], out=np.zeros(C), where=f['yt'] > 0)
        obs.append((y - mu) * f['active'] * (X @ q - (f['mean'] @ q)[cell] + intercept[cell]))
    obs = np.array(obs).T
    logr = obs[:, 1] / vals[1] - obs[:, 0] / vals[0]
    return (dict(mean0=vals[0], mean1=vals[1], delta=vals[2], RR=vals[1] / vals[0]), obs, logr)

def inference(obs, source, date, rank, family):
    rows = []
    days = (date - date.min()).astype(int)
    for width in [1, 7, 14, 28]:
        t = codes(days // width)
        V = multi_cov(obs, source, t, rank)['two_way']
        df = min(source.max(), t.max())
        se = np.sqrt(np.diag(V)) if np.all(np.diag(V) > 0) else np.full(obs.shape[1], np.nan)
        rows.append((width, V, se, float(df), stats.t.ppf(1 - 0.05 / (2 * family), df)))
    return rows

def main():
    olsrows = []
    prows = []
    audits = []
    coeff = {}
    support = []
    joints = []
    for platform in ['facebook', 'twitter']:
        d = load(platform)
        X = ukrainian_design(d).to_numpy()
        names = CHANNELS[platform]
        Y = d[names].to_numpy(float)
        for scale, restricted in [('week', False), ('date', False), ('date', True)]:
            s, t, c = grouping(d, platform, scale)
            mask = np.ones(len(d), dtype=bool)
            C = c.max() + 1
            cnt = np.zeros((C, 2))
            np.add.at(cnt, (c, X[:, 1].astype(int)), 1)
            both = cnt.min(1) > 0
            if restricted:
                mask = both[c]
            xx = X[mask]
            yy = Y[mask]
            ss = codes(s[mask])
            tt = codes(t[mask])
            cc = codes(c[mask])
            tag = scale + ('_both' if restricted else '_all')
            support.append(dict(platform=platform, design=tag, posts=len(xx), sources=int(ss.max() + 1), cells=int(cc.max() + 1), both_label_cells=int(both.sum()), both_label_posts=int(both[c].sum())))
            z = cell_ols(xx, np.column_stack([yy, (yy > 0).astype(float)]), cc)
            K = len(names)
            for target, kk in [('count', slice(0, K)), ('any', slice(K, 2 * K))]:
                for width, V, se, df, q in inference(z['obs'][:, kk], ss, tt, z['rank'], K):
                    for k, ch in enumerate(names):
                        b = z['beta'][1, kk][k]
                        olsrows.append(dict(platform=platform, design=tag, target=target, channel=ch, n=len(xx), block=width, beta=b, se=se[k], df=df, low=b - q * se[k], high=b + q * se[k]))
            D = []
            LL = []
            pv = []
            for k, ch in enumerate(names):
                f = ppml(xx, yy[:, k], cc)
                v, obs, lr = functional(f, xx, yy[:, k], cc)
                pv.append(v)
                D.append(obs[:, 2])
                LL.append(lr)
                coeff[platform + '_' + tag + '_' + ch] = f['beta']
                for width, V, se, df, q in inference(np.column_stack([obs[:, 2], lr]), ss, tt, z['rank'], K):
                    prows.append(dict(platform=platform, design=tag, channel=ch, n=len(xx), block=width, **v, delta_se=se[0], logRR_se=se[1], logRR_low=np.log(v['RR']) - q * se[1], logRR_high=np.log(v['RR']) + q * se[1], delta_low=v['delta'] - stats.t.ppf(0.975, df) * se[0], delta_high=v['delta'] + stats.t.ppf(0.975, df) * se[0], df=df))
                audits.append(dict(platform=platform, design=tag, channel=ch, n=len(xx), gradient=f['gradient'], iterations=f['iterations'], zero_cells=f['zero_cells'], positive_cells=f['positive_cells'], condition=f['condition']))
            DD = np.array(D).T
            LL = np.array(LL).T
            vals = np.array([[v['mean0'], v['mean1']] for v in pv])
            delta = sum((v['delta'] for v in pv))
            for width, VD, se, df, q in inference(DD, ss, tt, z['rank'], K):
                tblock = codes(tt // width)
                VL = multi_cov(LL, ss, tblock, z['rank'])['two_way']
                a = np.zeros(K)
                a[-1] = 1
                a[0] = -1
                theta = pv[-1]['RR'] / pv[0]['RR']
                st = np.sqrt(a @ VL @ a)
                sd = np.sqrt(VD.sum())
                crit = stats.t.ppf(0.975, df)
                joints.append(dict(platform=platform, design=tag, block=width, net=delta, net_low=delta - crit * sd, net_high=delta + crit * sd, ratio=theta, ratio_low=np.exp(np.log(theta) - crit * st), ratio_high=np.exp(np.log(theta) + crit * st), df=df))
            pd.DataFrame(olsrows).to_csv(ROOT / 'results/cell_OLS.csv', index=False)
            pd.DataFrame(prows).to_csv(ROOT / 'results/cell_PPML.csv', index=False)
            pd.DataFrame(support).to_csv(ROOT / 'results/cell_support.csv', index=False)
            pd.DataFrame(joints).to_csv(ROOT / 'results/cell_joint.csv', index=False)
            np.savez_compressed(ROOT / 'results/cell_model_parameters.npz', **coeff)
            (ROOT / 'audit/cell_models.json').write_text(json.dumps(audits, indent=2))
            print(platform, tag, 'complete', flush=True)
if __name__ == '__main__':
    main()
