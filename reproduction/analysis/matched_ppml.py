from robust_models import *

def model(X, y, g, t, start=None, omit=None, tol=1e-10, maxiter=80):
    n, p = X.shape
    G = g.max() + 1
    T = t.max() + 1
    P = p + T - 1
    active = np.ones(n) if omit is None else (g != omit).astype(float)
    yy = y * active
    yt = agg(yy, g, G)
    total = yy.sum()
    b = np.zeros(P) if start is None else start.copy()
    yx = X.T @ yy
    yd = agg(yy, t, T)[1:]

    def evaluate(b, hessian=False):
        dt = np.r_[0.0, b[p:]]
        eta = X @ b[:p] + dt[t]
        mx = np.full(G, -np.inf)
        np.maximum.at(mx, g, eta)
        e = np.exp(eta - mx[g])
        den = agg(e, g, G)
        mu = yt[g] * e / den[g]
        val = (yt @ (mx + np.log(den)) - yy @ eta) / total
        grad = np.r_[X.T @ mu - yx, agg(mu, t, T)[1:] - yd] / total
        if not hessian:
            return (val, grad, mu, None, None)
        u = agg(mu[:, None] * X, g, G)
        c = np.zeros((G, T))
        np.add.at(c, (g, t), mu)
        mean = np.divide(np.column_stack([u, c[:, 1:]]), yt[:, None], out=np.zeros((G, P)), where=yt[:, None] > 0)
        H = np.zeros((P, P))
        H[:p, :p] = X.T @ (mu[:, None] * X)
        cross = agg(mu[:, None] * X, t, T).T[:, 1:]
        H[:p, p:] = cross
        H[p:, :p] = cross.T
        H[p:, p:] = np.diag(agg(mu, t, T)[1:])
        H -= mean.T @ (yt[:, None] * mean)
        return (val, grad, mu, H, mean)
    accepted = 0
    for it in range(maxiter):
        val, grad, mu, H, mean = evaluate(b, True)
        if abs(grad).max() < tol:
            break
        step = linalg.solve(H, grad * total, assume_a='pos')
        factor = 1.0
        for ls in range(30):
            vn, gn, mn, _, _ = evaluate(b - factor * step)
            if np.isfinite(vn) and (vn <= val - 0.0001 * factor * (grad @ step) or (vn <= val + 1e-12 and abs(gn).max() < abs(grad).max())):
                break
            factor *= 0.5
        else:
            raise RuntimeError('PPML line search failed')
        b -= factor * step
        accepted += 1
    val, grad, mu, H, mean = evaluate(b, True)
    if abs(grad).max() > max(2e-08, tol * 10):
        raise RuntimeError(f'PPML nonconvergence {abs(grad).max()}')
    return dict(beta=b, mu=mu, bread=linalg.inv(H), mean=mean, yt=yt, active=active, n=int(active.sum()), gradient=float(abs(grad).max()), iterations=accepted, zero_sources=int(np.sum(yt == 0)))

def functional(f, X, y, g, t, j=1):
    b = f['beta']
    mu = f['mu']
    p = X.shape[1]
    G = g.max() + 1
    T = t.max() + 1
    n = f['n']
    h = X[:, j]
    u0 = mu * np.exp(-b[j] * h)
    m0 = u0.sum() / n
    rr = np.exp(b[j])
    m1 = m0 * rr
    delta = m1 - m0
    base = np.r_[X.T @ u0, agg(u0, t, T)[1:]] / n
    base[j] -= u0 @ h / n
    gB = (rr - 1) * base
    gB[j] += rr * m0
    gA = (rr - 1) * agg(u0, g, G) / n
    eff = gB - f['mean'].T @ gA
    q = f['bread'] @ eff
    dt = np.r_[0.0, q[p:]]
    infl_delta = (y - mu) * f['active'] * (X @ q[:p] + dt[t] - (f['mean'] @ q)[g] + np.divide(gA, f['yt'], out=np.zeros(G), where=f['yt'] > 0)[g])
    q = f['bread'][:, j]
    dt = np.r_[0.0, q[p:]]
    infl_log = (y - mu) * f['active'] * (X @ q[:p] + dt[t] - (f['mean'] @ q)[g])
    return (dict(beta=b[j], RR=rr, mean0=m0, mean1=m1, delta=delta), infl_delta, infl_log)

def run():
    allrows = []
    audit = []
    joint = []
    checkpoint = ROOT / 'results/ppml_deletions.csv'
    allloo = pd.read_csv(checkpoint).to_dict('records') if checkpoint.exists() and '--resume' in sys.argv else []
    cached = {(r['platform'], r['time'], r['channel'], int(r['source'])): r for r in allloo}
    for platform in ['facebook', 'twitter']:
        d = load(platform)
        X = ukrainian_design(d).to_numpy()
        names = CHANNELS[platform]
        Y = d[names].to_numpy(float)
        g = codes(d['Page.Name' if platform == 'facebook' else 'user_id'])
        G = g.max() + 1
        date = codes(d.date)
        for time in ['month', 'date']:
            t = codes(pd.to_datetime(d.date).dt.to_period('M')) if time == 'month' else date
            rank = G + X.shape[1] + t.max()
            DI = []
            LI = []
            fits = []
            values = []
            loo_values = []
            for k, ch in enumerate(names):
                print(platform, time, ch, 'main', flush=True)
                fit = model(X, Y[:, k], g, t)
                val, di, li = functional(fit, X, Y[:, k], g, t)
                fits.append(fit)
                values.append(val)
                DI.append(di)
                LI.append(li)
                cov = multi_cov(np.column_stack([di, li]), g, date, rank)
                for method, V in cov.items():
                    df = min(G - 1, 199) if method == 'two_way' else G - 1
                    crit = stats.t.ppf(0.975, df)
                    q = stats.t.ppf(1 - 0.05 / (2 * len(names)), df)
                    allrows.append(dict(platform=platform, time=time, channel=ch, method=method, **val, delta_se=np.sqrt(V[0, 0]), delta_low=val['delta'] - crit * np.sqrt(V[0, 0]), delta_high=val['delta'] + crit * np.sqrt(V[0, 0]), logRR_se=np.sqrt(V[1, 1]), logRR_low=val['beta'] - q * np.sqrt(V[1, 1]), logRR_high=val['beta'] + q * np.sqrt(V[1, 1]), df=df))
                one = []
                for s in range(G):
                    key = (platform, time, ch, s)
                    if key in cached:
                        one.append({v: cached[key][v] for v in ['beta', 'RR', 'mean0', 'mean1', 'delta']})
                        continue
                    ff = model(X, Y[:, k], g, t, start=fit['beta'], omit=s, tol=2e-09)
                    vv, _, _ = functional(ff, X, Y[:, k], g, t)
                    one.append(vv)
                    allloo.append(dict(platform=platform, time=time, channel=ch, source=s, **vv, gradient=ff['gradient']))
                    if s % 20 == 0:
                        print(platform, time, ch, 'deletion', s, '/', G, flush=True)
                        pd.DataFrame(allloo).to_csv(checkpoint, index=False)
                loo_values.append(one)
                audit.append(dict(platform=platform, time=time, channel=ch, gradient=fit['gradient'], iterations=fit['iterations'], zero_total_sources=fit['zero_sources'], n=len(d), sources=int(G), time_levels=int(t.max() + 1)))
                pd.DataFrame(allrows).to_csv(ROOT / 'results/ppml_inference.csv', index=False)
                pd.DataFrame(allloo).to_csv(ROOT / 'results/ppml_deletions.csv', index=False)
                (ROOT / 'audit/ppml_models.json').write_text(json.dumps(audit, indent=2))
            DI = np.array(DI).T
            LI = np.array(LI).T
            Dcov = multi_cov(DI, g, date, rank)
            Lcov = multi_cov(LI, g, date, rank)
            betas = np.array([v['beta'] for v in values])
            deltas = np.array([v['delta'] for v in values])
            share = len(names) - 1
            like = 0
            c = np.zeros(len(names))
            c[share] = 1
            c[like] = -1
            for method in Dcov:
                df = min(G - 1, 199) if method == 'two_way' else G - 1
                crit = stats.t.ppf(0.975, df)
                se = np.sqrt(Dcov[method].sum())
                lb = float(c @ betas)
                ls = np.sqrt(c @ Lcov[method] @ c)
                joint.append(dict(platform=platform, time=time, method=method, net_delta=float(deltas.sum()), net_se=se, net_low=deltas.sum() - crit * se, net_high=deltas.sum() + crit * se, adjusted_ratio=np.exp(lb), log_ratio=lb, log_ratio_se=ls, ratio_low=np.exp(lb - crit * ls), ratio_high=np.exp(lb + crit * ls), df=df))
            L = np.array([[v['beta'] for v in one] for one in loo_values]).T
            DD = np.array([[v['delta'] for v in one] for one in loo_values]).T
            LC = (G - 1) / G * ((L - L.mean(0)).T @ (L - L.mean(0)))
            DC = (G - 1) / G * ((DD - DD.mean(0)).T @ (DD - DD.mean(0)))
            crit = stats.t.ppf(0.975, G - 1)
            q = stats.t.ppf(1 - 0.05 / (2 * len(names)), G - 1)
            for k, ch in enumerate(names):
                val = values[k]
                se = np.sqrt(DC[k, k])
                sl = np.sqrt(LC[k, k])
                allrows.append(dict(platform=platform, time=time, channel=ch, method='CV3J', **val, delta_se=se, delta_low=val['delta'] - crit * se, delta_high=val['delta'] + crit * se, logRR_se=sl, logRR_low=val['beta'] - q * sl, logRR_high=val['beta'] + q * sl, df=G - 1))
            lb = float(c @ betas)
            ls = np.sqrt(c @ LC @ c)
            se = np.sqrt(DC.sum())
            joint.append(dict(platform=platform, time=time, method='CV3J', net_delta=float(deltas.sum()), net_se=se, net_low=deltas.sum() - crit * se, net_high=deltas.sum() + crit * se, adjusted_ratio=np.exp(lb), log_ratio=lb, log_ratio_se=ls, ratio_low=np.exp(lb - crit * ls), ratio_high=np.exp(lb + crit * ls), df=G - 1))
            np.savez_compressed(ROOT / 'results' / f'ppml_{platform}_{time}_covariances.npz', beta=betas, delta=deltas, **{f'delta_{k}': v for k, v in Dcov.items()}, **{f'logRR_{k}': v for k, v in Lcov.items()}, delta_CV3J=DC, logRR_CV3J=LC)
            pd.DataFrame(allrows).to_csv(ROOT / 'results/ppml_inference.csv', index=False)
            pd.DataFrame(joint).to_csv(ROOT / 'results/ppml_joint.csv', index=False)
            print(platform, time, 'complete', flush=True)
if __name__ == '__main__':
    run()
