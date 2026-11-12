from cell_models import *
FOCAL = ['Likes', 'Love', 'Angry', 'Haha']

def fit_reml(y, V, moderator=None):
    G, K = y.shape
    scale = y.std(0, ddof=1)
    ys = y / scale
    Vs = V / np.outer(scale, scale)[None, :, :]
    U = np.ones((G, 1)) if moderator is None else np.column_stack([np.ones(G), moderator])
    q = U.shape[1]
    F = np.array([np.kron(U[g:g + 1], np.eye(K)) for g in range(G)])
    ridge = 1e-10
    Vs = Vs + np.eye(K)[None, :, :] * ridge

    def evaluate(par, full=False):
        L = np.zeros((K, K))
        ix = 0
        for i in range(K):
            for j in range(i + 1):
                L[i, j] = np.exp(par[ix]) if i == j else par[ix]
                ix += 1
        Tau = L @ L.T
        S = Vs + Tau
        W = np.linalg.inv(S)
        A = np.einsum('gki,gkl,glj->ij', F, W, F)
        r = np.einsum('gki,gkl,gl->i', F, W, ys)
        beta = linalg.solve(A, r, assume_a='pos')
        res = ys - np.einsum('gki,i->gk', F, beta)
        loss = 0.5 * (np.linalg.slogdet(S)[1].sum() + np.linalg.slogdet(A)[1] + np.einsum('gk,gkl,gl->', res, W, res))
        Ainv = linalg.inv(A)
        WF = np.einsum('gkl,glj->gkj', W, F)
        wr = np.einsum('gkl,gl->gk', W, res)
        GT = 0.5 * np.sum(W - np.einsum('gki,ij,glj->gkl', WF, Ainv, WF) - wr[:, :, None] * wr[:, None, :], axis=0)
        GL = 2 * GT @ L
        grad = []
        for i in range(K):
            for j in range(i + 1):
                grad.append(GL[i, j] * L[i, j] if i == j else GL[i, j])
        return (loss, Tau, W, beta, F) if full else (loss, np.array(grad))
    bounds = []
    start = []
    for i in range(K):
        for j in range(i + 1):
            bounds.append((-12, 3) if i == j else (-5, 5))
            start.append(np.log(0.5) if i == j else 0.0)
    attempts = []
    initial = np.array(start)
    analytic = evaluate(initial)[1]
    numeric = []
    for j in range(len(initial)):
        e = np.zeros_like(initial)
        e[j] = 1e-05
        numeric.append((evaluate(initial + e)[0] - evaluate(initial - e)[0]) / 2e-05)
    gradient_check = float(np.max(abs(analytic - numeric) / np.maximum(1, abs(np.array(numeric)))))
    for shift in [0.0, -1.5, 1.0, -3.0]:
        x = np.array(start)
        ii = 0
        for i in range(K):
            for j in range(i + 1):
                if i == j:
                    x[ii] += shift
                ii += 1
        r = optimize.minimize(evaluate, x, jac=True, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-14, 'gtol': 1e-07, 'maxiter': 3000, 'maxls': 60, 'maxcor': 30})
        attempts.append(r)
    best = min(attempts, key=lambda r: r.fun)
    polish = optimize.minimize(evaluate, best.x, jac=True, method='SLSQP', bounds=bounds, options={'ftol': 1e-11, 'maxiter': 2000})
    attempts.append(polish)
    best = min(attempts, key=lambda r: r.fun)
    loss, Tau, W, beta, F = evaluate(best.x, True)
    prior = np.einsum('gki,i->gk', F, beta)
    post = prior + np.einsum('ij,gjk,gk->gi', Tau, W, ys - prior)
    cov = np.array([Tau - Tau @ ww @ Tau for ww in W])
    cov = (cov + cov.transpose(0, 2, 1)) / 2
    rng = np.random.default_rng(219734)
    prob = []
    for g in range(G):
        ev, Q = np.linalg.eigh(cov[g])
        draw = post[g] + rng.normal(size=(20000, K)) @ (Q * np.sqrt(np.maximum(ev, 0))).T
        prob.append(np.mean((draw[:, 0] < 0) & (draw[:, 1] < 0) & (draw[:, 2] > 0) & (draw[:, 3] > 0)))
    return dict(post=post * scale, cov=cov * np.outer(scale, scale)[None, :, :], Tau=Tau * np.outer(scale, scale), prior=prior * scale, prob=np.array(prob), loss=float(loss), success=bool(best.success), message=str(best.message), gradient=float(np.max(abs(best.jac))), gradient_check=gradient_check, attempt_losses=[float(r.fun) for r in attempts], ridge_standardized=ridge, scale=scale)

def main():
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    Y = d[FOCAL].to_numpy(float)
    s, t, c = grouping(d, 'facebook', 'date')
    m = codes(pd.to_datetime(d.date).dt.to_period('M'))
    estimates = []
    Vday = []
    Vblock = []
    ids = []
    nposts = []
    diagnostics = []
    for sg in range(s.max() + 1):
        ix = s == sg
        H = X[ix, 1]
        n = ix.sum()
        if n < 50 or min((H == 0).sum(), (H == 1).sum()) < 5:
            continue
        M = np.column_stack([np.ones(n), H, X[ix, 0], X[ix, 2:], np.eye(m.max() + 1)[m[ix], 1:]])
        chosen = []
        rank = 0
        for j in range(M.shape[1]):
            r = np.linalg.matrix_rank(M[:, chosen + [j]])
            if r > rank:
                chosen.append(j)
                rank = r
        mm = M[:, chosen]
        B = linalg.inv(mm.T @ mm)
        b = B @ mm.T @ Y[ix]
        res = Y[ix] - mm @ b
        j = chosen.index(1)
        obs = (mm @ B[:, j])[:, None] * res
        Vs = []
        groups = []
        for width in [1, 14]:
            h = codes(t[ix] // width)
            I = ag(obs, h)
            G = len(I)
            V = I.T @ I * G / (G - 1) * (n - 1) / (n - len(chosen))
            Vs.append(V)
            groups.append(G)
        eligible = groups[0] >= 10 and groups[1] >= 5 and np.all(np.diag(Vs[0]) > 1e-12) and np.all(np.diag(Vs[1]) > 1e-12)
        diagnostics.append(dict(source=sg, n=int(n), days=groups[0], blocks14=groups[1], dropped_nuisance=int(M.shape[1] - len(chosen)), hierarchical_eligible=eligible))
        if not eligible:
            continue
        ids.append(sg)
        nposts.append(n)
        estimates.append(b[j])
        Vday.append(Vs[0])
        Vblock.append(Vs[1])
    y = np.array(estimates)
    vs = [np.array(Vday), np.array(Vblock)]
    rows = []
    summ = []
    covs = {}
    moderator = np.log(nposts)
    moderator = (moderator - moderator.mean()) / moderator.std()
    for label, V, mod in [('day', vs[0], None), ('block14', vs[1], None), ('block14_volume', vs[1], moderator)]:
        result = fit_reml(y, V, mod)
        covs[label + '_Tau'] = result['Tau']
        covs[label + '_post_cov'] = result['cov']
        covs[label + '_sampling_cov'] = V
        for g, sg in enumerate(ids):
            for k, ch in enumerate(FOCAL):
                se = np.sqrt(V[g, k, k])
                ps = np.sqrt(result['cov'][g, k, k])
                post = result['post'][g, k]
                rows.append(dict(model=label, source=sg, n=int(nposts[g]), channel=ch, beta=y[g, k], se=se, low=y[g, k] - 1.96 * se, high=y[g, k] + 1.96 * se, posterior_mean=post, posterior_low=post - 1.96 * ps, posterior_high=post + 1.96 * ps, posterior_joint_probability=result['prob'][g]))
        raw_pattern = (y[:, 0] < 0) & (y[:, 1] < 0) & (y[:, 2] > 0) & (y[:, 3] > 0)
        p = result['post']
        pat = (p[:, 0] < 0) & (p[:, 1] < 0) & (p[:, 2] > 0) & (p[:, 3] > 0)
        summ.append(dict(model=label, sources=len(ids), raw_pattern=int(raw_pattern.sum()), posterior_mean_pattern=int(pat.sum()), posterior_joint_over95=int((result['prob'] > 0.95).sum()), mean_joint_probability=float(result['prob'].mean()), success=result['success'], gradient=result['gradient'], gradient_check=result['gradient_check'], attempt_losses=result['attempt_losses'], message=result['message'], REML_loss=result['loss'], between_source_sd=dict(zip(FOCAL, np.sqrt(result['Tau'].diagonal()).tolist())), assumptions='Gaussian exchangeability, estimated sampling covariance treated as known, conditional hyperparameters, independent source errors; common cross-source shocks not identified by within-source block variance.'))
        print(label, summ[-1], flush=True)
    pd.DataFrame(rows).to_csv(ROOT / 'results/hierarchical_sources.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT / 'results/source_eligibility.csv', index=False)
    (ROOT / 'audit/hierarchical_models.json').write_text(json.dumps(summ, indent=2))
    np.savez_compressed(ROOT / 'results/hierarchical_covariances.npz', **covs)
if __name__ == '__main__':
    main()
