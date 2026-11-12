from cell_models import *

def cr2_general(Sg, scores, tau, j=1):
    G, P, _ = Sg.shape
    B = linalg.inv(Sg.sum(0))
    T = np.einsum('g,gij->ij', tau, Sg)
    c = np.eye(P)[:, j]
    theta = []
    for g in range(G):
        ev, V = linalg.eigh(Sg[g])
        keep = ev > max(ev.max(), 1) * 1e-11
        ev = ev[keep]
        V = V[:, keep]
        if len(ev) == 0:
            theta.append(np.zeros(P))
            continue
        L = np.sqrt(ev)[:, None] * V.T
        K = tau[g] * np.eye(len(ev)) - 2 * tau[g] * L @ B @ L.T + L @ B @ T @ B @ L.T
        ee, U = linalg.eigh(K)
        if ee.min() <= 0:
            raise RuntimeError('Nonpositive working residual covariance')
        A = U * np.sqrt(tau[g] / ee) @ U.T
        theta.append(V / np.sqrt(ev) @ A @ (np.sqrt(ev) * (V.T @ B @ c)))
    theta = np.array(theta)
    influence = np.einsum('gi,gi->g', theta, scores)
    variance = influence @ influence
    AA = np.einsum('gij,gj->gi', Sg, theta)
    dd = np.einsum('gi,gi->g', theta, AA)
    AB = AA @ B @ AA.T
    Gram = np.diag(tau * dd) - (tau[:, None] + tau[None, :]) * AB + AA @ B @ T @ B @ AA.T
    df = float(np.trace(Gram) ** 2 / np.sum(Gram * Gram))
    return (variance, df)

def main():
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    Y = d[CHANNELS['facebook']].to_numpy(float)
    s, t, c = grouping(d, 'facebook', 'date')
    YY = np.column_stack([Y, (Y > 0).astype(float)])
    a = cell_ols(X, YY, c)
    G = s.max() + 1
    Sg = []
    scores = []
    for sg in range(G):
        ix = s == sg
        Sg.append(a['x'][ix].T @ a['x'][ix])
        scores.append(a['x'][ix].T @ a['res'][ix])
    Sg = np.array(Sg)
    scores = np.array(scores)
    rows = []
    for k, ch in enumerate(CHANNELS['facebook'] * 2):
        target = 'count' if k < 8 else 'any'
        tau = ag(a['res'][:, k] ** 2, s) / np.bincount(s)
        floor = np.median(tau[tau > 0]) * 1e-06
        tau = np.maximum(tau, floor)
        for name, tt in [('identity', np.ones(G)), ('source_residual_variance', tau)]:
            v, df = cr2_general(Sg, scores[:, :, k], tt)
            se = np.sqrt(v)
            b = a['beta'][1, k]
            q = stats.t.ppf(1 - 0.05 / 16, df)
            rows.append(dict(target=target, channel=ch, working_covariance=name, beta=b, se=se, df=df, low=b - q * se, high=b + q * se, variance_floor=floor if name != 'identity' else 0))
    pd.DataFrame(rows).to_csv(ROOT / 'results/working_covariance_sensitivity.csv', index=False)
    print(pd.DataFrame(rows).query("target=='count' and channel in ['Likes','Love','Angry','Haha']").to_string(index=False))
if __name__ == '__main__':
    main()
