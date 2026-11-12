from cell_models import *
FOCAL = ['Likes', 'Love', 'Angry', 'Haha']

def prepare(X, Y, cell):
    nuisance = np.delete(X, 1, axis=1)
    ng = np.bincount(cell)
    z = nuisance - ag(nuisance, cell)[cell] / ng[cell, None]
    yy = Y - ag(Y, cell)[cell] / ng[cell, None]
    B = linalg.inv(z.T @ z)
    yr = yy - z @ (B @ z.T @ yy)
    return (z, B, yr, ng)

def fit_latent(h, z, B, yr, cell, ng):
    hw = h - ag(h, cell)[cell] / ng[cell]
    hr = hw - z @ (B @ (z.T @ hw))
    den = hr @ hr
    if den < 1e-08:
        raise RuntimeError('Latent hostility unidentified')
    return hr @ yr / den

def main():
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    H = X[:, 1]
    s, t, c = grouping(d, 'facebook', 'date')
    Y = d[FOCAL].to_numpy(float)
    YY = np.column_stack([Y, (Y > 0).astype(float)])
    z, B, yr, ng = prepare(X, YY, c)
    baseline = fit_latent(H, z, B, yr, c, ng)
    rows = []
    for k, ch in enumerate(FOCAL * 2):
        target = 'count' if k < 4 else 'any'
        num = float(H @ yr[:, k])
        sign = np.sign(num)
        for mode in ['positive_label_corrections_only', 'either_label_corrections']:
            gain = sign * (2 * H - 1) * yr[:, k]
            if mode == 'positive_label_corrections_only':
                gain = np.where(H == 1, gain, -np.inf)
            eligible = np.flatnonzero(gain > 0)
            order = eligible[np.argsort(-gain[eligible])]
            cumulative = np.cumsum(gain[order])
            kk = int(np.searchsorted(cumulative, abs(num)) + 1)
            if kk > len(order):
                rows.append(dict(channel=ch, target=target, mode=mode, tips=False))
                continue
            h = H.copy()
            h[order[:kk]] = 1 - h[order[:kk]]
            beta = fit_latent(h, z, B, yr, c, ng)[k]
            hprev = H.copy()
            hprev[order[:kk - 1]] = 1 - hprev[order[:kk - 1]]
            before = fit_latent(hprev, z, B, yr, c, ng)[k]
            assert sign * beta <= 1e-07 and sign * before > 0
            rows.append(dict(channel=ch, target=target, mode=mode, tips=True, original_beta=baseline[k], minimum_flips=kk, fraction_all_posts=kk / len(H), fraction_positive_labels=kk / (H == 1).sum(), positive_labels_corrected=int((H[order[:kk]] == 1).sum()), negative_labels_corrected=int((H[order[:kk]] == 0).sum()), beta_after=beta, beta_one_fewer=before, interpretation='Minimum count for an unconstrained outcome-residual-ranked hypothetical perturbation; no estimate of actual label errors, no FPR identified.'))
    pd.DataFrame(rows).to_csv(ROOT / 'results/adversarial_label_tipping.csv', index=False)
    print(pd.DataFrame(rows)[['channel', 'target', 'mode', 'minimum_flips', 'fraction_all_posts']].to_string(index=False), flush=True)
    cutoff = float(np.quantile(Y[H == 1, 2], 0.9))
    high = Y[:, 2] >= cutoff
    rng = np.random.default_rng(893417)
    out = []
    for low in [0.0, 0.02]:
        for highrate in [0.0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6]:
            for negative in [0.0, 0.02]:
                prob = np.where(H == 1, np.where(high, highrate, low), negative)
                draws = 100 if prob.max() > 0 else 1
                betas = []
                errors = []
                for rep in range(draws):
                    change = rng.random(len(H)) < prob
                    h = H.copy()
                    h[change] = 1 - h[change]
                    betas.append(fit_latent(h, z, B, yr, c, ng))
                    errors.append(int(change.sum()))
                bs = np.array(betas)
                for k, ch in enumerate(FOCAL * 2):
                    out.append(dict(channel=ch, target='count' if k < 4 else 'any', positive_low_correction=low, positive_high_Angry_correction=highrate, negative_correction=negative, high_Angry_cutoff=cutoff, draws=draws, median_beta=float(np.median(bs[:, k])), assumption_q05=float(np.quantile(bs[:, k], 0.05)), assumption_q95=float(np.quantile(bs[:, k], 0.95)), same_point_sign_fraction=float(np.mean(np.sign(bs[:, k]) == np.sign(baseline[k]))), mean_changed_posts=float(np.mean(errors))))
                print('label grid', low, highrate, negative, flush=True)
    pd.DataFrame(out).to_csv(ROOT / 'results/differential_label_grid.csv', index=False)
    (ROOT / 'audit/measurement_sensitivity.json').write_text(json.dumps(dict(seed=893417, high_Angry_cutoff=cutoff, scope='Assumption-driven correction fractions among observed labels. No human ground truth or identified misclassification model. Quantiles describe simulated labels conditional on assumptions, not sampling confidence.'), indent=2))
if __name__ == '__main__':
    main()
