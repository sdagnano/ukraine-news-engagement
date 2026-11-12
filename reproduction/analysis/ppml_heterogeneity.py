from cell_models import *
FOCAL = ['Likes', 'Love', 'Angry', 'Haha']

def expanded(X, source, quartile):
    add = np.column_stack([X[:, 1] * (quartile[source] == j) for j in [1, 2, 3]])
    Z = np.column_stack([X, add])
    A = np.zeros_like(Z)
    A[:, 1] = 1
    for j in [1, 2, 3]:
        A[:, X.shape[1] + j - 1] = quartile[source] == j
    return (Z, A)

def main():
    d = load('facebook')
    X = ukrainian_design(d).to_numpy()
    s, t, c = grouping(d, 'facebook', 'date')
    Y = d[FOCAL].to_numpy(float)
    ng = np.bincount(s)
    cuts = np.quantile(ng, [0.25, 0.5, 0.75])
    quartile = np.searchsorted(cuts, ng, side='right')
    Z, A = expanded(X, s, quartile)
    rows = []
    source_rows = []
    aud = []
    params = {}
    for k, ch in enumerate(FOCAL):
        f = ppml(Z, Y[:, k], c)
        v, oi, lr = functional(f, Z, Y[:, k], c, A)
        for width, V, se, df, q in inference(np.column_stack([oi[:, 2], lr]), s, t, len(np.bincount(c)) + Z.shape[1], 8):
            rows.append(dict(model='quartile_H_slopes', population='all', channel=ch, block=width, **v, logRR_low=np.log(v['RR']) - q * se[1], logRR_high=np.log(v['RR']) + q * se[1], n=len(d)))
        params[ch + '_quartile'] = f['beta']
        aud.append(dict(model='quartile', channel=ch, gradient=f['gradient'], zero_cells=f['zero_cells'], condition=f['condition']))
        for sg in range(s.max() + 1):
            ix = s == sg
            H = X[ix, 1]
            if ix.sum() < 50 or min((H == 0).sum(), (H == 1).sum()) < 5:
                continue
            xx = X[ix]
            cc = codes(c[ix])
            ngc = np.bincount(cc)
            wx = xx - ag(xx, cc)[cc] / ngc[cc, None]
            order = [1, 0] + list(range(2, X.shape[1]))
            chosen = []
            rank = 0
            for j in order:
                nr = np.linalg.matrix_rank(wx[:, chosen + [j]])
                if nr > rank:
                    chosen.append(j)
                    rank = nr
            if 1 not in chosen:
                source_rows.append(dict(source=sg, channel=ch, n=int(ix.sum()), finite=False, reason='H unidentified within source/day'))
                continue
            try:
                ff = ppml(xx[:, chosen], Y[ix, k], cc)
                slope = ff['beta'][chosen.index(1)]
                m0 = np.mean(ff['mu'] * np.exp(-slope * H))
                m1 = m0 * np.exp(slope)
                source_rows.append(dict(source=sg, channel=ch, n=int(ix.sum()), finite=True, beta=slope, mean0=m0, mean1=m1, delta=m1 - m0, gradient=ff['gradient'], zero_cells=ff['zero_cells'], dropped_controls=X.shape[1] - len(chosen)))
            except (RuntimeError, linalg.LinAlgError) as e:
                source_rows.append(dict(source=sg, channel=ch, n=int(ix.sum()), finite=False, reason=str(e)))
        print(ch, 'source fits completed', flush=True)
    sr = pd.DataFrame(source_rows)
    ok = sr.groupby('source').finite.sum()
    common = ok[ok == len(FOCAL)].index.to_numpy()
    mask = np.isin(s, common)
    for label, xx, aa in [('constant_H', X, None), ('quartile_H_slopes', Z, A)]:
        for k, ch in enumerate(FOCAL):
            ff = ppml(xx[mask], Y[mask, k], codes(c[mask]))
            v, oi, lr = functional(ff, xx[mask], Y[mask, k], codes(c[mask]), aa[mask] if aa is not None else None)
            rows.append(dict(model=label, population='common_source_finite', channel=ch, block=0, **v, n=int(mask.sum()), sources=len(common)))
    for ch in FOCAL:
        z = sr[(sr.channel == ch) & sr.source.isin(common)].copy()
        m0 = np.average(z.mean0, weights=z.n)
        m1 = np.average(z.mean1, weights=z.n)
        rows.append(dict(model='all_slopes_source_specific', population='common_source_finite', channel=ch, block=0, mean0=m0, mean1=m1, delta=m1 - m0, RR=m1 / m0, n=int(z.n.sum()), sources=len(common)))
    pd.DataFrame(rows).to_csv(ROOT / 'results/ppml_heterogeneity.csv', index=False)
    sr.to_csv(ROOT / 'results/ppml_source_slopes.csv', index=False)
    (ROOT / 'audit/ppml_heterogeneity.json').write_text(json.dumps(dict(quartile_cutpoints=cuts.tolist(), sources_per_quartile=np.bincount(quartile).tolist(), common_finite_sources=common.tolist(), models=aud, scope='Finite-source subset is outcome-selected; source-specific models also relax nuisance slopes. No inference is attached to a source-specific/common-population point comparison.'), indent=2))
    np.savez_compressed(ROOT / 'results/ppml_heterogeneity_parameters.npz', **params)
    sw = grouping(d, 'facebook', 'week')[2]
    scores = []
    for fold in range(5):
        train = t % 5 != fold
        test = ~train
        mapping = np.unique(sw[train])
        test &= np.isin(sw, mapping)
        ctr = codes(sw[train])
        ctest = np.searchsorted(mapping, sw[test])
        ntrain = np.bincount(ctr)
        for label, xx in [('constant_H', X), ('quartile_H', Z)]:
            for k, ch in enumerate(FOCAL):
                f = ppml(xx[train], Y[train, k], ctr)
                eta = xx[train] @ f['beta']
                mx = np.full(len(mapping), -np.inf)
                np.maximum.at(mx, ctr, eta)
                den = ag(np.exp(eta - mx[ctr]), ctr)
                logalpha = np.full(len(mapping), -np.inf)
                pos = f['yt'] > 0
                logalpha[pos] = np.log(f['yt'][pos]) - mx[pos] - np.log(den[pos])
                pred = np.exp(logalpha[ctest] + xx[test] @ f['beta'])
                y = Y[test, k]
                finite = pred > 0
                dev = np.full(len(y), np.inf)
                dev[(pred == 0) & (y == 0)] = 0
                dev[finite] = 2 * (np.where(y[finite] > 0, y[finite] * np.log(np.maximum(y[finite], 1e-300) / pred[finite]), 0) - (y[finite] - pred[finite]))
                scores.append(dict(fold=fold, model=label, channel=ch, heldout_posts=len(y), unseen_cell_posts=int((~train).sum() - test.sum()), zero_prediction_positive_outcome=int(((pred == 0) & (y > 0)).sum()), MSE=float(np.mean((y - pred) ** 2)), MAE=float(np.mean(abs(y - pred))), positive_training_cell_posts=int(finite.sum()), deviance_positive_training_cells=float(np.mean(dev[finite])), deviance_all=float(np.mean(dev))))
        print('prediction fold', fold, 'complete', flush=True)
    pd.DataFrame(scores).to_csv(ROOT / 'results/ppml_date_holdout.csv', index=False)
if __name__ == '__main__':
    main()
