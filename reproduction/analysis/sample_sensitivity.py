from cell_models import *
rows = []
verify = []
for platform in ['facebook', 'twitter']:
    d = load(platform)
    X = ukrainian_design(d).to_numpy()
    Y = d[CHANNELS[platform]].to_numpy(float)
    s, t, c = grouping(d, platform, 'date')
    a = cell_ols(X, Y, c)
    h = 1 / a['cell_counts'][c] + np.einsum('ij,ij->i', a['x'] @ a['bread'], a['x'])
    dff = np.divide(-a['obs'], 1 - h[:, None], out=np.zeros_like(a['obs']), where=1 - h[:, None] > 1e-10)
    targets = list(range(Y.shape[1])) if platform == 'twitter' else [0, 1, 2, 3]
    for k in targets:
        ch = CHANNELS[platform][k]
        idx = int(np.argmax(abs(dff[:, k])))
        keep = np.arange(len(d)) != idx
        direct = cell_ols(X[keep], Y[keep], codes(c[keep]))['beta'][1, k] - a['beta'][1, k]
        verify.append(dict(platform=platform, channel=ch, max_single_deletion_formula_error=float(abs(direct - dff[idx, k]))))
        if platform != 'facebook':
            continue
        for direction in ['raise_coefficient', 'lower_coefficient']:
            order = np.argsort(-dff[:, k] if direction == 'raise_coefficient' else dff[:, k])
            drop = order[:139]
            mask = np.ones(len(d), bool)
            mask[drop] = False
            beta = cell_ols(X[mask], Y[mask], codes(c[mask]))['beta'][1]
            rows.append(dict(target_channel=ch, selection=direction, omitted=139, original_beta=a['beta'][1, k], refitted_beta=beta[k], refitted_total=float(beta.sum()), scope='Influence-selected stress subset, not identification of original omitted139 and not a globally sharp deletion bound.'))
pd.DataFrame(rows).to_csv(ROOT / 'results/unknown139_stress.csv', index=False)
(ROOT / 'audit/post_deletion_validation.json').write_text(json.dumps(verify, indent=2))
print(pd.DataFrame(rows).to_string(index=False))
assert max((v['max_single_deletion_formula_error'] for v in verify)) < 1e-07
