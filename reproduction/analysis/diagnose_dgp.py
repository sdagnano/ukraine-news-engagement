from cell_models import *
rng = np.random.default_rng(918348)
d = load('facebook')
X = ukrainian_design(d).to_numpy()
Y = d[CHANNELS['facebook']].to_numpy(float)
s, t, c = grouping(d, 'facebook', 'date')
g = codes(t // 14)
mu = np.column_stack([ppml(np.delete(X, 1, axis=1), Y[:, k], c)['mu'] for k in range(Y.shape[1])])
pools = np.load(ROOT / 'results/calibration_pools.npz')
sp, tp = (pools['source'], pools['time'])
F = pd.read_csv(ROOT / 'results/calibration_full_fit.csv')
alpha = F[F.alpha.notna()].set_index('channel').loc[CHANNELS['facebook'], 'alpha'].to_numpy()
out = []
for rep in range(100):
    fs = sp[rng.integers(len(sp), size=s.max() + 1)]
    ft = tp[rng.integers(len(tp), size=g.max() + 1)]
    means = mu * np.exp(X[:, 1, None] * (fs[s] + ft[g]))
    for k, ch in enumerate(CHANNELS['facebook']):
        y = rng.poisson(means[:, k]) if alpha[k] < 1e-10 else rng.negative_binomial(1 / alpha[k], 1 / (1 + alpha[k] * means[:, k]))
        out.append(dict(rep=rep, channel=ch, zero=float((y == 0).mean()), q99=float(np.quantile(y, 0.99)), q999=float(np.quantile(y, 0.999)), maximum=float(y.max())))
pd.DataFrame(out).to_csv(ROOT / 'results/full_DGP_distribution_draws.csv', index=False)
rows = []
z = pd.DataFrame(out)
for k, ch in enumerate(CHANNELS['facebook']):
    a = z[z.channel == ch]
    rows.append(dict(channel=ch, observed_q999=float(np.quantile(Y[:, k], 0.999)), simulated_q999_median=float(a.q999.median()), observed_maximum=float(Y[:, k].max()), simulated_maximum_median=float(a.maximum.median()), simulated_maximum_q05=float(a.maximum.quantile(0.05)), simulated_maximum_q95=float(a.maximum.quantile(0.95))))
pd.DataFrame(rows).to_csv(ROOT / 'results/full_DGP_tail_fit.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
