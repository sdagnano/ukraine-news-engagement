from cell_models import *

def poisson_separation_certificate(X, y, cell, relative_tolerance=1e-10):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    cell = np.asarray(cell, dtype=int)
    p = X.shape[1]
    C = int(cell.max() + 1)
    pos = y > 0
    counts = np.bincount(cell[pos], minlength=C)
    sums = ag(X[pos], cell[pos], C)
    means = np.divide(sums, counts[:, None], out=np.zeros_like(sums), where=counts[:, None] > 0)
    Z = X[pos] - means[cell[pos]]
    scale = np.sqrt(np.sum(Z * Z, axis=0))
    safe = np.where(scale > 0, scale, 1.0)
    gram = (Z / safe).T @ (Z / safe)
    eigen = np.linalg.eigvalsh(gram)
    threshold = relative_tolerance * max(float(eigen.max()), 1.0)
    rank = int(np.sum(eigen > threshold))
    certified = rank == p
    return dict(no_additional_slope_separation_certified=bool(certified), status='full_positive_row_within_cell_rank' if certified else 'inconclusive_rank_deficiency_requires_feasibility_check', positive_row_within_cell_rank=rank, slope_columns=int(p), positive_rows=int(pos.sum()), positive_cells=int(np.sum(counts > 0)), all_zero_cells=int(np.sum(counts == 0)), scaled_gram_min_eigenvalue=float(eigen.min()), scaled_gram_max_eigenvalue=float(eigen.max()), rank_threshold=float(threshold), relative_tolerance=float(relative_tolerance), method='Within-cell centered positive-response design Gram, independent of optimizer and fitted means')
