"""Paired bootstrap of FID against fixed reference statistics (results/e3c/PROTOCOL.md, results/dns/PROTOCOL.md).
FID(mu, S) = |mu - mu_r|^2 + tr S + tr S_r - 2 tr sqrt(S_r^1/2 S S_r^1/2); S_r^1/2 is computed once (the reference is fixed), so each
resample needs one symmetric eigendecomposition. The same resampled image indices are used for every arm (paired), and the
CI is conditional on the reference set and on n."""
import numpy as np
import scipy.linalg


class FIDRef:
    def __init__(self, mu_r, s_r):
        self.mu_r, self.s_r = mu_r.astype(np.float64), s_r.astype(np.float64)
        w, V = np.linalg.eigh((self.s_r + self.s_r.T) / 2)
        self.sq = (V * np.sqrt(np.clip(w, 0, None))) @ V.T
        self.tr_r = float(np.trace(self.s_r))

    def fid(self, F):
        """float64 throughout; on the GPU when one is available (same arithmetic, much faster than numpy here)."""
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if not hasattr(self, "_t"):
            self._t = (torch.as_tensor(self.sq, device=dev), torch.as_tensor(self.mu_r, device=dev))
        sq, mu_r = self._t
        X = torch.as_tensor(np.asarray(F, dtype=np.float64), device=dev)
        mu = X.mean(0)
        Xc = X - mu
        S = Xc.T @ Xc / (X.shape[0] - 1)
        M = sq @ S @ sq
        ev = torch.linalg.eigvalsh((M + M.T) / 2)
        return float(((mu - mu_r) ** 2).sum() + torch.trace(S) + self.tr_r - 2 * ev.clamp_min(0).sqrt().sum())


def paired_boot(ref, feats, pairs, n_boot=1000, seed=0):
    """feats: name -> [n, 2048] (same image order); pairs: [(a, b)] -> FID(a) - FID(b) with 95% percentile CIs."""
    names = sorted({x for p in pairs for x in p})
    n = len(next(iter(feats.values())))
    point = {k: ref.fid(feats[k]) for k in names}
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in names}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k in names:
            draws[k].append(ref.fid(feats[k][idx]))
    out = {"fid": point, "fid_ci": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items()}, "pairs": {}}
    for a, b in pairs:
        d = np.asarray(draws[a]) - np.asarray(draws[b])
        out["pairs"][f"{a}_vs_{b}"] = {"diff": point[a] - point[b], "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    return out
