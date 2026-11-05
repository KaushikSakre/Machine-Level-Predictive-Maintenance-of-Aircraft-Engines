"""Shared C-MAPSS pipeline (numpy only: pandas is blocked by Windows Smart App Control on this machine).
Everything that is *fitted* (clusterer, scaler, sensor selector, feature scaler) is fitted on TRAINING-split engines only.
Validation engines and the official test file are only transformed."""
import os
import time
import numpy as np, torch, torch.nn as nn
from sklearn.cluster import KMeans

DATA = os.environ.get("CMAPSS_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "CMAPSSData"))
W, CAP, H = 30, 125, 30
K = {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
# array layout: col0 unit, col1 cycle, cols2-4 op settings, cols5-25 sensors s1..s21, col26 RUL (train only)
U, C, OSI, SI, R = 0, 1, slice(2, 5), slice(5, 26), 26


def load(fd, data=DATA):
    tr = np.loadtxt(f"{data}/train_{fd}.txt"); te = np.loadtxt(f"{data}/test_{fd}.txt")
    rul = np.loadtxt(f"{data}/RUL_{fd}.txt")
    mx = np.zeros(int(tr[:, U].max()) + 1); np.maximum.at(mx, tr[:, U].astype(int), tr[:, C])
    tr = np.column_stack([tr, mx[tr[:, U].astype(int)] - tr[:, C]])
    return tr, te, rul


def split_units(units, seed=0, frac=0.2):
    u = np.array(sorted(set(int(x) for x in units))); rng = np.random.RandomState(seed); rng.shuffle(u)
    n = int(round(frac * len(u)))
    return sorted(u[n:].tolist()), sorted(u[:n].tolist())  # train, val


class Preprocessor:
    """clusterer (operating conditions) + per-condition z-score + sensor selector. fit() = train engines only."""
    def __init__(self, k, seed=0):
        self.k, self.seed = k, seed

    def fit(self, a):
        self.fit_n_rows_ = len(a)
        self.km_ = KMeans(self.k, n_init=10, random_state=self.seed).fit(a[:, OSI])
        cond = self.km_.predict(a[:, OSI]); sx = a[:, SI]
        mu, sd = np.zeros((self.k, 21)), np.zeros((self.k, 21))
        for c in range(self.k):
            m = cond == c
            if m.sum() > 1: mu[c], sd[c] = sx[m].mean(0), sx[m].std(0, ddof=1)
        keep = sd.mean(0) > 1e-5                    # selector: drop sensors constant within a condition
        self.keep_ = keep; self.sensors_ = [f"s{i + 1}" for i in np.where(keep)[0]]
        self.mu_, self.sd_ = mu[:, keep], np.where(sd[:, keep] == 0, 1.0, sd[:, keep])
        return self

    def transform(self, a):
        cond = self.km_.predict(a[:, OSI])
        return ((a[:, SI][:, self.keep_] - self.mu_[cond]) / self.sd_[cond]).astype(np.float32)

    def params(self):
        return dict(centers=self.km_.cluster_centers_.copy(), mu=self.mu_.copy(), sd=self.sd_.copy(),
                    sensors=list(self.sensors_))


class FeatureScaler:
    def fit(self, X): self.m_, self.s_ = X.mean(0), X.std(0) + 1e-9; return self
    def transform(self, X): return (X - self.m_) / self.s_
    def params(self): return dict(m=self.m_.copy(), s=self.s_.copy())


def _feats(a):
    t = np.arange(len(a)) - (len(a) - 1) / 2
    slope = (t[:, None] * (a - a.mean(0))).sum(0) / (t ** 2).sum()
    return np.concatenate([a[-1], a.mean(0), a.std(0), a.min(0), a.max(0), slope])


FEAT_KINDS = ["last", "mean", "std", "min", "max", "slope"]


def make_windows(x, units, rul=None, last_only=False):
    """x: (rows, F) normalised, rows ordered by (unit, cycle). Returns seq (N,W,F), flat feats, y(rul), unit ids."""
    S, X, y, u = [], [], [], []
    for unit in np.unique(units):
        idx = np.where(units == unit)[0]; a = x[idx]
        a = np.vstack([np.repeat(a[:1], W - 1, 0), a])
        for i in ([len(idx) - 1] if last_only else range(len(idx))):
            w = a[i:i + W]; S.append(w); X.append(_feats(w)); u.append(unit)
            if rul is not None: y.append(rul[idx[i]])
    return np.array(S, np.float32), np.array(X, np.float32), np.array(y), np.array(u)


class Dataset:  # plain container
    pass


def build_dataset(fd, seed=0, frames=None, fit_on="train", pre=None):
    """fit_on='train' is the correct behaviour. fit_on='all' exists ONLY as a deliberately-leaky negative control for tests.
    pre: optional already-fitted Preprocessor (used for the cross-subset test: fitted elsewhere, transform only)."""
    tr, te, rul_te = frames if frames is not None else load(fd)
    tr_units, va_units = split_units(tr[:, U], seed)
    is_tr = np.isin(tr[:, U].astype(int), tr_units)
    if pre is None:
        pre = Preprocessor(K[fd], seed).fit(tr[is_tr] if fit_on == "train" else tr)
    xtr_all, xte = pre.transform(tr), pre.transform(te)
    S, X, y, u = make_windows(xtr_all, tr[:, U].astype(int), tr[:, R])
    St, Xt, _, _ = make_windows(xte, te[:, U].astype(int), last_only=True)
    m = np.isin(u, va_units)
    fs = FeatureScaler().fit(X[~m] if fit_on == "train" else X)
    d = Dataset()
    d.fd, d.pre, d.fscaler, d.train_units, d.val_units = fd, pre, fs, tr_units, va_units
    d.S_tr, d.S_va, d.S_te = S[~m], S[m], St
    d.X_tr, d.X_va, d.X_te = fs.transform(X[~m]), fs.transform(X[m]), fs.transform(Xt)
    d.rul_tr, d.rul_va, d.rul_te = y[~m], y[m], rul_te      # raw (uncapped) RUL
    d.y_tr, d.y_va = np.minimum(d.rul_tr, CAP), np.minimum(d.rul_va, CAP)
    d.c_tr, d.c_va, d.c_te = (d.rul_tr <= H).astype(int), (d.rul_va <= H).astype(int), (rul_te <= H).astype(int)
    d.n_feat, d.n_rows_train_file = S.shape[2], len(tr)
    d.feature_names = [f"{k}_{s}" for k in FEAT_KINDS for s in pre.sensors_]
    return d


# ---------------- metrics
def rmse(a, b): return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def nasa(true, pred):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


def reg_metrics(true_raw, pred):
    p = np.clip(pred, 0, CAP)
    return dict(rmse_raw=rmse(true_raw, p), rmse_capped=rmse(np.minimum(true_raw, CAP), p), nasa=nasa(true_raw, p))


# ---------------- sequence models (trained from scratch)
class GRUNet(nn.Module):
    def __init__(s, f, h=64):
        super().__init__(); s.g = nn.GRU(f, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): o, _ = s.g(x); return s.o(o[:, -1]).squeeze(-1)


class _Block(nn.Module):
    def __init__(s, cin, cout, dil, k=3, p=0.1):
        super().__init__(); s.pad = (k - 1) * dil
        s.c1 = nn.Conv1d(cin, cout, k, dilation=dil); s.c2 = nn.Conv1d(cout, cout, k, dilation=dil)
        s.act, s.dr = nn.ReLU(), nn.Dropout(p); s.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()
    def forward(s, x):
        y = s.dr(s.act(s.c1(nn.functional.pad(x, (s.pad, 0)))))
        y = s.dr(s.act(s.c2(nn.functional.pad(y, (s.pad, 0)))))
        return s.act(y + s.res(x))


class TCNNet(nn.Module):  # causal dilated residual TCN, receptive field > 30 steps
    def __init__(s, f, ch=32, dils=(1, 2, 4, 8)):
        super().__init__(); L, c = [], f
        for d in dils: L.append(_Block(c, ch, d)); c = ch
        s.net = nn.Sequential(*L); s.o = nn.Linear(ch, 1)
    def forward(s, x): return s.o(s.net(x.transpose(1, 2))[:, :, -1]).squeeze(-1)


def train_seq(kind, d, seed, epochs=10, bs=256, lr=2e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m = (GRUNet if kind == "gru" else TCNNet)(d.n_feat)
    opt, lossf = torch.optim.Adam(m.parameters(), lr), nn.MSELoss()
    xt, yt = torch.tensor(d.S_tr), torch.tensor(d.y_tr / CAP, dtype=torch.float32)
    xv, xte = torch.tensor(d.S_va), torch.tensor(d.S_te)
    times, best, state = [], 1e9, None
    for ep in range(epochs):
        t0 = time.time(); m.train(); perm = torch.randperm(len(xt))
        for i in range(0, len(xt), bs):
            b = perm[i:i + bs]; opt.zero_grad(); lossf(m(xt[b]), yt[b]).backward(); opt.step()
        times.append(time.time() - t0); m.eval()
        with torch.no_grad(): v = rmse(d.y_va, m(xv).numpy() * CAP)
        if v < best: best, state = v, {k: t.clone() for k, t in m.state_dict().items()}
    m.load_state_dict(state); m.eval()
    with torch.no_grad(): pred = m(xte).numpy() * CAP
    return dict(**reg_metrics(d.rul_te, pred), val_rmse_capped=best, sec_per_epoch=float(np.mean(times)), epochs=epochs,
                params=int(sum(p.numel() for p in m.parameters()))), m
