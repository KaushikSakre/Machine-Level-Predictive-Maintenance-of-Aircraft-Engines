"""Data audit for NASA C-MAPSS (numpy only). Run:  python -m cmapss.audit   (from the repo root, with src on the path / via pytest.ini)

Checks, for FD001-FD004:
  * every file exists and its SHA-256 equals src/cmapss/expected_sha256.json
  * engine counts (train/test) equal the published values: 100/100, 260/259, 100/100, 249/248
  * raw row counts, 26 columns, no NaN/inf
  * every engine's cycles are exactly 1..n (no gaps)
  * the RUL file has one row per test engine
Writes results/data_audit.json and exits with status 1 if anything fails.
"""
import hashlib, json, os, sys
import numpy as np
from cmapss import lib as L

EXPECTED_ENGINES = {"FD001": (100, 100), "FD002": (260, 259), "FD003": (100, 100), "FD004": (249, 248)}
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def cycles_contiguous(a):
    """True if, for every engine, the cycle column is exactly 1..n_engine_rows."""
    units = a[:, L.U].astype(int)
    for u in np.unique(units):
        c = a[units == u, L.C]
        if not np.array_equal(c, np.arange(1, len(c) + 1)): return False
    return True


def audit(data_dir=L.DATA):
    expected = json.load(open(os.path.join(HERE, "expected_sha256.json")))
    out = {"data_dir": os.path.abspath(data_dir), "files": {}, "subsets": {}, "ok": True}
    for name, exp in expected.items():
        p = os.path.join(data_dir, name)
        got = sha256(p) if os.path.exists(p) else None
        out["files"][name] = {"present": got is not None, "sha256_matches": got == exp}
        out["ok"] &= got == exp
    for fd, (e_tr, e_te) in EXPECTED_ENGINES.items():
        tr = np.loadtxt(os.path.join(data_dir, f"train_{fd}.txt")); te = np.loadtxt(os.path.join(data_dir, f"test_{fd}.txt"))
        rul = np.loadtxt(os.path.join(data_dir, f"RUL_{fd}.txt"))
        n_tr, n_te = len(np.unique(tr[:, 0])), len(np.unique(te[:, 0]))
        s = dict(train_engines=n_tr, test_engines=n_te, expected_train_engines=e_tr, expected_test_engines=e_te,
                 train_rows=int(len(tr)), test_rows=int(len(te)), rul_rows=int(len(rul)), columns=int(tr.shape[1]),
                 finite=bool(np.isfinite(tr).all() and np.isfinite(te).all() and np.isfinite(rul).all()),
                 train_cycles_contiguous=cycles_contiguous(tr), test_cycles_contiguous=cycles_contiguous(te),
                 train_engine_len_min_max=[int(np.bincount(tr[:, 0].astype(int))[1:].min()), int(np.bincount(tr[:, 0].astype(int))[1:].max())],
                 test_engine_len_min_max=[int(np.bincount(te[:, 0].astype(int))[1:].min()), int(np.bincount(te[:, 0].astype(int))[1:].max())])
        s["ok"] = bool((n_tr, n_te) == (e_tr, e_te) and s["columns"] == 26 and te.shape[1] == 26 and s["rul_rows"] == n_te
                       and s["finite"] and s["train_cycles_contiguous"] and s["test_cycles_contiguous"])
        out["subsets"][fd] = s; out["ok"] &= s["ok"]
    return out


def main():
    r = audit()
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    json.dump(r, open(os.path.join(REPO, "results", "data_audit.json"), "w"), indent=1)
    print(f"data dir: {r['data_dir']}")
    print(f"files: {sum(v['sha256_matches'] for v in r['files'].values())}/{len(r['files'])} match expected SHA-256")
    print(f"{'subset':7} {'train eng':>9} {'test eng':>8} {'train rows':>10} {'test rows':>9} {'RUL rows':>8}  status")
    for fd, s in r["subsets"].items():
        print(f"{fd:7} {s['train_engines']:>9} {s['test_engines']:>8} {s['train_rows']:>10} {s['test_rows']:>9} {s['rul_rows']:>8}  {'PASS' if s['ok'] else 'FAIL'}")
    print("OVERALL:", "PASS" if r["ok"] else "FAIL")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
