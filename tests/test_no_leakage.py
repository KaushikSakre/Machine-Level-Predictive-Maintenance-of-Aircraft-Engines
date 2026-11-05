"""Fails if any fitted component (KMeans clusterer, per-condition scaler, sensor selector, feature scaler)
has seen validation-engine or test data. Method: poison val+test rows with huge values; every fitted parameter
must be bit-identical to the clean run. A deliberately leaky control proves the test can actually detect leakage."""
import numpy as np, pytest
from cmapss import lib as L


def _poison(frames, val_units):
    tr, te, rul = frames
    tr2, te2 = tr.copy(), te.copy()
    for a, mask in ((tr2, np.isin(tr2[:, L.U].astype(int), val_units)), (te2, np.ones(len(te2), bool))):
        a[mask, L.SI] = a[mask, L.SI] * 1000.0 + 77.0
        a[mask, L.OSI] = a[mask, L.OSI] * 50.0 + 3.0
    return tr2, te2, rul


def _params_equal(a, b):
    pa, pb = a.pre.params(), b.pre.params(); fa, fb = a.fscaler.params(), b.fscaler.params()
    # KMeans centres: sklearn's threaded reductions leave ~1e-21 float noise on identical input (measured on FD001).
    # Real leakage from the poisoned rows would move them by O(1) or more, so atol=1e-9 is far below any leak.
    return (np.allclose(pa["centers"], pb["centers"], rtol=0, atol=1e-9) and np.array_equal(pa["mu"], pb["mu"]) and
            np.array_equal(pa["sd"], pb["sd"]) and pa["sensors"] == pb["sensors"] and
            np.array_equal(fa["m"], fb["m"]) and np.array_equal(fa["s"], fb["s"]))


@pytest.mark.parametrize("fd", ["FD001", "FD004"])
def test_fitted_objects_never_see_val_or_test(fd):
    frames = L.load(fd)
    clean = L.build_dataset(fd, seed=0, frames=frames)
    poisoned = L.build_dataset(fd, seed=0, frames=_poison(frames, clean.val_units))
    # poison really reached the transformed val/test data...
    assert not np.allclose(clean.S_va, poisoned.S_va) and not np.allclose(clean.S_te, poisoned.S_te)
    # ...but nothing fitted changed, and training windows are untouched
    assert _params_equal(clean, poisoned), "a fitted component depends on validation/test data"
    assert np.array_equal(clean.S_tr, poisoned.S_tr)


@pytest.mark.parametrize("fd", ["FD001", "FD004"])
def test_split_is_engine_wise_and_fit_rows_match_train(fd):
    tr, _, _ = L.load(fd)
    d = L.build_dataset(fd, seed=0)
    assert not set(d.train_units) & set(d.val_units)
    assert set(d.train_units) | set(d.val_units) == set(int(x) for x in np.unique(tr[:, L.U]))
    assert d.pre.fit_n_rows_ == int(np.isin(tr[:, L.U].astype(int), d.train_units).sum())  # fit rows == train-engine rows only
    assert len(d.S_tr) + len(d.S_va) == len(tr)


def test_negative_control_detects_deliberate_leak():
    """If the pipeline fitted on train+val ('all'), the poison test above MUST notice."""
    fd = "FD001"; frames = L.load(fd)
    clean = L.build_dataset(fd, seed=0, frames=frames, fit_on="all")
    poisoned = L.build_dataset(fd, seed=0, frames=_poison(frames, clean.val_units), fit_on="all")
    assert not _params_equal(clean, poisoned), "leak went undetected: the leakage test is vacuous"
