"""The data audit must pass on the real data and must fail on tampered data (so it is not vacuous)."""
import shutil, numpy as np
from cmapss import audit, lib as L


def test_audit_passes_on_real_data():
    r = audit.audit()
    assert r["ok"], r
    assert {fd: (s["train_engines"], s["test_engines"]) for fd, s in r["subsets"].items()} == audit.EXPECTED_ENGINES


def test_audit_detects_tampered_file(tmp_path):
    for f in ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt"):
        shutil.copy(f"{L.DATA}/{f}", tmp_path / f)
    for fd in ("FD002", "FD003", "FD004"):
        for kind in ("train", "test", "RUL"):
            shutil.copy(f"{L.DATA}/{kind}_{fd}.txt", tmp_path / f"{kind}_{fd}.txt")
    p = tmp_path / "train_FD001.txt"; p.write_text(p.read_text().replace("1 1 ", "1 2 ", 1))   # break engine 1's cycle numbering
    r = audit.audit(str(tmp_path))
    assert not r["ok"] and not r["subsets"]["FD001"]["train_cycles_contiguous"]
