"""Stage 2 EDA (numpy + matplotlib + scipy only). Run:  PYTHONPATH=src python -m cmapss.eda
Descriptive analysis on the TRAINING files only (no test data, no models). Writes figures to results/figures/ and
results/eda_summary.json + results/eda_sensor_table.md.
Per-regime normalisation here mirrors lib.Preprocessor but keeps all 21 sensors so flat ones remain visible.
Trend statistics are Spearman correlations between a sensor and RUL: descriptive, NOT causal."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from cmapss import lib as L

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIG = os.path.join(REPO, "results", "figures"); os.makedirs(FIG, exist_ok=True)
FDS = ["FD001", "FD002", "FD003", "FD004"]
NAMES = [f"s{i}" for i in range(1, 22)]


def regime_norm(tr, k, seed=0):
    """All 21 sensors, per-regime z-score fitted on the training file. Constant-within-regime sensors keep sd=1 -> flat zero."""
    km = KMeans(k, n_init=10, random_state=seed).fit(tr[:, L.OSI]); cond = km.predict(tr[:, L.OSI])
    sx = tr[:, L.SI]; mu, sd = np.zeros((k, 21)), np.zeros((k, 21))
    for c in range(k):
        m = cond == c; mu[c], sd[c] = sx[m].mean(0), sx[m].std(0, ddof=1)
    z = (sx - mu[cond]) / np.where(sd > 1e-5, sd, 1.0)[cond]
    return z, cond, sd.mean(0), km


def eta2(x, cond):
    """share of a sensor's variance explained by the operating regime (0 when there is one regime)."""
    tot = ((x - x.mean()) ** 2).sum()
    if tot == 0 or len(np.unique(cond)) < 2: return float("nan")
    return float(sum((cond == c).sum() * (x[cond == c].mean() - x.mean()) ** 2 for c in np.unique(cond)) / tot)


def rho(x, y):
    return float("nan") if np.ptp(x) == 0 else float(spearmanr(x, y).statistic)


def binned(z, rul, edges):
    cen, lo, med, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (rul >= a) & (rul < b)
        if m.sum() < 20: continue
        cen.append((a + b) / 2); q = np.percentile(z[m], [10, 50, 90]); lo.append(q[0]); med.append(q[1]); hi.append(q[2])
    return np.array(cen), np.array(lo), np.array(med), np.array(hi)


def main():
    summ, data = {}, {}
    for fd in FDS:
        tr, te, rul_te = L.load(fd); k = L.K[fd]
        z, cond, within_sd, km = regime_norm(tr, k); rul = tr[:, L.R]; sx = tr[:, L.SI]
        pre = L.Preprocessor(k, 0).fit(tr)                                    # cross-check against the pipeline's selector
        dropped = [NAMES[j] for j in range(21) if within_sd[j] <= 1e-5]
        assert sorted(dropped) == sorted(set(NAMES) - set(pre.sensors_)), "EDA flat-sensor rule disagrees with lib.Preprocessor"
        S = []
        for j, n in enumerate(NAMES):
            S.append(dict(sensor=n, raw_std=float(sx[:, j].std()), within_regime_sd=float(within_sd[j]),
                          n_unique_values=int(len(np.unique(sx[:, j]))), eta2_regime_raw=eta2(sx[:, j], cond),
                          spearman_raw=rho(sx[:, j], rul), spearman_normalised=rho(z[:, j], rul), dropped_by_selector=n in dropped))
        lens = np.bincount(tr[:, L.U].astype(int))[1:]; tl = np.bincount(te[:, L.U].astype(int))[1:]
        summ[fd] = dict(k=k, regime_sizes=np.bincount(cond, minlength=k).tolist(), flat_sensors=dropped, sensors=S,
                        train_engine_len=dict(min=int(lens.min()), median=float(np.median(lens)), max=int(lens.max())),
                        test_engine_len=dict(min=int(tl.min()), median=float(np.median(tl)), max=int(tl.max())))
        data[fd] = (tr, z, cond, rul, lens, tl, km)
        print(fd, "flat:", dropped, "| regimes:", summ[fd]["regime_sizes"], flush=True)

    # ---- figure 1: sensor vs remaining life, per subset (normalised; median with 10-90% band over all engines)
    edges = np.arange(0, 255, 5)
    for fd in FDS:
        tr, z, cond, rul, *_ = data[fd]; S = summ[fd]["sensors"]
        fig, axs = plt.subplots(5, 5, figsize=(16, 12), sharex=True)
        for j, ax in enumerate(axs.ravel()[:21]):
            c, lo, med, hi = binned(z[:, j], rul, edges)
            ax.fill_between(c, lo, hi, alpha=.25, color="tab:blue", lw=0); ax.plot(c, med, color="tab:blue", lw=1.5)
            r = S[j]["spearman_normalised"]
            ax.set_title(f"{NAMES[j]}" + (" (flat)" if S[j]["dropped_by_selector"] else f"  ρ={r:+.2f}"), fontsize=9,
                         color="tab:red" if S[j]["dropped_by_selector"] else "black")
            ax.tick_params(labelsize=7)
        for ax in axs.ravel()[21:]: ax.axis("off")
        axs[0, 0].invert_xaxis()
        fig.supxlabel("remaining useful life (cycles; time runs left to right)"); fig.supylabel("per-regime z-score of the sensor")
        fig.suptitle(f"{fd}: sensors vs remaining life (training engines; median, 10-90% band; ρ = Spearman with RUL, descriptive only)")
        fig.tight_layout(); fig.savefig(f"{FIG}/eda_trends_{fd}.png", dpi=110); plt.close(fig)

    # ---- figure 2: before/after per-regime normalisation on FD002 and FD004
    for fd in ("FD002", "FD004"):
        tr, z, cond, rul, lens, tl, km = data[fd]; S = summ[fd]["sensors"]
        top = sorted([s for s in S if not s["dropped_by_selector"]], key=lambda s: -abs(s["spearman_normalised"]))[:2]
        fig, axs = plt.subplots(3, 3, figsize=(15, 11))
        ax = axs[0, 0]; sc = ax.scatter(tr[::5, 2], tr[::5, 3], c=cond[::5], cmap="tab10", s=3); ax.set_xlabel("operating setting 1"); ax.set_ylabel("operating setting 2")
        ax.set_title(f"{fd}: operating settings coloured by KMeans regime (k={L.K[fd]})")
        ax = axs[0, 1]; ax.bar(range(L.K[fd]), summ[fd]["regime_sizes"]); ax.set_title("rows per regime"); ax.set_xlabel("regime")
        ax = axs[0, 2]; ax.axis("off")
        ax.text(0, .95, "regime share of variance (eta², raw sensors)\nand Spearman |ρ| with RUL, raw → normalised:", va="top", fontsize=9)
        for i, s in enumerate(top):
            ax.text(0, .78 - .2 * i, f"{s['sensor']}: η²={s['eta2_regime_raw']:.2f}   |ρ| {abs(s['spearman_raw']):.2f} → {abs(s['spearman_normalised']):.2f}", va="top", fontsize=10)
        eng = [int(u) for u in np.unique(tr[:, L.U])[:3]]
        for i, s in enumerate(top):
            j = NAMES.index(s["sensor"])
            for u in eng:
                m = tr[:, L.U] == u
                axs[1 + i, 0].plot(tr[m, L.C], tr[m, 5 + j], lw=.8, label=f"engine {u}"); axs[1 + i, 1].plot(tr[m, L.C], z[m, j], lw=.8)
            axs[1 + i, 0].set_title(f"{s['sensor']} RAW vs cycle"); axs[1 + i, 1].set_title(f"{s['sensor']} per-regime NORMALISED vs cycle")
            for a in axs[1 + i, :2]: a.set_xlabel("cycle")
            ax = axs[1 + i, 2]; ax.scatter(rul[::7], tr[::7, 5 + j], s=2, c=cond[::7], cmap="tab10", alpha=.6)
            ax.set_title(f"{s['sensor']} RAW vs RUL, coloured by regime"); ax.set_xlabel("RUL"); ax.invert_xaxis()
        axs[1, 0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(f"{FIG}/eda_normalisation_{fd}.png", dpi=110); plt.close(fig)

    # ---- figure 3: engine lengths
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.4), sharey=True)
    for ax, fd in zip(axs, FDS):
        _, _, _, _, lens, tl, _ = data[fd]
        ax.hist(lens, bins=25, alpha=.7, label="train (run to failure)"); ax.hist(tl, bins=25, alpha=.7, label="test (truncated)")
        ax.set_title(f"{fd}: cycles per engine"); ax.set_xlabel("cycles")
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG}/eda_engine_lengths.png", dpi=110); plt.close(fig)

    json.dump(summ, open(os.path.join(REPO, "results", "eda_summary.json"), "w"), indent=1)
    with open(os.path.join(REPO, "results", "eda_sensor_table.md"), "w", encoding="utf8") as f:
        for fd in FDS:
            f.write(f"### {fd} (k={summ[fd]['k']}; flat within regime: {', '.join(summ[fd]['flat_sensors']) or 'none'})\n\n")
            f.write("| sensor | within-regime sd | unique values | η² regime (raw) | ρ raw | ρ normalised |\n|---|---|---|---|---|---|\n")
            for s in sorted(summ[fd]["sensors"], key=lambda s: -(0 if np.isnan(s["spearman_normalised"]) else abs(s["spearman_normalised"]))):
                e = "n/a" if np.isnan(s["eta2_regime_raw"]) else f"{s['eta2_regime_raw']:.3f}"
                r1 = "n/a" if np.isnan(s["spearman_raw"]) else f"{s['spearman_raw']:+.3f}"; r2 = "n/a" if np.isnan(s["spearman_normalised"]) else f"{s['spearman_normalised']:+.3f}"
                f.write(f"| {s['sensor']}{' (flat)' if s['dropped_by_selector'] else ''} | {s['within_regime_sd']:.2e} | {s['n_unique_values']} | {e} | {r1} | {r2} |\n")
            f.write("\n")
    print("EDA done")


if __name__ == "__main__":
    main()
