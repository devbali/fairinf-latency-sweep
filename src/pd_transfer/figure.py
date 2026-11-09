"""Histogram of which decode batch a prefill step's last KV joins, against the predicted one.
Usage: python3 figure.py SCORED.pkl OUT.png"""
import pickle, sys
from collections import Counter
import numpy as np



def batch_pos(r, t, after):
    l = np.array(sorted(set(r["dlog_before"] + r["decode_steps"])))
    prev = l[l <= t]
    return l, (int(len(prev) - 1) if len(prev) and t - prev[-1] <= after else None)


def offset(r, key="TJ_pred"):
    """Decode batches between the predicted and the actual pickup: + the actual is later."""
    t_act, t_pred = r["kvs"][-1], r["first_end"] + r[key] / 1000
    l, a = batch_pos(r, t_act, 0.010)
    _, p = batch_pos(r, t_pred, 0.004)
    if a is None and p is None:
        return 0
    if a is not None and p is not None:
        return a - p
    lo, hi = sorted((t_act, t_pred))
    lo_c = l[a] if a is not None and t_act == lo else l[p] if p is not None and t_pred == lo else lo
    hi_c = l[a] if a is not None and t_act == hi else l[p] if p is not None and t_pred == hi else hi
    n = int(((l > lo_c + 0.0005) & (l <= hi_c + 0.0005)).sum())
    return n if t_act > t_pred else -n


def main(rows, OUT):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    off = np.array([offset(r) for r in rows])
    N = np.array([r["N"] for r in rows])
    cats = [-2, -1, 0, 1, 2]
    labels = ["2+ earlier", "1 earlier", "predicted", "1 later", "2+ later"]
    clip = np.clip(off, -2, 2)

    BLUE, INK, INK2, GRID, SURF = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(11, 7.2), facecolor=SURF)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.35, 1], hspace=0.55, wspace=0.25)


    def bars(ax, sel, title, big=False):
        n = sel.sum()
        pct = [100 * np.mean(clip[sel] == c) for c in cats]
        ax.set_facecolor(SURF)
        b = ax.bar(range(5), pct, width=0.72, color=BLUE, edgecolor=SURF, linewidth=2)
        for rect, v in zip(b, pct):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (1.2 if big else 2), f"{v:.1f}%" if big else f"{v:.0f}%",
                    ha="center", va="bottom", color=INK, fontsize=10 if big else 8.5)
        ax.set_xticks(range(5), labels if big else ["2+ e", "1 e", "pred", "1 l", "2+ l"], fontsize=10 if big else 8.5)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, loc="left", color=INK, fontsize=12 if big else 10, pad=8)
        return pct


    ax = fig.add_subplot(gs[0, :])
    pct = bars(ax, np.ones(len(rows), bool), f"Decode batch the last KV joins, vs predicted  ({len(rows)} prefill steps)", big=True)
    ax.set_ylabel("share of prefill steps (%)")
    ax.annotate("", xy=(1.62, 92), xytext=(3.38, 92), arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.text(2.5, 94, f"predicted or next: {pct[2] + pct[3]:.1f}%", ha="center", color=INK2, fontsize=10)
    for i, (lo, hi, name) in enumerate(((1, 2, "1 request"), (2, 9, "2-8 requests"), (9, 33, "9-32 requests"), (33, 10**9, "33+ requests"))):
        sel = (N >= lo) & (N < hi)
        a = fig.add_subplot(gs[1, i])
        bars(a, sel, f"{name}  (n={sel.sum()})")
        if i == 0:
            a.set_ylabel("share (%)")
        else:
            a.set_yticklabels([])
    fig.text(0.01, 0.005, "e = earlier, l = later than the predicted batch. Prediction: posting model + copy + pickup at the end of the "
             "running decode step; decode step times from the decode server's log.", color=INK2, fontsize=8.5)
    fig.savefig(OUT, dpi=150, facecolor=SURF, bbox_inches="tight")
    c = Counter(clip.tolist())
    print({labels[cats.index(k)]: f"{100 * v / len(rows):.1f}%" for k, v in sorted(c.items())})


if __name__ == "__main__":
    main(pickle.load(open(sys.argv[1], "rb")), sys.argv[2])
