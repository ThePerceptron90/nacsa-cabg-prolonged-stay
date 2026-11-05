"""
NACSA PLoS project | shared figures (Step 8.4d onwards)
=============================================================================
Four curve plots, each drawn one panel per stage for any set of models on
either dataset (2012-16 out-of-fold or 2017-19):

    roc_panel()          true vs false positive rate; AUC (95% CI) in the legend
    pr_panel()           precision vs recall; the event rate as reference line
    calibration_panel()  the lowess curve used for ICI (decision 116) vs the diagonal
    decision_panel()     net benefit 0-50%, agreed 10-30% range shaded, flag-all/none
    stage_figure()       puts one kind of panel per stage into a saved figure

Style (decisions 120-122): each model keeps one colour everywhere; all lines are
solid; legends show model names only; ROC legends (colour, name, AUC with 95% CI)
sit inside each panel with the font shrunk until clear of the curves; other plots
have one shared legend under the panels;
calibration curves on a fixed 0-70% range (decision 141) between each model's 1st and 99th
percentile, with a predicted-risk histogram by outcome underneath;
no plotted point rests on a count of 1-9 (NICOR rule); fixed axis ranges (never stretched to the data, so no
minimum or maximum is revealed); PDF (vector) and 300-dpi PNG.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

import nacsa_common as nc
import nacsa_modelling as nm

# ---- style (decision 120) ---------------------------------------------------
MODEL_STYLE = {
    "M0": dict(label="EuroSCORE II", color="#52514e"),
    "M1": dict(label="Logistic regression", color="#4a3aa7"),
    "M2": dict(label="Ridge regression (splines)", color="#2a78d6"),
    "M3": dict(label="XGBoost", color="#eda100"),
    "M4": dict(label="Random forest", color="#008300"),
    "M5": dict(label="Neural network (MLP)", color="#e87ba4"),
}
SHORT_NAME = {"M0": "EuroSCORE II", "M1": "Logistic", "M2": "Ridge", "M3": "XGBoost",
              "M4": "Random forest", "M5": "MLP"}       # for the AUC list inside ROC panels
REFERENCE_GREY = "#a9a8a3"
TEXT = "#0b0b0b"
TEXT_MUTED = "#52514e"
GRID = "#e6e5e1"
LINE_WIDTH = 1.4
STAGE_TITLES = {"preop": "Pre-operative", "end_of_surgery": "Intra-operative",
                "early_postop": "Early post-operative", "complication_aware": "Complication-aware"}
CM = 1 / 2.54
DOUBLE_COLUMN_WIDTH = 17.5 * CM                    # typical double-column width (journal value to verify)
NB_RANGE = (0.10, 0.30)                           # judged range (decision 98)
NB_DISPLAY = (0.0, 0.50)                          # drawn range (decision 98)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7.5, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": TEXT_MUTED, "axes.linewidth": 0.6, "axes.labelcolor": TEXT,
    "xtick.color": TEXT_MUTED, "ytick.color": TEXT_MUTED, "text.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "pdf.fonttype": 42, "savefig.dpi": 300,
})


def _style(model):
    """Every model line is solid; models are told apart by colour (decision 122)."""
    s = MODEL_STYLE[model]
    return dict(color=s["color"], linestyle="-", linewidth=LINE_WIDTH, label=s["label"])


def _square(ax, lim=(0, 1)):
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")


# ---- privacy: no point may rest on a count of 1-9 (NICOR small-number rule) ------
def _safe(*counts):
    """True where none of the counts is between 1 and SMALL_CELL - 1."""
    ok = np.ones_like(np.asarray(counts[0], float), dtype=bool)
    for c in counts:
        c = np.rint(np.asarray(c, float))
        ok &= ~((c >= 1) & (c < nc.SMALL_CELL))
    return ok


# ---- panels -------------------------------------------------------------------
LEGEND_FONT_MAX, LEGEND_FONT_MIN, LEGEND_FONT_STEP = 6.5, 3.5, 0.25


def roc_panel(ax, y, preds, ci=None):
    """ROC curves with the legend INSIDE the panel (decision 122): colour line, model
    name and, if ci is given (model -> (estimate, low, high)), the AUC with 95% CI.
    The legend is placed in the lower-right corner by stage_figure(), which shrinks
    its font until it touches no curve. Returns the plotted points per model."""
    y = np.asarray(y)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    ax.plot([0, 1], [0, 1], color=REFERENCE_GREY, linewidth=0.8)
    drawn, full, short = {}, [], []
    for m, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        tp, fp = tpr * n1, fpr * n0
        keep = _safe(tp, fp, n1 - tp, n0 - fp)
        auc_txt = ""
        if ci and m in ci:
            est, lo, hi = ci[m]
            auc_txt = f"  {est:.3f} ({lo:.3f}-{hi:.3f})"
        st = _style(m)
        st["label"] = MODEL_STYLE[m]["label"] + auc_txt
        ax.plot(fpr[keep], tpr[keep], **st)
        drawn[m] = (fpr[keep], tpr[keep])
        full.append(MODEL_STYLE[m]["label"] + auc_txt)
        short.append(SHORT_NAME[m] + auc_txt)
    _square(ax)
    ax.set_xlabel("1 - specificity")
    ax.set_ylabel("Sensitivity")
    pts = np.vstack([np.column_stack(v) for v in drawn.values()])     # model curves only (not the diagonal)
    ax._roc_legend = dict(full=full, short=short, title="AUC (95% CI)" if ci else None, points=pts)
    return drawn


def _fit_roc_legend(ax):
    """Place the ROC legend in the lower-right corner, shrinking the font (and, only if
    needed, switching to short model names) until no model-curve point lies inside it.
    The legend has a white background, so it may cover the grey chance diagonal."""
    info = ax._roc_legend
    handles = ax.get_lines()[1:]                         # skip the diagonal
    pts_display = ax.transData.transform(info["points"])
    for labels in (info["full"], info["short"]):
        size = LEGEND_FONT_MAX
        while size >= LEGEND_FONT_MIN - 1e-9:
            leg = ax.legend(handles, labels, loc="lower right", fontsize=size, title=info["title"],
                            title_fontsize=size, handlelength=1.6, handletextpad=0.5, borderaxespad=0.3,
                            labelspacing=0.3, alignment="right", frameon=True, facecolor="white",
                            edgecolor="none", framealpha=1.0)              # covers the grey diagonal
            ax.figure.canvas.draw()
            box = leg.get_window_extent()
            inside = ((pts_display[:, 0] >= box.x0) & (pts_display[:, 0] <= box.x1)
                      & (pts_display[:, 1] >= box.y0) & (pts_display[:, 1] <= box.y1))
            if not inside.any():
                return {"fontsize": size, "names": "full" if labels is info["full"] else "short", "overlap": False}
            size -= LEGEND_FONT_STEP
    return {"fontsize": LEGEND_FONT_MIN, "names": "short", "overlap": True}


def pr_panel(ax, y, preds):
    y = np.asarray(y)
    n1 = int(y.sum())
    rate = float(np.mean(y))
    ax.axhline(rate, color=REFERENCE_GREY, linewidth=0.8, label="Event rate")
    drawn = {}
    for m, p in preds.items():
        prec, rec, _ = precision_recall_curve(y, p)
        tp = rec * n1
        with np.errstate(divide="ignore", invalid="ignore"):
            flagged = np.where(prec > 0, tp / prec, 0)
        keep = _safe(tp, flagged - tp, n1 - tp) & (flagged >= nc.SMALL_CELL)
        ax.plot(rec[keep], prec[keep], **_style(m))
        drawn[m] = (rec[keep], prec[keep])
    _square(ax)
    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    return drawn


CALIBRATION_RANGE = (0.0, 0.70)                  # fixed in advance (decision 121); 60% -> 70%, decision 141
CALIBRATION_TRIM = (1, 99)                        # draw curves between these percentiles (decision 121)
HIST_BIN = 0.02


def calibration_panel(ax, y, preds, lim=CALIBRATION_RANGE, hist_ax=None, hist_model=None):
    """Lowess calibration curves (decision 116), drawn between the 1st and 99th percentile
    of each model's predicted risk on a fixed 0-70% range (decisions 121, 141). If hist_ax is
    given, the predicted-risk distribution of `hist_model` (default: Ridge regression if
    present, else the first model) is drawn there by outcome, as % of each group."""
    y = np.asarray(y)
    ax.plot(lim, lim, color=REFERENCE_GREY, linewidth=0.8, label="Perfect calibration")
    drawn, warnings_ = {}, []
    for m, p in preds.items():
        p = np.asarray(p, float)
        lo, hi = np.percentile(p, CALIBRATION_TRIM)
        if hi > lim[1]:
            warnings_.append(f"{MODEL_STYLE[m]['label']}: 99th percentile {100 * hi:.0f}% above the "
                             f"{100 * lim[1]:.0f}% axis limit")
        order = np.argsort(p)
        curve = nm.calibration_curve(y, p)[order]
        ps = p[order]
        keep = (ps >= lo) & (ps <= hi)
        ax.plot(ps[keep], curve[keep], **_style(m))
        drawn[m] = (ps[keep], curve[keep])
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    if hist_ax is None:                     # with a histogram underneath, the figure size keeps it near-square
        ax.set_aspect("equal")
    ax.set_ylabel("Observed proportion (smoothed)")
    if hist_ax is None:
        ax.set_xlabel("Predicted risk")
    else:
        hm = hist_model if hist_model in preds else ("M2" if "M2" in preds else next(iter(preds)))
        p = np.asarray(preds[hm], float)
        edges = np.arange(lim[0], lim[1] + HIST_BIN / 2, HIST_BIN)
        for grp, sign, colour, label in ((1, 1, MODEL_STYLE[hm]["color"], f"Risk distribution, long stay ({SHORT_NAME[hm]})"),
                                         (0, -1, REFERENCE_GREY, f"Risk distribution, no long stay ({SHORT_NAME[hm]})")):
            counts, _ = np.histogram(p[y == grp], bins=edges)
            share = np.where((counts >= 1) & (counts < nc.SMALL_CELL), 0, counts) / max((y == grp).sum(), 1)
            hist_ax.bar(edges[:-1], sign * 100 * share, width=HIST_BIN * 0.9, align="edge",
                        color=colour, linewidth=0, label=label)
        hist_ax.axhline(0, color=TEXT_MUTED, linewidth=0.5)
        hist_ax.set_xlim(*lim)
        hist_ax.set_xlabel("Predicted risk")
        hist_ax.set_ylabel("% of group")
        hist_ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}"))
        hist_ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.tick_params(labelbottom=False)
    for w in warnings_:
        print("WARNING (calibration axis):", w)
    drawn["_warnings"] = warnings_
    return drawn


def decision_panel(ax, y, preds, ylim=(-0.05, None)):
    """Net benefit over 0-50% (decision 98), agreed 10-30% range shaded. A threshold is
    not drawn for a model when its flagged true or false positives number 1-9."""
    y = np.asarray(y)
    t = np.round(np.arange(0.01, NB_DISPLAY[1] + 1e-9, 0.01), 2)
    ax.axvspan(*NB_RANGE, color=GRID, alpha=0.6, linewidth=0, label="Range judged (10-30%)")
    rate = float(np.mean(y))
    ax.plot(t, [nm.net_benefit_treat_all(y, x) for x in t], color=REFERENCE_GREY,
            linewidth=0.9, label="Flag all")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.7, label="Flag none")
    drawn = {}
    for m, p in preds.items():
        p = np.asarray(p, float)
        nb = np.array([nm.net_benefit(y, p, x) for x in t])
        tp = np.array([np.sum((p >= x) & (y == 1)) for x in t])
        fp = np.array([np.sum((p >= x) & (y == 0)) for x in t])
        nb = np.where(_safe(tp, fp), nb, np.nan)
        ax.plot(t, nb, **_style(m))
        drawn[m] = (t, nb)
    ax.set_xlim(*NB_DISPLAY)
    ax.set_ylim(ylim[0], ylim[1] if ylim[1] is not None else max(0.05, rate * 1.1))
    ax.set_xlabel("Threshold risk")
    ax.set_ylabel("Net benefit")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    return drawn


PANELS = {"roc": roc_panel, "pr": pr_panel, "calibration": calibration_panel, "decision": decision_panel}


def stage_figure(kind, by_stage, path_stem, title=None, panel_kwargs=None, arrange="row", letters=False):
    """One panel per stage. by_stage: stage -> (y, preds) or (y, preds, extra kwargs).
    Saves path_stem.pdf and path_stem.png; returns (figure, drawn data per stage)."""
    stages = list(by_stage)
    width = DOUBLE_COLUMN_WIDTH
    if kind == "calibration":
        fig, grid = plt.subplots(2, len(stages), figsize=(width, width / len(stages) * 1.75),
                                 layout="constrained", gridspec_kw={"height_ratios": [4, 1.2]},
                                 sharex="col", squeeze=False)
        axes, hist_axes = grid[0], grid[1]
    else:
        if arrange == "column":                         # panels stacked, each about 9 cm wide
            side = 9 * CM
            fig, axes = plt.subplots(len(stages), 1, figsize=(side, side * len(stages) * 0.95),
                                     layout="constrained", squeeze=False)
            axes = axes[:, 0]
        else:
            fig, axes = plt.subplots(1, len(stages), figsize=(width, width / len(stages) * 1.3),
                                     layout="constrained", squeeze=False)
            axes = axes[0]
        hist_axes = [None] * len(stages)
    drawn = {}
    for ax, hax, stage in zip(axes, hist_axes, stages):
        y, preds, *extra = by_stage[stage]
        kw = dict(panel_kwargs or {})
        if extra:
            kw.update(extra[0])
        if kind == "calibration":
            kw["hist_ax"] = hax
        drawn[stage] = PANELS[kind](ax, np.asarray(y), preds, **kw)
        name = STAGE_TITLES.get(stage, stage)
        ax.set_title(f"{'ABCDEF'[stages.index(stage)]}   {name}" if letters else name, loc="left", fontweight="bold")
    if kind == "roc":
        # legends inside each panel, font shrunk until clear of the curves (decision 122)
        fig.canvas.draw()
        for ax, stage in zip(axes, stages):
            drawn[stage]["_legend"] = _fit_roc_legend(ax)
            if drawn[stage]["_legend"]["overlap"]:
                print(f"WARNING (ROC legend): {stage} legend still touches a curve at the smallest font")
    else:
        # one shared legend under the panels (models are the same in every panel)
        handles, labels = axes[0].get_legend_handles_labels()
        if hist_axes[0] is not None:
            h2, l2 = hist_axes[0].get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(labels), 3 if len(labels) > 6 else 4),
                   handlelength=2.6)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=9)
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png")
    plt.close(fig)
    return fig, drawn
