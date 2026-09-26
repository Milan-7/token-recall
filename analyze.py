"""Analyse results/*.csv -> tables/*.tex, figures/*.pdf, results/summary.json.

Usage:
    python src/analyze.py --results results --out .
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

TOK_CAP = 6  # object token counts >= 6 are pooled into "6+"

# Ordinal blue ramp (light -> dark = small -> large); one accent for the control.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b", "#081f40"]
CONTROL = "#eb6834"
MARKERS = ["o", "s", "^", "D", "v"]


# ================================================================ statistics
def logit_fit(y, X, cluster=None, l2=1e-6, max_iter=100):
    """Logistic regression by Newton-Raphson.

    A tiny ridge (l2) only stabilises the Hessian; it does not change
    well-identified coefficients. If `cluster` is given, standard errors are
    cluster-robust (sandwich), used when the same fact appears once per model.
    """
    n, k = X.shape
    b = np.zeros(k)
    pen = np.full(k, l2)
    pen[0] = 0.0  # intercept unpenalised
    for _ in range(max_iter):
        p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + np.diag(pen)
        g = X.T @ (y - p) - pen * b
        step = np.linalg.solve(H, g)
        b += step
        if np.max(np.abs(step)) < 1e-9:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30)))
    Hinv = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen))
    if cluster is None:
        cov = Hinv
    else:
        s = X * (y - p)[:, None]
        meat = np.zeros((k, k))
        for g_ in np.unique(cluster):
            sg = s[cluster == g_].sum(0)
            meat += np.outer(sg, sg)
        G = len(np.unique(cluster))
        cov = Hinv @ meat @ Hinv * G / (G - 1)
    se = np.sqrt(np.diag(cov))
    z = b / se
    return b, se, 2 * norm.sf(np.abs(z))


def design(df, numeric, fe="prop"):
    """Intercept + numeric covariates + relation fixed effects (drop-first)."""
    X = [np.ones(len(df))]
    names = ["const"]
    for c in numeric:
        X.append(df[c].to_numpy(float))
        names.append(c)
    if fe:
        levels = sorted(df[fe].unique())
        for lv in levels[1:]:
            X.append((df[fe] == lv).to_numpy(float))
            names.append(f"{fe}={lv}")
    return np.column_stack(X), names


def drop_constant_relations(df, y):
    """Relations where a model is always right or always wrong carry no
    within-relation information under fixed effects (as in conditional logit)."""
    m = df.groupby("prop")[y].transform("mean")
    return df[(m > 0) & (m < 1)]


def fit_table(df, y, numeric, cluster=None):
    d = drop_constant_relations(df, y)
    X, names = design(d, numeric)
    b, se, p = logit_fit(d[y].to_numpy(float), X,
                         cluster=None if cluster is None else d[cluster].to_numpy())
    rows = []
    for j, nm in enumerate(names):
        if nm in numeric:
            rows.append(dict(term=nm, coef=b[j], se=se[j], p=p[j],
                             OR=np.exp(b[j]), lo=np.exp(b[j] - 1.96 * se[j]),
                             hi=np.exp(b[j] + 1.96 * se[j])))
    return pd.DataFrame(rows), len(d), d["prop"].nunique()


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    ph = k / n
    den = 1 + z**2 / n
    c = (ph + z**2 / (2 * n)) / den
    h = z * np.sqrt(ph * (1 - ph) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


# ================================================================ data
def load(results_dir):
    files = [f for f in glob.glob(os.path.join(results_dir, "*.csv"))]
    if not files:
        raise SystemExit(f"No CSVs in {results_dir}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ["gen_correct", "tf_all", "tf_first", "copy_all", "copy_first"]:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1"]).astype(int)
    df["log_s_pop"] = np.log10(df["s_pop"].clip(lower=0) + 1)
    df["log_o_pop"] = np.log10(df["o_pop"].clip(lower=0) + 1)
    df["obj_tok"] = df["n_obj_tok"].clip(upper=TOK_CAP)
    df["subj_tok"] = df["n_subj_tok"].clip(upper=TOK_CAP)
    df["obj_words"] = df["n_obj_words"].clip(upper=TOK_CAP)
    df["extra_tok"] = (df["n_obj_tok"] - df["n_obj_words"]).clip(lower=0, upper=TOK_CAP)
    df["log_params"] = np.log10(df["n_params"])
    df["short"] = df["model"].str.split("/").str[-1]
    df["fact"] = df["prop"] + "|" + df["subj"] + "|" + df["obj"]
    return df


def model_order(df):
    return (df.groupby("short")["n_params"].first().sort_values().index.tolist())


# ================================================================ tables
def fmt_or(r):
    star = "***" if r.p < .001 else "**" if r.p < .01 else "*" if r.p < .05 else ""
    return f"{r.OR:.2f}{star} [{r.lo:.2f}, {r.hi:.2f}]"


def tex_table(df, caption, label, path):
    cols = "l" + "r" * (df.shape[1] - 1)
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\resizebox{\linewidth}{!}{%",
             rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join(df.columns) + r" \\", r"\midrule"]
    for _, row in df.iterrows():
        lines.append(" & ".join(str(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    with open(path, "w") as f:
        f.write("\n".join(lines).replace("_", r"\_").replace(r"\_{", "_{"))


def accuracy_table(df, models, ycol):
    rows = []
    for t in range(1, TOK_CAP + 1):
        row = {"Object tokens": f"{t}+" if t == TOK_CAP else str(t)}
        for m in models:
            d = df[(df.short == m) & (df.obj_tok == t)]
            row[m] = f"{d[ycol].mean():.3f} ({len(d)})" if len(d) else "--"
        rows.append(row)
    return pd.DataFrame(rows)


# ================================================================ figures
def style(ax):
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#999999")
    ax.tick_params(colors="#444444", labelsize=8)


def curve(df, ycol, min_n):
    xs, ys, lo, hi = [], [], [], []
    for t in range(1, TOK_CAP + 1):
        d = df[df.obj_tok == t]
        if len(d) < min_n:
            continue
        k = d[ycol].sum()
        l, h = wilson(k, len(d))
        xs.append(t); ys.append(k / len(d)); lo.append(l); hi.append(h)
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi)


def fig_by_model(df, models, out, min_n):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    panels = [("gen_correct", "(a) Factual recall"), ("copy_all", "(b) Copy control")]
    for ax, (ycol, title) in zip(axes, panels):
        for i, m in enumerate(models):
            x, y, lo, hi = curve(df[df.short == m], ycol, min_n)
            c = RAMP[min(i, len(RAMP) - 1)] if len(models) > 1 else RAMP[1]
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=c, lw=2, marker=MARKERS[i % 5], ms=5, label=m)
        style(ax)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks(range(1, TOK_CAP + 1))
        ax.set_xticklabels([str(t) if t < TOK_CAP else f"{t}+" for t in range(1, TOK_CAP + 1)])
        ax.set_xlabel("Object length (subword tokens)", fontsize=8)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Accuracy", fontsize=8)
    axes[1].legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, out, "fig1_accuracy_by_tokens")


def fig_popularity(df, model, out, min_n, q=4):
    d = df[df.short == model].copy()
    d["pop_q"] = pd.qcut(d["log_s_pop"], q, labels=False, duplicates="drop")
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    nq = int(d["pop_q"].max()) + 1
    for qi in range(nq):
        x, y, lo, hi = curve(d[d.pop_q == qi], "gen_correct", min_n)
        c = RAMP[min(qi, len(RAMP) - 1)]
        ax.plot(x, y, color=c, lw=2, marker=MARKERS[qi % 5], ms=5,
                label=f"Q{qi+1}" + (" (rarest)" if qi == 0 else " (most popular)" if qi == nq - 1 else ""))
    style(ax)
    ax.set_xticks(range(1, TOK_CAP + 1))
    ax.set_xticklabels([str(t) if t < TOK_CAP else f"{t}+" for t in range(1, TOK_CAP + 1)])
    ax.set_xlabel("Object length (subword tokens)", fontsize=8)
    ax.set_ylabel("Accuracy", fontsize=8)
    ax.set_title(f"Within subject-popularity quartiles ({model})", fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, out, "fig2_popularity_strata")


def fig_positional(df, models, out, max_k=5, min_n=30):
    """P(token k correct | gold prefix) for multi-token objects."""
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for i, m in enumerate(models):
        vecs = df[(df.short == m) & (df.n_obj_tok >= 2)]["tf_vec"].astype(str)
        xs, ys = [], []
        for k in range(max_k):
            v = [s[k] == "1" for s in vecs if len(s) > k]
            if len(v) >= min_n:
                xs.append(k + 1); ys.append(np.mean(v))
        c = RAMP[min(i, len(RAMP) - 1)]
        ax.plot(xs, ys, color=c, lw=2, marker=MARKERS[i % 5], ms=5, label=m)
    style(ax)
    ax.set_xlabel("Position within object (token index)", fontsize=8)
    ax.set_ylabel("Top-1 accuracy given gold prefix", fontsize=8)
    ax.set_title("Where multi-token recall fails", fontsize=8, loc="left")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    save(fig, out, "fig3_positional")


def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default=".")
    ap.add_argument("--min_n", type=int, default=30, help="min facts per plotted bin")
    args = ap.parse_args()

    df = load(args.results)
    models = model_order(df)
    tdir = os.path.join(args.out, "tables")
    os.makedirs(tdir, exist_ok=True)
    summary = {"models": models, "n_facts": int(df.groupby("short").size().iloc[0]),
               "token_count_distribution": df[df.short == models[0]]["obj_tok"]
               .value_counts().sort_index().to_dict()}

    # --- descriptive accuracy
    tex_table(accuracy_table(df, models, "gen_correct"),
              "Factual recall accuracy by object length (n in parentheses).",
              "tab:acc", os.path.join(tdir, "acc_by_tokens.tex"))
    tex_table(accuracy_table(df, models, "copy_all"),
              "Copy-control exact-match accuracy by object length.",
              "tab:copy", os.path.join(tdir, "copy_by_tokens.tex"))

    # --- per-model regressions
    base = ["obj_tok", "subj_tok", "log_s_pop", "log_o_pop"]
    split = ["obj_words", "extra_tok", "subj_tok", "log_s_pop", "log_o_pop"]
    reg_rows, split_rows = [], []
    summary["per_model"] = {}
    for m in models:
        d = df[df.short == m]
        res, n, nrel = fit_table(d, "gen_correct", base)
        ctrl, n_c, _ = fit_table(d.assign(prop="all"), "copy_all", base) \
            if 0 < d["copy_all"].mean() < 1 else (None, 0, 0)
        spl, _, _ = fit_table(d, "gen_correct", split)
        r = res.set_index("term")
        reg_rows.append({"Model": m,
                         "OR / object token": fmt_or(r.loc["obj_tok"]),
                         "OR / subject token": fmt_or(r.loc["subj_tok"]),
                         "OR / 10x subj. pop.": fmt_or(r.loc["log_s_pop"]),
                         "Copy ctrl: OR / obj. token":
                             fmt_or(ctrl.set_index("term").loc["obj_tok"]) if ctrl is not None else "ceiling",
                         "n": n})
        s = spl.set_index("term")
        split_rows.append({"Model": m,
                           "OR / extra word": fmt_or(s.loc["obj_words"]),
                           "OR / extra subword split": fmt_or(s.loc["extra_tok"])})
        summary["per_model"][m] = {
            "accuracy": float(d["gen_correct"].mean()),
            "tf_exact": float(d["tf_all"].mean()),
            "copy_exact": float(d["copy_all"].mean()),
            "n_used_in_regression": int(n), "relations_used": int(nrel),
            "regression": res.round(4).to_dict("records"),
            "words_vs_fragmentation": spl.round(4).to_dict("records"),
            "copy_control_regression": None if ctrl is None else ctrl.round(4).to_dict("records"),
        }
    tex_table(pd.DataFrame(reg_rows),
              "Logistic regression of recall on object/subject length and popularity, "
              "with relation fixed effects. Odds ratios [95\\% CI]; *p<.05, **p<.01, ***p<.001.",
              "tab:reg", os.path.join(tdir, "regression.tex"))
    tex_table(pd.DataFrame(split_rows),
              "Separating answer length in words from subword fragmentation within words.",
              "tab:split", os.path.join(tdir, "words_vs_fragmentation.tex"))

    # --- pooled scale interaction (cluster-robust by fact)
    if len(models) > 1:
        dd = df.copy()
        dd["tok_x_scale"] = dd["obj_tok"] * (dd["log_params"] - dd["log_params"].mean())
        dd["scale_c"] = dd["log_params"] - dd["log_params"].mean()
        res, n, _ = fit_table(dd, "gen_correct", base + ["scale_c", "tok_x_scale"], cluster="fact")
        summary["pooled_scale_interaction"] = res.round(4).to_dict("records")

    # --- positional failure summary
    summary["positional"] = {}
    for m in models:
        d = df[(df.short == m) & (df.n_obj_tok >= 2)]
        summary["positional"][m] = {
            "n_multi_token": int(len(d)),
            "first_token_correct": float(d["tf_first"].mean()),
            "all_correct_given_first": float(d.loc[d.tf_first == 1, "tf_all"].mean())
            if d.tf_first.sum() else None,
        }

    # --- figures
    fig_by_model(df, models, args.out, args.min_n)
    fig_popularity(df, models[-1], args.out, args.min_n)
    fig_positional(df, models, args.out)

    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str)[:4000])


if __name__ == "__main__":
    main()
