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
import matplotlib.ticker
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import normalize

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
    np.seterr(over="ignore")
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
def answer_in_subject(subj, answers_json):
    """True if a gold alias (>2 chars) appears verbatim in the subject name,
    e.g. subject 'County Durham', object 'Durham'. Such items can be solved
    by copying from the prompt, so they are excluded in a robustness check."""
    s = f" {normalize(str(subj))} "
    for a in json.loads(answers_json):
        na = normalize(a)
        if len(na) > 2 and f" {na} " in s:
            return True
    return False


def load(results_dir):
    files = [f for f in glob.glob(os.path.join(results_dir, "*.csv"))]
    if not files:
        raise SystemExit(f"No CSVs in {results_dir}")
    # Per-position vectors like "011" must be read as strings, otherwise pandas
    # parses them as integers and drops the leading zeros.
    df = pd.concat([pd.read_csv(f, dtype={"tf_vec": str, "copy_vec": str})
                    for f in files], ignore_index=True)
    for c in ["gen_correct", "tf_all", "tf_first", "copy_all", "copy_first"]:
        df[c] = df[c].astype(str).str.lower().isin(["true", "1"]).astype(int)
    df["gen"] = df["gen"].fillna("").astype(str)
    df["leak"] = [answer_in_subject(s, a) for s, a in zip(df["subj"], df["answers"])]
    # Few-shot confound: some gold answers coincide with a demonstration answer
    # (e.g. 'Paris' as place of birth). A model that simply repeats a
    # demonstration answer is then right by chance, and such answers are
    # mostly short. We control for this and also exclude these items.
    demos_path = os.path.join(results_dir, "demos.json")
    demos = json.load(open(demos_path)) if os.path.exists(demos_path) else {}
    demo_norm = {r: {normalize(o) for _, o in v} for r, v in demos.items()}
    df["gold_is_demo"] = [int(any(normalize(a) in demo_norm.get(p, set()) for a in json.loads(ans)))
                          for p, ans in zip(df["prop"], df["answers"])]
    df["gen_is_demo"] = [int(any(normalize(g).startswith(o) for o in demo_norm.get(p, set()) if o))
                         for p, g in zip(df["prop"], df["gen"])]
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
    panels = [("gen_correct", "(a) Factual recall"), ("copy_all", "(b) In-context control (answer given)")]
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
    """Accuracy by object length within quartiles of OBJECT popularity, the
    strongest single predictor and the main competing explanation."""
    d = df[df.short == model].copy()
    d["pop_q"] = pd.qcut(d["log_o_pop"], q, labels=False, duplicates="drop")
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
    ax.set_title(f"Within object-popularity quartiles ({model})", fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    save(fig, out, "fig2_popularity_strata")


def fig_positional(df, models, out, min_n=30):
    """(a) First-token accuracy by object length, recall vs. control.
    (b) Probability of completing the answer once the first token is right."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    ticks = range(1, TOK_CAP + 1)
    labels = [str(t) if t < TOK_CAP else f"{t}+" for t in ticks]
    for i, m in enumerate(models):
        d = df[df.short == m]
        c = RAMP[min(i, len(RAMP) - 1)]
        x, y, _, _ = curve(d, "tf_first", min_n)
        axes[0].plot(x, y, color=c, lw=2, marker=MARKERS[i % 5], ms=5, label=m)
        if i == len(models) - 1:
            x, y, _, _ = curve(d, "copy_first", min_n)
            axes[0].plot(x, y, color=CONTROL, lw=2, ls="--", marker=MARKERS[i % 5], ms=5,
                         label=f"{m}, in-context control")
        dd = d[(d.n_obj_tok >= 2) & (d.tf_first == 1)]
        x, y, _, _ = curve(dd, "tf_all", min_n)
        axes[1].plot(x, y, color=c, lw=2, marker=MARKERS[i % 5], ms=5, label=m)
    axes[0].set_title("(a) First token correct", fontsize=9, loc="left")
    axes[1].set_title("(b) Full answer correct | first token correct", fontsize=9, loc="left")
    for ax in axes:
        style(ax)
        ax.set_xticks(list(ticks)); ax.set_xticklabels(labels)
        ax.set_xlabel("Object length (subword tokens)", fontsize=8)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("Top-1 accuracy (teacher-forced)", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [l.replace("-deduped", "") for l in labels], fontsize=7, frameon=False,
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    save(fig, out, "fig3_positional")


def fig_forest(rows, models, out):
    """Odds ratios (95% CI) with relation and popularity-decile controls:
    (a) per additional object token, (b) per additional within-word split."""
    series = [("gen", "Recall, generation (aliases accepted)", RAMP[2], "o"),
              ("strict", "Recall, teacher-forced exact", "#1baf7a", "s"),
              ("ctrl", "In-context control, teacher-forced exact", CONTROL, "D")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    ys = np.arange(len(models))[::-1]
    for ax, prefix, title in [(axes[0], "", "(a) Per additional object token"),
                              (axes[1], "split_", "(b) Per additional within-word split")]:
        for j, (key, lab, col, mk) in enumerate(series):
            off = (1 - j) * 0.22
            for y, m in zip(ys, models):
                r = rows[m][prefix + key]
                ax.plot([r["lo"], r["hi"]], [y + off] * 2, color=col, lw=2)
                ax.plot(r["OR"], y + off, marker=mk, color=col, ms=5, linestyle="none",
                        label=lab if m == models[0] else None)
        ax.axvline(1, color="#999999", lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.25])
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style(ax)
        ax.grid(axis="x", color="#e5e5e5", linewidth=0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xlabel("Odds ratio (log scale)", fontsize=8)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([m.replace("-deduped", "") for m in models], fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "fig4_odds_ratios")


def save(fig, out, name):
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ["pdf", "png"]:
        fig.savefig(os.path.join(out, "figures", f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ================================================================ main
LINEAR = ["obj_tok", "subj_tok", "gold_is_demo", "log_s_pop", "log_o_pop"]
N_BINS = 10  # popularity deciles: the popularity-accuracy relation is strongly non-linear


def pop_bins(d, k=N_BINS):
    """Add dummy columns for quantile bins of subject and object popularity."""
    d = d.copy()
    names = []
    for col in ["log_s_pop", "log_o_pop"]:
        q = pd.qcut(d[col].rank(method="first"), k, labels=False)
        for j in range(1, k):
            nm = f"{col}_b{j}"
            d[nm] = (q == j).astype(float)
            names.append(nm)
    return d, ["obj_tok", "subj_tok", "gold_is_demo"] + names


def or_row(res, term="obj_tok"):
    r = res.set_index("term").loc[term]
    return {k: float(r[k]) for k in ["OR", "lo", "hi", "p", "coef", "se"]}


def same_metric_contrast(d, covars):
    """Stack recall and control rows (both teacher-forced exact on the same
    target) and test whether the object-length slope differs between them.
    Relation x condition fixed effects; SEs clustered by fact."""
    a = d.assign(y=d["tf_all"], cond=0)
    b = d.assign(y=d["copy_all"], cond=1)
    s = pd.concat([a, b], ignore_index=True)
    s["prop"] = s["prop"] + "|" + s["cond"].astype(str)
    s["tok_x_recall"] = s["obj_tok"] * (1 - s["cond"])
    res, _, _ = fit_table(s, "y", covars + ["tok_x_recall"], cluster="fact")
    return {"control": or_row(res, "obj_tok"), "ratio": or_row(res, "tok_x_recall")}


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
    first = df[df.short == models[0]]
    summary = {"models": models, "n_facts": int(len(first)),
               "token_count_distribution": first["obj_tok"].value_counts().sort_index().to_dict(),
               "answer_in_subject_rate": float(first["leak"].mean()),
               "gold_equals_demo_rate": float(first["gold_is_demo"].mean()),
               "gold_equals_demo_by_tokens": first.groupby("obj_tok")["gold_is_demo"].mean().round(3).to_dict(),
               "spearman": first[["n_obj_tok", "log_o_pop", "log_s_pop", "n_subj_tok"]]
               .corr(method="spearman").round(3).to_dict()}

    tex_table(accuracy_table(df, models, "gen_correct"),
              "Factual recall accuracy (generation, aliases accepted) by object length; "
              "unadjusted, n in parentheses.",
              "tab:acc", os.path.join(tdir, "acc_by_tokens.tex"))
    tex_table(accuracy_table(df, models, "copy_all"),
              "In-context control: teacher-forced exact match by object length when the fact "
              "is stated in the prompt; unadjusted.",
              "tab:copy", os.path.join(tdir, "copy_by_tokens.tex"))

    main_rows, pop_rows, split_rows, pos_rows, rob_rows = [], [], [], [], []
    forest = {}
    summary["per_model"] = {}
    for m in models:
        d = df[df.short == m]
        db, cov = pop_bins(d)

        gen, n, nrel = fit_table(db, "gen_correct", cov)
        strict, _, _ = fit_table(db, "tf_all", cov)
        con = same_metric_contrast(db, cov)
        forest[m] = {"gen": or_row(gen), "strict": or_row(strict), "ctrl": con["control"]}
        main_rows.append({"Model": m,
                          "Recall, generation": fmt_or(pd.Series(forest[m]["gen"])),
                          "Recall, exact": fmt_or(pd.Series(forest[m]["strict"])),
                          "Control, exact": fmt_or(pd.Series(con["control"])),
                          "Recall/control ratio": fmt_or(pd.Series(con["ratio"])),
                          "n": n})

        lin, _, _ = fit_table(d, "gen_correct", LINEAR)
        r = lin.set_index("term")
        pop_rows.append({"Model": m,
                         "OR / 10x obj. pop.": fmt_or(r.loc["log_o_pop"]),
                         "OR / 10x subj. pop.": fmt_or(r.loc["log_s_pop"]),
                         "OR / subject token": fmt_or(r.loc["subj_tok"])})

        split_cov = ["obj_words", "extra_tok"] + cov[1:]
        spl = {}
        for y, tag in [("gen_correct", "gen"), ("tf_all", "exact"), ("copy_all", "ctrl")]:
            spl[tag] = fit_table(db, y, split_cov)[0].set_index("term")
        forest[m]["split_gen"] = or_row(spl["gen"].reset_index(), "extra_tok")
        forest[m]["split_strict"] = or_row(spl["exact"].reset_index(), "extra_tok")
        forest[m]["split_ctrl"] = or_row(spl["ctrl"].reset_index(), "extra_tok")
        split_rows.append({"Model": m,
                           "Gen.: word": fmt_or(spl["gen"].loc["obj_words"]),
                           "Gen.: split": fmt_or(spl["gen"].loc["extra_tok"]),
                           "Exact: word": fmt_or(spl["exact"].loc["obj_words"]),
                           "Exact: split": fmt_or(spl["exact"].loc["extra_tok"]),
                           "Control: word": fmt_or(spl["ctrl"].loc["obj_words"]),
                           "Control: split": fmt_or(spl["ctrl"].loc["extra_tok"])})

        mt = d[d.n_obj_tok >= 2]
        fails = mt[mt.tf_all == 0]
        pos = {"first_token_recall": float(mt["tf_first"].mean()),
               "first_token_control": float(mt["copy_first"].mean()),
               "complete_given_first": float(mt.loc[mt.tf_first == 1, "tf_all"].mean())
               if mt.tf_first.sum() else float("nan"),
               "failures_at_first_token": float((fails["tf_first_fail"] == 0).mean())}
        pos_rows.append({"Model": m,
                         "First token (recall)": f"{pos['first_token_recall']:.3f}",
                         "First token (control)": f"{pos['first_token_control']:.3f}",
                         "Complete | first correct": f"{pos['complete_given_first']:.3f}",
                         "Failures at first token": f"{pos['failures_at_first_token']:.3f}"})

        rob = {}
        for y, tag in [("gen_correct", "gen"), ("tf_all", "exact")]:
            rob[f"{tag}_linear_pop"] = or_row(fit_table(d, y, LINEAR)[0])
            rob[f"{tag}_quartile_pop"] = or_row(fit_table(*pop_bins(d, 4)[:1], y, pop_bins(d, 4)[1])[0])
            rob[f"{tag}_decile_pop"] = or_row(gen if tag == "gen" else strict)
            rob[f"{tag}_no_answer_in_subject"] = or_row(fit_table(db[db.leak == 0], y, cov)[0])
            rob[f"{tag}_1_to_4_tokens"] = or_row(fit_table(db[db.n_obj_tok <= 4], y, cov)[0])
            rob[f"{tag}_no_gold_demo"] = or_row(fit_table(db[db.gold_is_demo == 0], y, cov)[0])
            rob[f"{tag}_split_no_gold_demo"] = or_row(
                fit_table(db[db.gold_is_demo == 0], y, ["obj_words", "extra_tok"] + cov[1:])[0],
                "extra_tok")
        f2 = lambda k: f"{rob[k]['OR']:.2f}"
        for tag, lab in [("gen", "generation"), ("exact", "exact")]:
            rob_rows.append({"Model": m, "Metric": lab,
                             "Linear log pop.": f2(f"{tag}_linear_pop"),
                             "Pop. quartiles": f2(f"{tag}_quartile_pop"),
                             "Pop. deciles (main)": f2(f"{tag}_decile_pop"),
                             "Excl. answer in subject": f2(f"{tag}_no_answer_in_subject"),
                             "1--4 tokens only": f2(f"{tag}_1_to_4_tokens"),
                             "Excl. gold = demo answer": f2(f"{tag}_no_gold_demo"),
                             "Split, excl. gold = demo": f2(f"{tag}_split_no_gold_demo")})

        summary["per_model"][m] = {
            "accuracy_generation": float(d["gen_correct"].mean()),
            "accuracy_exact": float(d["tf_all"].mean()),
            "control_exact": float(d["copy_all"].mean()),
            "n_used_in_regression": int(n), "relations_used": int(nrel),
            "obj_token_OR": forest[m], "recall_control_ratio": con["ratio"],
            "linear_popularity_model": lin.round(4).to_dict("records"),
            "words_vs_fragmentation": {k: v.reset_index().round(4).head(3).to_dict("records")
                                       for k, v in spl.items()},
            "positional": pos, "robustness": rob,
            "wrong_answers_copying_a_demo_answer": float(d.loc[d.gen_correct == 0, "gen_is_demo"].mean()),
        }

    note = (" Odds ratios [95\\% CI]; relation fixed effects; *p<.05, **p<.01, ***p<.001.")
    tex_table(pd.DataFrame(main_rows),
              "Effect of one additional object token on the odds of a correct answer, controlling "
              "for subject length, subject- and object-popularity deciles, whether the gold answer "
              "equals a demonstration answer, and relation. The ratio "
              "column tests whether the slope for recall differs from the in-context control "
              "(same exact-match metric; fact-clustered SEs)." + note,
              "tab:main", os.path.join(tdir, "main_effects.tex"))
    tex_table(pd.DataFrame(pop_rows),
              "Other predictors (generation metric, linear log$_{10}$ popularity)." + note,
              "tab:pop", os.path.join(tdir, "popularity.tex"))
    tex_table(pd.DataFrame(split_rows),
              "Decomposing answer length into the number of words and the number of extra "
              "subword splits within words (tokens minus words), for recall under both "
              "metrics and for the in-context control (popularity deciles)." + note,
              "tab:split", os.path.join(tdir, "words_vs_fragmentation.tex"))
    tex_table(pd.DataFrame(pos_rows),
              "Objects of two or more tokens: where recall fails (teacher-forced).",
              "tab:pos", os.path.join(tdir, "positional.tex"))
    tex_table(pd.DataFrame(rob_rows),
              "Robustness of the odds ratio per object token to the popularity control, "
              "to excluding items whose answer appears in the subject name or equals a "
              "demonstration answer, and to restricting to 1--4-token objects. The last column "
              "is the within-word split effect when demonstration-answer items are excluded.", "tab:rob", os.path.join(tdir, "robustness.tex"))

    if len(models) > 1:
        dd, cov = pop_bins(df)
        dd["scale_c"] = dd["log_params"] - dd["log_params"].mean()
        dd["tok_x_scale"] = dd["obj_tok"] * dd["scale_c"]
        for y in ["gen_correct", "tf_all"]:
            res, _, _ = fit_table(dd, y, cov + ["scale_c", "tok_x_scale"], cluster="fact")
            summary[f"pooled_scale_interaction_{y}"] = or_row(res, "tok_x_scale")

    fig_by_model(df, models, args.out, args.min_n)
    fig_popularity(df, models[-1], args.out, args.min_n)
    fig_positional(df, models, args.out, args.min_n)
    fig_forest(forest, models, args.out)

    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print("wrote tables/, figures/ and", os.path.join(args.results, "summary.json"))


if __name__ == "__main__":
    main()
