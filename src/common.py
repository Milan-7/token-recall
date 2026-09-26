"""Model-free helpers: data loading, prompt construction, answer matching.

Kept separate from probe.py so they can be unit-tested without torch.
"""
import json
import random
import re

from templates import TEMPLATES


# ---------------------------------------------------------------- parsing
def as_list(x):
    """PopQA stores alias lists as JSON strings; accept strings, lists, arrays."""
    if x is None:
        return []
    if isinstance(x, str):
        try:
            v = json.loads(x)
        except (json.JSONDecodeError, ValueError):
            return [x]
        return [str(a) for a in v] if isinstance(v, list) else [str(v)]
    return [str(a) for a in list(x)]


def normalize(s):
    """SQuAD-style normalisation: lowercase, strip punctuation and articles."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def is_match(generation, answers):
    """True if any normalised gold alias occurs as a whole-word span in the
    normalised generation (the substring criterion of Mallen et al., 2023)."""
    g = f" {normalize(generation)} "
    for a in answers:
        na = normalize(a)
        if na and f" {na} " in g:
            return True
    return False


# ---------------------------------------------------------------- data
REQUIRED_COLS = ["subj", "prop", "obj", "s_pop", "o_pop", "possible_answers"]


def prepare_popqa(df, n=None, seed=0, n_demos=3):
    """Filter to templated relations, hold out demonstrations, optionally subsample.

    Returns (eval_df, demos) where demos[rel] = [(subj, obj), ...].
    """
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"PopQA columns changed; missing {missing}. "
                         f"Found: {list(df.columns)}")
    df = df[df["prop"].isin(TEMPLATES)].copy()
    df["answers"] = [
        list(dict.fromkeys([o] + as_list(a)))  # canonical object first, dedup
        for o, a in zip(df["obj"], df["possible_answers"])
    ]

    rng = random.Random(seed)
    demos, held_out = {}, []
    for rel, g in df.groupby("prop"):
        idx = rng.sample(sorted(g.index), n_demos)
        demos[rel] = [(df.at[i, "subj"], df.at[i, "obj"]) for i in idx]
        held_out += idx
    df = df.drop(index=held_out)

    if n is not None and n < len(df):
        df = df.sample(n=n, random_state=seed)
    return df.reset_index(drop=True), demos


# ---------------------------------------------------------------- prompts
def fact_prompt(rel, subj, demos):
    """Few-shot completion prompt. Demonstrations use the same relation."""
    t = TEMPLATES[rel]
    lines = [f"{t.format(s=ds)} {do}." for ds, do in demos[rel]]
    lines.append(t.format(s=subj))
    return "\n".join(lines)


def copy_prompt(rel, subj, obj, demos):
    """In-context (open-book) control: identical prompt and target to the
    factual probe, but the complete fact is stated once at the top. The model
    only has to copy the answer from context, not retrieve it from its
    parameters, so answer length should matter little if the pipeline is
    unbiased."""
    context = f"{TEMPLATES[rel].format(s=subj)} {obj.strip()}."
    return context + "\n" + fact_prompt(rel, subj, demos)


def target(text):
    """Continuation string. The leading space matters for BPE tokenisers:
    ' Paris' and 'Paris' are different tokens."""
    return " " + text.strip()
