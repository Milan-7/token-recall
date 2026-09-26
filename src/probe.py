"""Run the factual-recall probe and the in-context control for one model.

Usage (Colab / any GPU):
    python src/probe.py --model EleutherAI/pythia-410m-deduped --n 5000

Writes results/<model-short-name>.csv with one row per fact.
"""
import argparse
import json
import os
import time

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import copy_prompt, fact_prompt, is_match, prepare_popqa, target


# ---------------------------------------------------------------- scoring
@torch.no_grad()
def teacher_forced(model, tok, prompts, targets, device, batch_size):
    """Score each gold target token given the prompt and the gold prefix.

    For target tokens t_1..t_n returns, per example:
      tf_first    : argmax at position 1 equals t_1
      tf_all      : argmax equals gold at every position (= greedy would emit it)
      tf_first_fail: index of the first wrong position, -1 if none
      tf_sum_lp / tf_mean_lp : summed / per-token log-probability of the target
      tf_first_rank: rank of t_1 in the vocabulary (0 = top)
      tf_vec      : per-position correctness as a string, e.g. "110"
    """
    p_ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
    t_ids = [tok(t, add_special_tokens=False).input_ids for t in targets]
    order = sorted(range(len(prompts)), key=lambda i: len(p_ids[i]) + len(t_ids[i]))
    pad = tok.pad_token_id
    out = [None] * len(prompts)

    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        seqs = [p_ids[i] + t_ids[i] for i in idx]
        L = max(len(s) for s in seqs)
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, s in enumerate(seqs):  # right padding: real tokens keep positions 0..len-1
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits

        for r, i in enumerate(idx):
            P, T = len(p_ids[i]), t_ids[i]
            rows = logits[r, P - 1:P - 1 + len(T)].float().log_softmax(-1)
            gold = torch.tensor(T, device=rows.device)
            lp = rows.gather(1, gold[:, None]).squeeze(1)
            correct = (rows.argmax(-1) == gold).tolist()
            first_rank = int((rows[0] > lp[0]).sum())
            first_fail = next((k for k, c in enumerate(correct) if not c), -1)
            out[i] = dict(
                tf_first=bool(correct[0]),
                tf_all=bool(all(correct)),
                tf_first_fail=first_fail,
                tf_sum_lp=float(lp.sum()),
                tf_mean_lp=float(lp.mean()),
                tf_first_rank=first_rank,
                tf_vec="".join("1" if c else "0" for c in correct),
            )
    return out


@torch.no_grad()
def generate(model, tok, prompts, device, batch_size, max_new_tokens):
    """Greedy decoding; returns the first generated line for each prompt."""
    tok.padding_side = "left"
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    out = [None] * len(prompts)
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for i, t in zip(idx, texts):
            out[i] = t.split("\n")[0].strip()
    tok.padding_side = "right"
    return out


def check_tokenisation(tok, prompts, targets, k=200):
    """Scoring tokenises prompt and target separately. Verify that this equals
    tokenising the joined string, otherwise positions would be misaligned."""
    bad = 0
    for p, t in list(zip(prompts, targets))[:k]:
        joint = tok(p + t, add_special_tokens=False).input_ids
        split = tok(p, add_special_tokens=False).input_ids + tok(t, add_special_tokens=False).input_ids
        bad += joint != split
    return bad / min(k, len(prompts))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=None, help="subsample size (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    t0 = time.time()

    raw = load_dataset("akariasai/PopQA", split="test").to_pandas()
    df, demos = prepare_popqa(raw, n=args.n, seed=args.seed)
    print(f"[data] {len(df)} facts, {df['prop'].nunique()} relations "
          f"({df.attrs.get('n_dropped_missing', 0)} rows dropped for missing labels)")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.model}: {n_params/1e6:.0f}M params on {device}")

    ntok = lambda s: len(tok(target(s), add_special_tokens=False).input_ids)
    df["n_obj_tok"] = df["obj"].map(ntok)
    df["n_subj_tok"] = df["subj"].map(ntok)
    df["n_obj_words"] = df["obj"].map(lambda s: len(s.split()))
    df["n_obj_chars"] = df["obj"].map(lambda s: len(s.strip()))

    prompts = [fact_prompt(r, s, demos) for r, s in zip(df["prop"], df["subj"])]
    targets = [target(o) for o in df["obj"]]
    ctrl_prompts = [copy_prompt(r, s, o, demos)
                    for r, s, o in zip(df["prop"], df["subj"], df["obj"])]

    mis = check_tokenisation(tok, prompts, targets)
    print(f"[check] tokenisation mismatch rate (should be ~0): {mis:.3f}")
    print("[check] example prompt:\n" + prompts[0] + "  <" + df['obj'].iloc[0] + ">\n")

    df["gen"] = generate(model, tok, prompts, device, args.batch_size, args.max_new_tokens)
    df["gen_correct"] = [is_match(g, a) for g, a in zip(df["gen"], df["answers"])]
    print(f"[gen] accuracy {df['gen_correct'].mean():.3f}  ({time.time()-t0:.0f}s)")

    tf = pd.DataFrame(teacher_forced(model, tok, prompts, targets, device, args.batch_size))
    ctrl = pd.DataFrame(teacher_forced(model, tok, ctrl_prompts, targets, device, args.batch_size))
    ctrl.columns = [c.replace("tf_", "copy_") for c in ctrl.columns]
    df = pd.concat([df, tf, ctrl], axis=1)
    print(f"[tf] exact {df['tf_all'].mean():.3f} | [in-context control] exact {df['copy_all'].mean():.3f}")

    df["model"] = args.model
    df["n_params"] = n_params
    df["answers"] = df["answers"].map(json.dumps)
    keep = ["model", "n_params", "id", "prop", "subj", "obj", "answers", "s_pop", "o_pop",
            "n_obj_tok", "n_subj_tok", "n_obj_words", "n_obj_chars", "gen", "gen_correct"]
    keep = [c for c in keep if c in df.columns]
    keep += [c for c in df.columns if c.startswith(("tf_", "copy_"))]

    os.makedirs(args.out_dir, exist_ok=True)
    short = args.model.split("/")[-1]
    df[keep].to_csv(os.path.join(args.out_dir, f"{short}.csv"), index=False)
    with open(os.path.join(args.out_dir, "demos.json"), "w") as f:
        json.dump(demos, f, indent=1)
    print(f"[done] {short}: {time.time()-t0:.0f}s total")


if __name__ == "__main__":
    main()
