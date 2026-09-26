# Tokenization as a hidden variable in factual recall

Code for the term paper *[your title]* (Advanced Topics in Computational Text and Media Sciences, 2026).

**Question.** How much of a language model's factual-recall accuracy is explained by how many subword tokens the answer is split into, once entity popularity is controlled for?

## Setup

| Component | Choice |
|---|---|
| Data | PopQA (Mallen et al., 2023), 16 relations, subject/object popularity (Wikipedia page views) and answer aliases |
| Models | Pythia 160M / 410M / 1.4B / 2.8B, deduped (Biderman et al., 2023) — one shared tokenizer, so scale varies while tokenization is fixed |
| Prompting | 3-shot cloze-style completion per relation; demonstrations are held out from evaluation (`results/demos.json`) |
| Main metric | Greedy generation; correct if any gold alias appears as a whole-word span (normalised) |
| Teacher-forced metrics | Per-token top-1 accuracy of the canonical object given the gold prefix; first-token rank; log-probability |
| Negative control | Copy task with the *same* target strings: the answer is given verbatim in the prompt, so length should not matter if the pipeline is unbiased |
| Analysis | Logistic regression with relation fixed effects; object tokens, subject tokens, log10 subject and object popularity. Token counts capped at 6. Pooled model tests token × scale interaction with fact-clustered standard errors |

## Run

On Google Colab (T4 GPU): open `run_colab.ipynb` and run top to bottom.

Locally with a GPU:

```bash
pip install -r requirements.txt
cd src
for m in pythia-160m-deduped pythia-410m-deduped pythia-1.4b-deduped pythia-2.8b-deduped; do
  python probe.py --model EleutherAI/$m --out_dir ../results
done
python analyze.py --results ../results --out ..
```

## Outputs

- `results/<model>.csv` — one row per fact: generation, correctness, token counts, teacher-forced and copy-control scores
- `results/summary.json` — all regression coefficients and headline numbers
- `tables/*.tex` — LaTeX tables (use `booktabs`)
- `figures/*.pdf` — figures for the paper

## Files

```
src/templates.py   relation templates and copy-control demonstrations
src/common.py      data preparation, prompts, answer matching (no torch dependency)
src/probe.py       model scoring: generation, teacher forcing, copy control
src/analyze.py     regressions, tables, figures
run_colab.ipynb    end-to-end Colab notebook
```
