# Tokenization as a hidden variable in factual recall

Code for the term paper  Advanced Topics in Computational Text and Media Sciences.

**Question.** How much of a language model's factual-recall accuracy is explained by how many subword tokens the answer is split into, once entity popularity is controlled for?

## Setup

| Component | Choice |
|---|---|
| Data | PopQA (Mallen et al., 2023), 16 relations, subject/object popularity (Wikipedia page views) and answer aliases |
| Models | Pythia 160M / 410M / 1.4B / 2.8B, deduped (Biderman et al., 2023) — one shared tokenizer, so scale varies while tokenization is fixed |
| Prompting | 3-shot cloze-style completion per relation; demonstrations are held out from evaluation (`results/demos.json`) |
| Main metric | Greedy generation; correct if any gold alias appears as a whole-word span (normalised) |
| Teacher-forced metrics | Per-token top-1 accuracy of the canonical object given the gold prefix; first-token rank; log-probability |
| Negative control | In-context (open-book) control: the same prompt and target, but the full fact is stated once at the top of the prompt, so the model only copies it. Length should matter little if the pipeline is unbiased |
| Analysis | Logistic regression with relation fixed effects, controlling for subject length and subject- and object-popularity deciles (token counts capped at 6). Object length is also decomposed into words and extra within-word subword splits. Recall and control are compared on the same exact-match metric (fact-clustered SEs). Robustness: linear/quartile popularity, excluding answers contained in the subject name, 1–4-token objects only |

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

- `results/<model>.csv` — one row per fact: generation, correctness, token counts, teacher-forced and in-context-control scores (`copy_*` columns)
- `results/summary.json` — all regression coefficients and headline numbers
- `tables/*.tex` — LaTeX tables (use `booktabs`)
- `figures/*.pdf` — figures for the paper

## Files

```
src/templates.py   relation templates
src/common.py      data preparation, prompts, answer matching (no torch dependency)
src/probe.py       model scoring: generation, teacher forcing, in-context control
src/analyze.py     regressions, tables, figures
run_colab.ipynb    end-to-end Colab notebook
```
