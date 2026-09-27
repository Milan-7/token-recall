# It's Not the Length, It's the Split: Subword Fragmentation and Factual Recall in Language Models

Code, data and results for the term paper of the same title (Milan Mahesh, *Advanced Topics in Computational Text and Media Sciences*, Universität Trier, Summer Term 2026).

**Question.** How much does the subword segmentation of an answer affect whether a language model recalls a fact, once entity popularity and relation type are controlled for, and where in the answer does recall fail?

**Main findings.**
- Each additional *within-word* subword split lowers the odds of correct recall (OR 0.65–0.87 per split across Pythia 160M–2.8B); additional *words* do not consistently hurt.
- An in-context control with identical prompts and targets shows no such penalty, so the effect is specific to recall from parameters.
- 96–98% of failures on multi-token answers occur at the first token.
- A linear popularity control and overlap between gold answers and few-shot demonstrations both inflate the apparent advantage of short answers.

## Setup

| Component | Choice |
|---|---|
| Data | PopQA (Mallen et al., 2023): 16 relations, subject/object popularity (monthly Wikipedia page views) and answer aliases; 14,216 evaluation facts |
| Models | Pythia 160M / 410M / 1.4B / 2.8B, deduped (Biderman et al., 2023): one shared tokenizer, so scale varies while tokenization is fixed |
| Prompting | 3-shot completion prompts per relation; demonstrations drawn once (seed 0) and excluded from evaluation (`results/demos.json`) |
| Metrics | (a) greedy generation, correct if any gold alias appears as a whole-word span; (b) teacher-forced exact match of the canonical answer; (c) first-token accuracy and rank |
| Negative control | In-context condition: same prompt and target, but the fact is stated once at the top of the prompt, so the model only has to copy it |
| Analysis | Logistic regression with relation fixed effects, subject length, subject- and object-popularity deciles and a demonstration-overlap indicator. Answer length is decomposed into words and extra within-word splits (tokens − words). Recall and control are compared on the same exact-match metric with fact-clustered standard errors. Robustness: linear/quartile popularity, excluding answers contained in the subject name or equal to a demonstration answer, 1–4-token answers only, per relation group and per relation |

## Run

On Google Colab (T4 GPU): open `run_colab.ipynb` and run the cells from top to bottom.

Locally with a GPU:

```bash
pip install -r requirements.txt
cd src
for m in pythia-160m-deduped pythia-410m-deduped pythia-1.4b-deduped pythia-2.8b-deduped; do
  python probe.py --model EleutherAI/$m --out_dir ../results
done
python analyze.py --results ../results --out ..
```

The analysis step needs no GPU and reproduces all tables and figures from the CSVs in `results/`.

## Outputs

- `results/<model>.csv`: one row per fact with generation, correctness, token counts, teacher-forced and in-context-control scores (`copy_*` columns)
- `results/summary.json`: all regression estimates and headline numbers
- `results/demos.json`: the few-shot demonstrations
- `tables/*.tex`: LaTeX tables used in the paper (require `booktabs`)
- `figures/*.pdf|png`: figures used in the paper

## Files

```
src/templates.py   relation templates
src/common.py      data preparation, prompts, answer matching (no torch dependency)
src/probe.py       model scoring: generation, teacher forcing, in-context control
src/analyze.py     regressions, tables, figures
run_colab.ipynb    end-to-end Colab notebook
requirements.txt   Python dependencies
```
