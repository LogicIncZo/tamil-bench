# Datasets — obtain before running

Raw datasets are **not** committed to this repo (keeps it light; MILU is gated on
Hugging Face and redistributing it would circumvent the gate).

## MILU (Paper 1 — MCQ)

- Source: https://huggingface.co/datasets/ai4bharat/MILU (config: `tamil`)
- License/gate: gated — accept the gate with your HF account, then either
  `huggingface-cli download ai4bharat/MILU --repo-type dataset` or use the
  programmatic gate-accept (see bench.py) with `HF_TOKEN` set.
- Expected file here: `milu_ta_test.parquet` (test split, Tamil config).

## IndicQA (Paper 2 — reading comprehension)

- Source: https://huggingface.co/datasets/ai4bharat/IndicQA (Tamil subset)
- License: CC-BY-SA-4.0 (derived from English Wikipedia passages, translated)
- Download: `huggingface-cli download ai4bharat/IndicQA --repo-type dataset`
  or fetch `indicqa.ta.json` (SQuAD-style JSON, `data` key, 253 articles).
- Expected file here: `indicqa.ta.json`.

Results in `results/` are self-contained evidence — model outputs + gold labels
— so every published score can be re-verified without the raw datasets.
