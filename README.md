# tamil-bench — Runnable Tamil LLM bench for API-only models

Zero-setup generative eval runner for OpenAI-compatible chat endpoints (OpenRouter default). Built because lm-eval MCQ tasks need loglikelihood (chat APIs don't return it) and SEA-HELM's harness needs vLLM/GPU.

**Live scoreboard (ELI5 + scores):** https://logicinczo.github.io/tamil-bench/ — served from `index.html` via GitHub Pages.

## Benchmarks

| Task | Dataset | Metric | Status |
| --- | --- | --- | --- |
| `milu` | ai4bharat/MILU, Tamil config (6,372 MCQs, India state-exam material) | accuracy (0-shot, letter parse; dev split not on parquet API) | open — gate accepted programmatically via `POST huggingface.co/datasets/ai4bharat/MILU/ask-access` (Bearer HF_TOKEN) |
| `indicqa` | ai4bharat/IndicQA Tamil (SQuAD-style, 1,804 questions) | EM + F1 (official SQuAD normalization) | open |

## Usage

```bash
cd /home/workspace/ThamizhKanimai/nlp/tamil-bench
python3 bench.py indicqa --model z-ai/glm-5.3-flash --n 100
python3 bench.py milu --model deepseek/deepseek-v4.1-flash --n 200   # after gate accept
```

- `--n` stratified sample size (seed 42, reproducible); omit for full dataset.
- `--workers` concurrency (default 8). `--endpoint` to override (any OpenAI-compatible URL + env key).
- Uses `OPENROUTER_API_KEY` from env. Results: JSONL per question + summary in `results/`.

## Baseline (2026-09-11, OpenRouter, 0-shot, n≈200 stratified by domain)

| Model | MILU-Tamil acc | 95% CI | IndicQA-Tamil EM (n=100) | F1 |
| --- | --- | --- | --- | --- |
| google/gemini-3.8-flash | 0.945 | 0.904–0.969 | 0.200 | 0.424 |
| openai/gpt-5.6-luna | 0.864 | 0.810–0.905 | 0.170 | 0.407 |
| deepseek/deepseek-v4.1-flash | 0.809 | 0.749–0.858 | 0.180 | 0.389 |
| z-ai/glm-5.3-flash | 0.799 | 0.738–0.849 | 0.190 | 0.368 |
| qwen/qwen3.8-flash | 0.688 | 0.621–0.749 | 0.220 | 0.381 |

Gemini 3.8 Flash breaks the earlier flash-tier tie decisively (+8pp over Luna on MILU, non-overlapping CIs vs everything else); its IndicQA F1 (0.424) is also best. GLM/DeepSeek remain a statistical tie. Qwen 3.8 flash is weakest on exam MCQ but posts the best IndicQA EM (0.220). All five beat GPT-4o's ~74% cross-language MILU average (2024). Qwen flagship tier exists (`qwen/qwen3.8-max-0902`, 13× price) — untested.

## Known gaps

- No Tamil in Global-MMLU (repo verified 2026-09-11 — no `ta` config).
- OpenAI IndQA: dataset never released (github.com/openai/indqa → 404); headroom signal only (GPT-5 best = 34.9%).
- SEA-HELM: cite the leaderboard (SEA-LION v4 tops Tamil at 68.47); don't run its harness on API models.

## Refresh pipeline

`python3 build_site.py [--no-push]` is the single repeatable refresh command:
it rescans `results/*.jsonl`, recomputes scores (identical math to `bench.py` —
errors count as wrong and stay in the denominator), rewrites
`results/summary.json`, regenerates the comparison charts in `assets/`, and
patches the scoreboard tables in `index.html` between the
`<!--ROWS:TASK:TIER-->` markers, then commits/pushes unless `--no-push`.

## Data sources & attribution

- **MILU** — AI4Bharat + IBM Research India, "MILU: A Multi-task Indic Language
  Understanding Benchmark" (arXiv:2411.02538), CC-BY-4.0, gated on HF
  (github.com/AI4Bharat/MILU). Tamil split: 6,372 MCQs from UPSC/state-PSC
  exams, 41 subjects, 8 domains (1,524 machine-translated, rest curated).
  This bench samples 199 questions stratified by subject (seed 42).
- **IndicQA** — AI4Bharat, "Towards Leaving No Indic Language Behind" (ACL 2023),
  CC-BY-SA-4.0 (github.com/AI4Bharat/IndicQA). Tamil: 1,804 questions
  (1,276 answerable) over 253 Wikipedia-derived articles; we use the
  answerable subset sample of 100 (seed 42), SQuAD-style EM/F1 scoring.
- Charts are CC-BY-SA too — attribute "tamil-bench" and share freely.
