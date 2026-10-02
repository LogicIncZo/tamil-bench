# Contributing to Tamil Bench

Tamil Bench is an open, API-only evaluation of language models on Tamil. Every
score it publishes is backed by a raw answer sheet in `results/`, so a reviewer
can recompute any number on the site without trusting this project.

There are four ways to help, and none of them require you to spend money:

| What | Cost to you | Start at |
| --- | --- | --- |
| Report a wrong or missing score | free | [Open an issue](https://github.com/LogicIncZo/tamil-bench/issues/new) |
| Add a model to the board | free if `:free`, metered otherwise | [Run request issue](https://github.com/LogicIncZo/tamil-bench/issues/new) |
| Add a new test / task | free | [Propose a task](https://github.com/LogicIncZo/tamil-bench/issues/new) |
| Improve the runner or the site | free | [Fork and open a PR](https://github.com/LogicIncZo/tamil-bench/fork) |

Money is a fifth option and the least useful one to the project. See
[Contribute financially](#contribute-financially) at the bottom; the page at
<https://logicinczo.github.io/tamil-bench/contribute.html> covers the same ground
with the live cost ledger.

## Ground rules

These are the invariants that keep the board comparable. Breaking one of them
invalidates scores rather than just making a diff ugly.

1. **Result sheets are append-only evidence.** Never hand-edit or delete a row in
   `results/*.jsonl` to change a score. Re-run the sheet or fix the analyser.
2. **Sheet names are load-bearing.** `build_site.py` parses each stem as
   `{task}_{org}_{model-slug}_n{N}.jsonl`, where `/` in a model id becomes `_`.
   A sheet that does not match that shape is silently ignored by the site build.
3. **Sampling is seed 42** (`SEED` in `bench.py`, `df.sample(frac=1,
   random_state=SEED)`). Cross-model comparability depends on every model
   seeing the same items in the same order. Never sample differently.
4. **API errors are visible, never disguised as zero.** `bench.py` scores over
   all rows; `build_site.py` recomputes over valid rows only and reports
   `n_errors` beside the score. Keep both behaviours if you touch scoring math.
5. **Small sheets stay off the leaderboard.** `FULL_MIN` in `build_site.py`
   (`milu` 150, `indicqa` 90, `xnli` 150) gates what counts. Smoke tests with
   `--n 5` are for your own confidence; they will not appear on the site.
6. **Every new model id must be registered** in `MODELS` in `build_site.py` or
   the build raises `KeyError`. The chart labels use a separate `SHORT` map.
7. **Datasets are not committed.** `data/*.json` and `data/*.parquet` are
   gitignored. Add a loader, not a copy of the data. MILU is gated on
   Hugging Face — go through the gate.
8. **The charts are English-only on purpose.** matplotlib cannot shape Tamil
   script. Tamil belongs in the HTML, where the browser shapes it correctly.

## Setting up

Python 3.12. Then:

```bash
pip install pandas requests matplotlib pyarrow

export OPENROUTER_API_KEY=...   # required by bench.py
export HF_TOKEN=...             # only for the gated MILU download
```

Datasets go in `data/` and are not in git — see `data/README.md` for how to
obtain each one.

## Adding a model to the board

Smoke-test first, then run the full suite. One full suite is 499 requests.

```bash
python3 bench.py indicqa --model vendor/model-name --n 5    # cheap sanity check
python3 bench.py milu    --model vendor/model-name --n 199
python3 bench.py indicqa --model vendor/model-name --n 100
python3 bench.py xnli    --model vendor/model-name --n 200
```

Then register the display name in `MODELS` in `build_site.py`, and preview:

```bash
python3 build_site.py --no-push
```

`build_site.py` **pushes by default** — running it bare commits to `main` and
publishes to GitHub Pages. That is a publish action. Use `--no-push` unless you
intend to release.

The `run_*.sh` and `*_sweep.sh` scripts are cohort runners. They are resumable
and skip models that already have sheets, so re-invoking one after a quota reset
is safe.

## Adding a new test

A new task is a real piece of work, so please open an issue before writing
much code. A proposal should answer:

- **What does it measure** that MILU, IndicQA and IndicXNLI do not already?
- **Where does the dataset come from**, and is the licence compatible with
  redistributing derived sheets in this repo?
- **Is the metric already implemented** in `bench.py` (EM, F1, accuracy, a
  multiple-choice parse), or does it need a new one? Explain the metric; prefer
  something with a standard definition over a bespoke score.
- **How many items**, and will it clear `FULL_MIN`? Raise `FULL_MIN`'s sibling
  in the same PR if the task is small.
- **Is it 0-shot or few-shot?** Few-shot needs `data/milu_ta_dev.parquet`
  locally, which is not available through the parquet API.

The mechanics of wiring one in:

1. Add a `load_*` and `run_*` pair in `bench.py`, and register the subcommand in
   `main()`'s loop over task names.
2. Write sheets as `{task}_{org}_{model-slug}_n{N}.jsonl`, one JSON object per
   answer, including the model's raw output, the gold label, and the OpenRouter
   `usage` block. The sheets are the evidence — a score nobody can recompute is
   not a result.
3. Add the task to `FULL_MIN` and to `compute()` in `build_site.py`, and add its
   score columns to the site.
4. Add a per-test page by adding an entry to the `pages` tuple in
   `generate_test_pages`. The sort, date and cost columns come for free.

Run `python3 build_site.py --no-push` and check that `results/summary.json`
gains the task and that `python3 bluff.py` still runs. Say what you tested in
the PR.

## What a good PR looks like

- One concern per PR. A new metric and a site redesign are two PRs.
- No dataset files, no result-sheet edits, no regenerated chart committed by
  hand — `build_site.py` owns the generated artefacts.
- The PR body states what changed, what you ran, and what the output was. If you
  touched scoring, paste the before/after numbers for a model that was already
  on the board; a scoring change that moves existing scores needs to be visible
  in the diff.
- Bilingual user-facing strings need both the English and the Tamil `data-lang`
  variant, added together in the same commit.

## Reporting a problem

A score report is most useful with the sheet name and the row count. You can
verify any published score yourself:

```bash
python3 build_site.py --no-push   # recomputes from results/ and prints per-model counts
```

If the site's number disagrees with what you compute, open an issue with the
sheet name and what you got.

## Conduct

Assume good faith. Criticise the measurement, not the person being measured —
a model's score is a fact about the model, and the point of the project is to
make that fact checkable.

## Licence

MIT. Contributions are accepted under the same terms.
