#!/usr/bin/env python3
"""Bluff catch: score how often models admit "the passage doesn't say" on
IndicQA-Tamil's unanswerable (category=NO) questions, using existing
results/indicqa_*.jsonl sheets. Zero API cost.

Method: join each sheet row to the dataset category by id. For NO rows,
classify the model's prediction:
  - abstain: matches a Tamil/English refusal lexicon (or empty non-error text)
  - bluff:   any other non-empty prediction (model invented an answer)
Rows whose prediction starts with __ERROR__ are excluded from the denominator.
Also computes the over-refusal rate on answerable (YES) questions.

Usage: python3 bluff.py            # print table
       python3 bluff.py --json     # emit results/bluff-summary.json
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"

ABSTAIN_PATTERNS = [
    "தெரியாது", "தெரியவில்லை", "தெரியல", "விடையில்லை", "விடை இல்லை",
    "இல்லை", "கிடைக்கவில்லை", "குறிப்பிடப்படவில்லை", "கூறப்படவில்லை",
    "தகவல் இல்லை", "கொடுக்கப்படவில்லை", "இயலாது", "முடியாது", "காணவில்லை",
    "வழங்கப்படவில்லை", "எழுதப்படவில்லை", "பதில் இல்லை", "பதிலில்லை",
    "no answer", "not mentioned", "not provided", "not given", "not stated",
    "not specified", "not found", "not available", "no information",
    "unknown", "cannot", "can't", "cannot determine", "unable", "n/a",
]
ABSTAIN_RE = re.compile("|".join(re.escape(p) for p in ABSTAIN_PATTERNS), re.I)


def norm(s):
    return unicodedata.normalize("NFC", str(s or "")).strip()


def categories():
    cat = {}
    raw = json.load(open(DATA / "indicqa.ta.json"))["data"]
    for art in raw:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                cat[str(qa["id"])] = qa.get("category", "YES")
    return cat


def score_sheet(path, cat):
    rows = [json.loads(l) for l in path.open() if l.strip()]
    no_tot = no_abstain = no_bluff = no_err = 0
    yes_tot = yes_abstain = 0
    yes_em_sum = yes_f1_sum = 0.0
    for r in rows:
        pred = norm(r.get("prediction"))
        c = cat.get(str(r.get("id")), "YES")
        if pred.startswith("__ERROR__"):
            if c == "NO":
                no_err += 1
            continue
        if c == "NO":
            no_tot += 1
            if not pred or ABSTAIN_RE.search(pred):
                no_abstain += 1
            else:
                no_bluff += 1
        else:
            yes_tot += 1
            yes_em_sum += float(r.get("em") or 0)
            yes_f1_sum += float(r.get("f1") or 0)
            if pred and ABSTAIN_RE.search(pred):
                yes_abstain += 1
    return {
        "n_unanswerable": no_tot,
        "n_errors_no": no_err,
        "abstain": no_abstain,
        "bluff": no_bluff,
        "bluff_rate": round(100 * no_bluff / no_tot, 1) if no_tot else None,
        "n_answerable": yes_tot,
        "over_refusal_rate": round(100 * yes_abstain / yes_tot, 1) if yes_tot else None,
        "em_answerable": round(100 * yes_em_sum / yes_tot, 1) if yes_tot else None,
        "f1_answerable": round(yes_f1_sum / yes_tot, 1) if yes_tot else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cat = categories()
    out = {}
    for f in sorted(RESULTS.glob("indicqa_*.jsonl")):
        if f.name.endswith("n5.jsonl"):
            continue
        model = f.stem[len("indicqa_"):].rsplit("_n", 1)[0]
        out[model] = score_sheet(f, cat)

    if not out:
        sys.exit("no indicqa sheets found")
    print(f"{'model':44s} {'bluff%':>7s} {'abstain':>8s} {'n_unans':>8s} {'EM_ans':>7s} {'F1_ans':>7s} {'over-ref%':>10s}")
    for m, s in sorted(out.items(), key=lambda kv: (kv[1]["bluff_rate"] or 0)):
        print(f"{m:44s} {s['bluff_rate']:>7} {s['abstain']:>8} {s['n_unanswerable']:>8} "
              f"{s['em_answerable'] or 0:>7} {s['f1_answerable'] or 0:>7} "
              f"{s['over_refusal_rate'] if s['over_refusal_rate'] is not None else '-':>10}")
    if args.json:
        (RESULTS / "bluff-summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {RESULTS/'bluff-summary.json'}")


if __name__ == "__main__":
    main()
