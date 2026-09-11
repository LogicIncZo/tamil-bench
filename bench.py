#!/usr/bin/env python3
"""Tamil LLM bench: API-model generative evaluation on OpenRouter-compatible endpoints.

Benchmarks:
  indicqa - ai4bharat/IndicQA Tamil (extractive QA, EM/F1) - open, no gate
  milu    - ai4bharat/MILU Tamil (exam MCQ, accuracy) - gated on HF; download
            data/milu_ta_test.parquet (and _dev) after accepting the gate.
  xnli    - IndicXNLI Tamil (3-way NLI, accuracy) - data/indicxnli_ta_test.parquet
            from AdaMLLab/indicxnli_repaired (mirror of ai4bharat/IndicXNLI test).

Usage:
  python3 bench.py indicqa --model z-ai/glm-5.3-flash --n 100
  python3 bench.py milu --model deepseek/deepseek-v4.1-flash --n 200 --shots 5
  python3 bench.py xnli --model google/gemini-3.8-flash --n 200
"""

import argparse
import collections
import json
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

BASE = Path(__file__).parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
SEED = 42

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def load_key():
    import os
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set")
    return key


def chat(model, messages, key, max_tokens=512, max_retries=6):
    for attempt in range(max_retries):
        try:
            r = requests.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
                timeout=180,
            )
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            text = msg.get("content") or msg.get("reasoning") or ""
            return text.strip()
        except (requests.RequestException, KeyError, IndexError) as e:
            if attempt == max_retries - 1:
                return f"__ERROR__: {e}"
            time.sleep(2 ** attempt * 2)
    return "__ERROR__: retries exhausted"


def parse_letter(text):
    if text.startswith("__ERROR__"):
        return None
    m = re.search(r"answer(?:\s+is|\s*[:\-])\s*\**([ABCD])\b", text, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([ABCD])\b", text)
    if m:
        return m.group(1).upper()
    return None


def norm_squad(s):
    s = unicodedata.normalize("NFC", s)
    s = "".join(
        ch for ch in s if not unicodedata.category(ch).startswith("P")
    )
    s = s.lower()
    return " ".join(s.split())


def em_f1(pred, gold):
    pred_n, gold_n = norm_squad(pred), norm_squad(gold)
    if pred_n == gold_n:
        return 1.0, 1.0
    pt, gt = pred_n.split(), gold_n.split()
    common = collections.Counter(pt) & collections.Counter(gt)
    num_same = sum(common.values())
    if num_same == 0 or not pt or not gt:
        return 0.0, 0.0
    precision = num_same / len(pt)
    recall = num_same / len(gt)
    f1 = 2 * precision * recall / (precision + recall)
    return 0.0, f1


def wilson(p, n, z=1.96):
    if n == 0:
        return 0, 0
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return max(0, centre - half), min(1, centre + half)


def load_indicqa():
    rows = []
    raw = json.load(open(DATA / "indicqa.ta.json"))["data"]
    for art in raw:
        for para in art["paragraphs"]:
            for qa in para["qas"]:
                golds = [a["text"] for a in qa["answers"]]
                if not golds:
                    continue
                rows.append(
                    {
                        "id": qa["id"],
                        "context": para["context"],
                        "question": qa["question"],
                        "golds": golds,
                    }
                )
    return rows


QA_SYS = "You are a helpful assistant that answers reading comprehension questions in Tamil."
QA_USER = (
    "Read the following passage and answer the question. "
    "Reply ONLY with a short phrase from the passage, in Tamil. No explanation.\n\n"
    "Passage: {context}\n\nQuestion: {question}\nAnswer:"
)


def run_indicqa(args, key):
    rows = load_indicqa()
    rows = sorted(rows, key=lambda r: str(r["id"]))
    rng = __import__("random").Random(SEED)
    rng.shuffle(rows)
    subset = rows[: args.n]
    print(f"IndicQA-Tamil: {len(subset)} questions, model={args.model}")

    def work(item):
        text = chat(
            args.model,
            [
                {"role": "system", "content": QA_SYS},
                {
                    "role": "user",
                    "content": QA_USER.format(
                        context=item["context"], question=item["question"]
                    ),
                },
            ],
            key,
        )
        em, f1 = em_f1(text, item["golds"][0])
        best_f1 = max(f1, *(em_f1(text, g)[1] for g in item["golds"][1:])) if len(item["golds"]) > 1 else f1
        return {**item, "prediction": text, "em": em, "f1": best_f1}

    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, r): r["id"] for r in subset}
        for i, fut in enumerate(as_completed(futs), 1):
            out.append(fut.result())
            if i % 20 == 0:
                print(f"  {i}/{len(subset)}")

    slug = args.model.replace("/", "_")
    path = RESULTS / f"indicqa_{slug}_n{len(subset)}.jsonl"
    out.sort(key=lambda r: str(r["id"]))
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    em = sum(r["em"] for r in out) / len(out)
    f1 = sum(r["f1"] for r in out) / len(out)
    errs = sum(1 for r in out if r["prediction"].startswith("__ERROR__"))
    lo, hi = wilson(em, len(out))
    print(f"\nIndicQA-Tamil  model={args.model}  n={len(out)}  errors={errs}")
    print(f"EM: {em:.3f} (95% CI {lo:.3f}-{hi:.3f})   F1: {f1:.3f}")
    print(f"saved: {path}")

XNLI_DATA = DATA / "indicxnli_ta_test.parquet"
XNLI_LETTERS = ("A", "B", "C")
XNLI_SYS = "You are a careful reasoner. Answer with the single letter A, B, or C."
XNLI_USER = (
    "பின்வரும் வாக்கியத்தைப் படிக்கவும்:\n\nமுன்னுரை: {premise}\n\n"
    "கூற்று: {hypothesis}\n\n"
    "முன்னுரையைப் பொறுத்து, கூற்று எந்த நிலையில் உள்ளது?\n"
    "A) கண்டிப்பாக உண்மை (முன்னுரையால் உறுதிப்படுத்தப்படுகிறது)\n"
    "B) கண்டிப்பாக தவறு (முன்னுரையுடன் முரண்படுகிறது)\n"
    "C) முடிவு செய்ய முடியாது (முன்னுரையில் தெளிவான தகவல் இல்லை)\n\n"
    "Reply with ONLY the single letter A, B, or C."
)


def load_xnli():
    df = pd.read_parquet(XNLI_DATA)
    df = df.sample(frac=1, random_state=SEED)
    return df.head(200)


def run_xnli(args, key):
    if not XNLI_DATA.exists():
        sys.exit(
            f"{XNLI_DATA} missing. Download the Tamil test split from "
            "https://huggingface.co/datasets/AdaMLLab/indicxnli_repaired "
            "(data/ta/test-00000-of-00001.parquet) and save it at that path."
        )
    df = load_xnli()
    subset = df.head(args.n)
    print(f"IndicXNLI-Tamil: {len(subset)} items, model={args.model}")
    label_map = {0: "A", 2: "B", 1: "C"}

    def work(rec):
        i, row = rec
        for attempt in range(3):
            try:
                txt = chat(
                    args.model,
                    [
                        {"role": "system", "content": XNLI_SYS},
                        {"role": "user", "content": XNLI_USER.format(
                            premise=row["premise"], hypothesis=row["hypothesis"])},
                    ],
                    key,
                    max_tokens=1024,
                )
                tail = txt[-300:]
                m = (re.search(r"answer(?:\s+is|\s*[:\-])\s*\**([ABC])\b", tail, re.I)
                     or re.search(r"\b([ABC])\b(?!.*\b[ABC]\b)", tail, re.S)
                     or re.search(r"\b([ABC])\b", txt, re.I))
                pred = m.group(1).upper() if m else ""
                gold = label_map[int(row["label"])]
                return {
                    "id": int(i),
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "gold": gold,
                    "prediction": pred,
                    "raw": txt,
                    "correct": pred == gold,
                }
            except Exception as exc:
                if attempt == 2:
                    return {"id": int(i), "prediction": f"__ERROR__ {exc}"}
                time.sleep(2 ** (attempt + 1))

    out = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, rec) for rec in subset.iterrows()]
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            out[r["id"]] = r
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(subset)}")
    rows = [out[i] for i in sorted(out)]
    slug = args.model.replace("/", "_")
    path = RESULTS / f"xnli_{slug}_n{len(rows)}.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    k = sum(1 for r in rows if r.get("correct"))
    p = k / len(rows) if rows else 0
    print(f"Accuracy: {p:.3f}")
    print(f"Wrote {path}")


MILU_URLS = {
    "test": "https://huggingface.co/api/datasets/ai4bharat/MILU/parquet/Tamil/test/0.parquet",
    "dev": "https://huggingface.co/api/datasets/ai4bharat/MILU/parquet/Tamil/dev/0.parquet",
}


def milu_download(token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for split, url in MILU_URLS.items():
        dest = DATA / f"milu_ta_{split}.parquet"
        if dest.exists() and dest.stat().st_size > 1000:
            continue
        r = requests.get(url, headers=headers, timeout=60)
        if r.status_code == 200 and r.content[:4] == b"PAR1":
            dest.write_bytes(r.content)
            print(f"downloaded {dest.name}")
        else:
            print(f"{split}: not available ({r.status_code}). Gate: https://huggingface.co/datasets/ai4bharat/MILU")


def milu_fields(df):
    cols = {c.lower().strip(): c for c in df.columns}
    opts = {}
    for key, pat in [
        ("A", "option_a"), ("B", "option_b"), ("C", "option_c"), ("D", "option_d"),
        ("A", "option1"), ("B", "option2"), ("C", "option3"), ("D", "option4"),
    ]:
        if pat in cols:
            opts[key] = cols[pat]
        elif pat.upper() in cols:
            opts[key] = cols[pat.upper()]
    if not opts and {"a", "b", "c", "d"} <= set(cols):
        opts = {k: cols[k] for k in "ABCD"}
    ans = next((cols[c] for c in ("answer", "answer_key", "answer_index", "label", "target") if c in cols), None)
    subj = next((cols[c] for c in ("subject", "domain", "category") if c in cols), None)
    q = next((cols[c] for c in ("question", "prompt") if c in cols), None)
    return opts, ans, subj, q


def run_milu(args, key):
    milu_download(load_hf_token())
    test_path = DATA / "milu_ta_test.parquet"
    if not test_path.exists():
        sys.exit("MILU Tamil not downloaded. Accept the gate at https://huggingface.co/datasets/ai4bharat/MILU with your HF account, then retry.")
    df = pd.read_parquet(test_path)
    opts, ansc, subj, q = milu_fields(df)
    if not opts or not ansc or not q:
        sys.exit(f"Unrecognised MILU columns: {list(df.columns)}")

    letters = list(opts.keys())
    def answer_of(row):
        v = row[ansc]
        if isinstance(v, str):
            t = v.strip()
            if t.upper() in letters:
                return t.upper()
            m = re.fullmatch(r"option\s*([1-4])", t.lower())
            if m:
                return letters[int(m.group(1)) - 1]
        if isinstance(v, (int, float)):
            return letters[int(v)]
        return None

    shots = []
    dev_path = DATA / "milu_ta_dev.parquet"
    if args.shots > 0 and dev_path.exists():
        ddf = pd.read_parquet(dev_path)
        dopts, dansc, dsubj, dq = milu_fields(ddf)
        for _, row in ddf.head(args.shots).iterrows():
            v = row[dansc]
            key_letter = v.strip().upper() if isinstance(v, str) else letters[int(v)]
            block = f"{row[dq]}\n" + "\n".join(f"{k}. {row[dopts[k]]}" for k in letters) + f"\nAnswer: {key_letter}"
            shots.append(block)
    shot_block = "\n\n".join(shots) + "\n\n" if shots else ""

    df = df.sample(frac=1, random_state=SEED)
    if subj and args.n < len(df):
        subset = (
            df.groupby(subj, group_keys=False)
            .apply(lambda g: g.head(max(1, round(args.n * len(g) / len(df)))))
            .head(args.n)
        )
    else:
        subset = df.head(args.n)
    print(f"MILU-Tamil: {len(subset)} questions, model={args.model}, shots={len(shots)}")

    def work(item):
        block = f"{item[q]}\n" + "\n".join(f"{k}. {item[opts[k]]}" for k in letters)
        text = chat(
            args.model,
            [
                {
                    "role": "user",
                    "content": (
                        "The following are multiple choice questions (with answers) about an Indian exam. "
                        "Reply ONLY with the letter of the correct option.\n\n"
                        f"{shot_block}{block}\nAnswer:"
                    ),
                }
            ],
            key,
        )
        pred = parse_letter(text)
        gold = answer_of(item)
        return {
            "subject": str(item[subj]) if subj else "",
            "gold": gold,
            "prediction": text,
            "pred_letter": pred,
            "correct": int(pred == gold) if pred and gold else 0,
        }

    out = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, row): i for i, row in subset.iterrows()}
        for i, fut in enumerate(as_completed(futs), 1):
            out.append(fut.result())
            if i % 25 == 0:
                print(f"  {i}/{len(subset)}")

    slug = args.model.replace("/", "_")
    path = RESULTS / f"milu_{slug}_n{len(out)}.jsonl"
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    acc = sum(r["correct"] for r in out) / len(out)
    lo, hi = wilson(acc, len(out))
    print(f"\nMILU-Tamil  model={args.model}  n={len(out)}")
    print(f"Accuracy: {acc:.3f} (95% CI {lo:.3f}-{hi:.3f})")
    if subj:
        by = collections.defaultdict(lambda: [0, 0])
        for r in out:
            by[r["subject"]][1] += 1
            by[r["subject"]][0] += r["correct"]
        print("By subject:")
        for s, (c, n) in sorted(by.items()):
            print(f"  {s}: {c}/{n} = {c/n:.3f}")
    print(f"saved: {path}")


def load_hf_token():
    import os
    return os.environ.get("HF_TOKEN")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="bench", required=True)
    for name in ("indicqa", "milu", "xnli"):
        sp = sub.add_parser(name)
        sp.add_argument("--model", required=True)
        sp.add_argument("--n", type=int, default=100)
        sp.add_argument("--workers", type=int, default=8)
        if name == "milu":
            sp.add_argument("--shots", type=int, default=5)
    args = p.parse_args()
    key = load_key()
    if args.bench == "indicqa":
        run_indicqa(args, key)
    elif args.bench == "xnli":
        run_xnli(args, key)
    else:
        run_milu(args, key)


if __name__ == "__main__":
    main()
