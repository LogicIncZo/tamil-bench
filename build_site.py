#!/usr/bin/env python3
"""tamil-bench site refresh: recompute scores, regenerate summary.json + charts, patch index.html.

Usage: python3 build_site.py [--no-push]
Idempotent — safe to re-run after each new results file lands.
"""
import json, math, re, subprocess, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FULL_MIN = {"milu": 150, "indicqa": 90, "xnli": 150}

MODELS = {
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
    "qwen/qwen3.8-flash": "Qwen 3.8 Flash",
    "xiaomi/mimo-v2.5": "Xiaomi MiMo v2.5",
    "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B A4B",
    "inclusionai/ling-3.0-flash-vl:free": "Ling 3.0 Flash VL",
    "nvidia/nemotron-3.5-lightning:free": "Nemotron 3.5 Lightning",
    "poolside/laguna-s-2.1:free": "Poolside Laguna-S 2.1",
}

def parse_stem(stem):
    for task in ("milu", "indicqa", "xnli"):
        if stem.startswith(task + "_"):
            rest = stem[len(task) + 1:]
            m = re.search(r"_n(\d+)$", rest)
            if not m:
                return None
            n = int(m.group(1))
            ident = rest[: m.start()]
            org, model = ident.split("_", 1)
            return task, f"{org}/{model}", n
    return None

def wilson(p, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)

def f1_score(pred, gold):
    pt, gt = pred.split(), gold.split()
    common = set(pt) & set(gt)
    if not common:
        return 0.0
    prec, rec = len(common) / len(pt), len(common) / len(gt)
    return 2 * prec * rec / (prec + rec)

def load():
    out = {"milu": defaultdict(list), "indicqa": defaultdict(list), "xnli": defaultdict(list)}
    for f in sorted(RESULTS.glob("*.jsonl")):
        p = parse_stem(f.stem)
        if not p:
            continue
        task, model, n = p
        rows = [json.loads(l) for l in f.open() if l.strip()]
        if len(rows) < FULL_MIN[task]:
            continue
        out[task][model].append((n, rows))
    return out

def valid_rows(rows):
    """Rows with an actual model answer -- drops API failures so they never
    masquerade as 0% accuracy."""
    out = []
    for r in rows:
        pred = (r.get("prediction") or "").strip()
        if not pred or pred.startswith("__ERROR__") or str(r.get("raw", "")).startswith("__ERROR__"):
            continue
        out.append(r)
    return out

def compute(data):
    scores = {"milu": {}, "indicqa": {}, "xnli": {}}
    for model, runs in data["milu"].items():
        n, rows = max(runs, key=lambda r: r[0])
        valid = valid_rows(rows)
        if not valid:
            continue
        m = len(valid)
        k = sum(1 for r in valid if r.get("correct"))
        lo, hi = wilson(k / m, m)
        scores["milu"][model] = {
            "n": m, "n_errors": len(rows) - m, "accuracy": round(100 * k / m, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)],
        }
    for model, runs in data["indicqa"].items():
        n, rows = max(runs, key=lambda r: r[0])
        valid = valid_rows(rows)
        if not valid:
            continue
        m = len(valid)
        em = sum(float(r["em"]) for r in valid) / m
        f1 = 100 * sum(float(r["f1"]) for r in valid) / m
        lo, hi = wilson(em, m)
        scores["indicqa"][model] = {
            "n": m, "n_errors": len(rows) - m, "em": round(100 * em, 1), "f1": round(f1, 1),
            "em_ci": [round(100 * lo, 1), round(100 * hi, 1)],
        }
    for model, runs in data.get("xnli", {}).items():
        n, rows = max(runs, key=lambda r: r[0])
        valid = valid_rows(rows)
        if not valid:
            continue
        m = len(valid)
        k = sum(1 for r in valid if r.get("correct"))
        lo, hi = wilson(k / m, m)
        scores["xnli"][model] = {
            "n": m, "n_errors": len(rows) - m, "accuracy": round(100 * k / m, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)],
        }
    return scores


ABSTAIN_RE = (
    "தெரியலை", "தெரியாது", "தெரியவில்லை", "விடையில்லை", "விடை இல்லை",
    "குறிப்பில் இல்லை", "குறிப்பிடப்படவில்லை", "சொல்லப்படவில்லை",
    "கிடைக்கவில்லை", "வழங்கப்படவில்லை", "அறியப்படவில்லை", "இல்லை",
    "no answer", "not mentioned", "not provided", "not specified",
    "not stated", "cannot", "can\u2019t", "can't", "unable", "unknown",
    "passage does not", "does not mention", "doesn't mention",
)


def is_abstention(pred):
    p = str(pred).strip().lower()
    if not p or p.startswith("__ERROR__"):
        return True
    return any(tok in p for tok in ABSTAIN_RE)


ABSTAIN_MARKERS = (
    "தெரியல", "தெரியாத", "தெரியவில்லை", "விடையில்லை", "விடை இல்லை", "இல்லை என",
    "கிடைக்கவில்லை", "குறிப்பிடவில்லை", "குறிப்பிடப்படவில்லை", "பதில் இல்லை",
    "சொல்லப்படவில்லை", "நிரூபிக்க முடியாது", "முடியாது", "சாத்தியமில்லை",
    "no answer", "not mentioned", "not specified", "not stated", "cannot be",
    "can't be", "unable to", "unknown", "not provided", "no information",
    "unanswerable", "not in the passage", "passage does not", "not given",
)

def bluff_scores(data):
    """Bluff catch: on unanswerable IndicQA traps (golds == ['']), did the model abstain?"""
    traps = {}
    for m, runs in data["indicqa"].items():
        n, rows = max(runs, key=lambda r: r[0])
        hit = [r for r in rows if [g.strip() for g in r.get("golds", [])] == [""]]
        if len(hit) < 10:
            continue
        bluff = 0
        for r in hit:
            pred = str(r.get("prediction", "")).strip().strip('"').lower()
            if not pred or pred.startswith("__ERROR__"):
                continue
            if not any(mk in pred for mk in ABSTAIN_MARKERS):
                bluff += 1
        traps[m] = {
            "n_traps": len(hit), "bluff_rate": round(100 * bluff / len(hit), 1),
            "abstain_rate": round(100 * (len(hit) - bluff) / len(hit), 1),
        }
    return traps

def xnli_scores(data):
    scores = {}
    for model, runs in data.get("xnli", {}).items():
        n, rows = max(runs, key=lambda r: r[0])
        valid = valid_rows(rows)
        if not valid:
            continue
        m = len(valid)
        k = sum(1 for r in valid if r.get("correct"))
        lo, hi = wilson(k / m, m)
        scores[model] = {
            "n": m, "accuracy": round(100 * k / m, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)],
        }
    return scores

def row_html(task, rank, model, s):
    name = MODELS[model]
    cls = ' class="top"' if rank == 1 else ""
    if task == "xnli":
        cells = (
            f'<td class="num score">{s["accuracy"]}%</td>'
            f'<td class="barcell"><span class="bar"><i style="width:{s["accuracy"]}%"></i></span></td>'
            f'<td class="num ci">{s["ci"][0]}–{s["ci"][1]}</td>'
        )
    elif task == "milu":
        cells = (
            f'<td class="num score">{s["accuracy"]}%</td>'
            f'<td class="barcell"><span class="bar"><i style="width:{s["accuracy"]}%"></i></span></td>'
            f'<td class="num ci">{s["ci"][0]}–{s["ci"][1]}</td>'
        )
    else:
        cells = (
            f'<td class="num score">{s["em"]}%</td>'
            f'<td class="num">{s["f1"]}</td>'
            f'<td class="barcell"><span class="bar"><i style="width:{s["f1"]}%"></i></span></td>'
            f'<td class="num ci">{s["em_ci"][0]}–{s["em_ci"][1]}</td>'
        )
    return (f'<tr{cls}><td class="rank">{rank}</td>'
            f'<td class="model">{model.replace(":free", "")}<small>{name}</small></td>{cells}</tr>')

PARKED = {
    "nvidia/nemotron-3.5-lightning:free",
    "inclusionai/ling-3.0-flash-vl:free",
}

def pending_row(model):
    name = MODELS[model]
    label = "parked — endpoint congested" if model in PARKED else "running\u2026"
    return (f'<tr class="pending"><td class="rank">·</td>'
            f'<td class="model">{model.replace(":free", "")}<small>{name}</small></td>'
            f'<td class="num score" colspan="3">{label}</td></tr>')

def rows(task, scores):
    key = "f1" if task == "indicqa" else "accuracy"
    have = [(m, s) for m, s in scores[task].items()]
    have.sort(key=lambda kv: -kv[1][key])
    out = [row_html(task, i + 1, m, s) for i, (m, s) in enumerate(have)]
    out += [pending_row(m) for m in MODELS if m not in scores[task]]
    return "\n          ".join(out)

def bluff_rows(bluff):
    have = sorted(bluff.items(), key=lambda kv: kv[1]["bluff_rate"])
    out = []
    for rank, (m, s) in enumerate(have, 1):
        cls = ' class="top"' if rank == 1 else ""
        out.append(
            f'<tr{cls}><td class="rank">{rank}</td>'
            f'<td class="model">{m.replace(":free", "")}<small>{MODELS[m]}</small></td>'
            f'<td class="num score">{s["bluff_rate"]}%</td>'
            f'<td class="barcell"><span class="bar"><i style="width:{s["bluff_rate"]}%"></i></span></td>'
            f'<td class="num ci">{s["n_traps"]} traps</td></tr>')
    return "\n          ".join(out)

def patch_html(scores, bluff, xnli):
    html = (ROOT / "index.html").read_text()
    for task, tag in (("milu", "MILU"), ("indicqa", "QA"), ("xnli", "XNLI")):
        html = re.sub(
            rf"(<!--ROWS:{tag}-->)(.*?)(<!--/ROWS:{tag}-->)",
            lambda m: m.group(1) + "\n          " + rows(task, scores) + "\n          " + m.group(3),
            html, flags=re.S)
    html = re.sub(
        r"(<!--ROWS:BLUFF-->)(.*?)(<!--/ROWS:BLUFF-->)",
        lambda m: m.group(1) + "\n          " + bluff_rows(bluff) + "\n          " + m.group(3),
        html, flags=re.S)
    (ROOT / "index.html").write_text(html)

def charts(scores):
    """Reference-style small-multiples comparison chart: one panel per metric,
    one color per model, value labels on top. All-English on purpose --
    matplotlib cannot shape Tamil script (no HarfBuzz), so Tamil stays on the
    site where browsers shape it correctly."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ORDER = sorted(scores["milu"], key=lambda m: -scores["milu"][m]["accuracy"])
    SHORT = {
        "google/gemini-3.8-flash": "Gemini 3.8",
        "openai/gpt-5.6-luna": "GPT-5.6 Luna",
        "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1",
        "z-ai/glm-5.3-flash": "GLM 5.3",
        "xiaomi/mimo-v2.5": "MiMo v2.5",
        "qwen/qwen3.8-flash": "Qwen 3.8",
        "inclusionai/ling-3.0-flash-vl:free": "Ling 3.0",
        "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B",
        "poolside/laguna-s-2.1:free": "Laguna-S 2.1",
        "nvidia/nemotron-3.5-lightning:free": "Nemotron 3.5",
    }
    PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
               "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#a0cbe8"]
    color = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(ORDER)}
    BG, INK, SOFT = "#fbfbf8", "#26221b", "#8d8779"
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.facecolor": BG, "axes.facecolor": BG,
        "text.color": INK, "axes.edgecolor": INK,
        "xtick.color": INK, "ytick.color": SOFT,
    })

    panels = [
        ("MILU \u00b7 exam MCQs", "accuracy", "{:.1f}", scores["milu"]),
        ("IndicQA \u00b7 exact match", "em", "{:.0f}", scores["indicqa"]),
        ("IndicQA \u00b7 F1 (word overlap)", "f1", "{:.1f}", scores["indicqa"]),
        ("IndicXNLI \u00b7 3-way logic", "accuracy", "{:.1f}", scores.get("xnli") or {}),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.2, 4.7), dpi=150)
    today = date.today().isoformat()
    for ax, (title, key, fmt, sc) in zip(axes, panels):
        ms = [m for m in ORDER if m in sc]
        vals = [sc[m][key] for m in ms]
        ax.bar(range(len(ms)), vals, color=[color[m] for m in ms],
               width=.72, zorder=3)
        for x, v in enumerate(vals):
            ax.text(x, v + 2, fmt.format(v), ha="center", va="bottom",
                    fontsize=8.6, fontweight="bold", color=INK, zorder=4)
        ax.set_xticks(range(len(ms)))
        ax.set_xticklabels([SHORT[m] for m in ms], rotation=32,
                           ha="right", fontsize=8.2)
        ax.set_title(title, fontsize=11.5, fontweight="bold", loc="left", pad=10)
        ax.set_ylim(0, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#d8d4c8")
        ax.tick_params(left=False, bottom=False)
        ax.grid(axis="y", color="#e7e4da", lw=.8, zorder=0)
    fig.suptitle("Tamil Bench \u2014 how 9 AI models score on three exams written in Tamil",
                 x=.02, y=.99, ha="left", fontsize=15, fontweight="bold")
    fig.text(.02, .925, "MILU: % of exam questions correct. IndicQA: exact-match vs partial (word-overlap) credit. "
             "IndicXNLI: 3-way entailment logic (chance = 33%). 0-shot, temp 0, " + today + ".",
             ha="left", fontsize=9, color=SOFT)
    fig.text(.02, .015, "MILU: 199 Qs \u00b7 IndicQA: 100 Qs \u00b7 IndicXNLI: 200 Qs (all seed 42) \u00b7 "
             "95% confidence intervals in the site tables \u00b7 "
             "github.com/LogicIncZo/tamil-bench",
             ha="left", fontsize=8, color=SOFT)
    fig.tight_layout(rect=(0, .05, 1, .88))
    OUT = ROOT / "assets"; OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "chart-scores.png", facecolor=BG)
    plt.close(fig)
    for old in ("chart-milu.png", "chart-indicqa.png"):
        (OUT / old).unlink(missing_ok=True)

def main():
    push = "--no-push" not in sys.argv
    data = load()
    scores = compute(data)
    bluff = bluff_scores(data)
    xnli = xnli_scores(data)
    summary = {
        "generated": date.today().isoformat(),
        "bench": "tamil-bench",
        "tasks": {
            "milu": {"n_target": 199, "models": scores["milu"]},
            "indicqa": {"n_target": 100, "models": scores["indicqa"]},
            "indicxnli": {"n_target": 200, "models": scores.get("xnli", {})},
            "indicqa_bluff": {"n_target": 100, "models": bluff},
        },
        "pending": [m for m in MODELS if m not in scores["milu"] or m not in scores["indicqa"]],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    patch_html(scores, bluff, xnli)
    charts(scores)
    print(f"milu models: {len(scores['milu'])}, indicqa models: {len(scores['indicqa'])}, "
          f"xnli models: {len(scores.get('xnli', {}))}, bluff models: {len(bluff)}")
    print(f"pending: {summary['pending']}")
    if push:
        subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
        r = subprocess.run(["git", "commit", "-m",
                            f"refresh: scores+charts {date.today().isoformat()} (build_site.py)"],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode == 0:
            subprocess.run(["git", "push"], cwd=ROOT, check=True)
            print("pushed")
        else:
            print("no changes to commit" if "nothing to commit" in r.stdout + r.stderr else r.stderr)

if __name__ == "__main__":
    main()
