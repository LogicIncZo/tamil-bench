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
FULL_MIN = {"milu": 150, "indicqa": 90}

MODELS = {
    "google/gemini-3.8-flash": ("Gemini 3.8 Flash", "paid"),
    "openai/gpt-5.6-luna": ("GPT-5.6 Luna", "paid"),
    "deepseek/deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "paid"),
    "z-ai/glm-5.3-flash": ("GLM 5.3 Flash", "paid"),
    "qwen/qwen3.8-flash": ("Qwen 3.8 Flash", "paid"),
    "xiaomi/mimo-v2.5": ("Xiaomi MiMo v2.5", "value"),
    "google/gemma-4-26b-a4b-it:free": ("Gemma 4 26B A4B (free)", "value"),
    "inclusionai/ling-3.0-flash-vl:free": ("Ling 3.0 Flash VL (free)", "value"),
    "nvidia/nemotron-3.5-lightning:free": ("Nemotron 3.5 Lightning (free)", "value"),
    "poolside/laguna-s-2.1:free": ("Poolside Laguna-S 2.1 (free)", "value"),
}

def parse_stem(stem):
    for task in ("milu", "indicqa"):
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
    out = {"milu": defaultdict(list), "indicqa": defaultdict(list)}
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

def compute(data):
    scores = {"milu": {}, "indicqa": {}}
    for model, runs in data["milu"].items():
        n, rows = max(runs, key=lambda r: r[0])
        k = sum(1 for r in rows if r.get("correct"))
        errs = sum(1 for r in rows if str(r.get("prediction", "")).startswith("__ERROR__"))
        lo, hi = wilson(k / n, n)
        scores["milu"][model] = {
            "n": n, "n_errors": errs, "accuracy": round(100 * k / n, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)],
        }
    for model, runs in data["indicqa"].items():
        n, rows = max(runs, key=lambda r: r[0])
        ems = [float(r["em"]) for r in rows]
        em = sum(ems) / n
        f1 = 100 * sum(float(r["f1"]) for r in rows) / n
        errs = sum(1 for r in rows if str(r.get("prediction", "")).startswith("__ERROR__"))
        lo, hi = wilson(em, n)
        scores["indicqa"][model] = {
            "n": n, "n_errors": errs, "em": round(100 * em, 1), "f1": round(f1, 1),
            "em_ci": [round(100 * lo, 1), round(100 * hi, 1)],
        }
    return scores

def row_html(task, rank, model, s):
    name, _tier = MODELS[model]
    cls = ' class="top"' if rank == 1 else ""
    if task == "milu":
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
            f'<td class="model">{model}<small>{name}</small></td>{cells}</tr>')

def pending_row(model):
    name, _ = MODELS[model]
    return (f'<tr class="pending"><td class="rank">·</td>'
            f'<td class="model">{model}<small>{name}</small></td>'
            f'<td class="num score" colspan="3">running…</td></tr>')

def tier_rows(task, scores, tier):
    have = {m: s for m, s in scores[task].items() if MODELS.get(m, ("", "paid"))[1] == tier}
    key = "accuracy" if task == "milu" else "f1"
    ordered = sorted(have.items(), key=lambda kv: -kv[1][key])
    rows = [row_html(task, i + 1, m, s) for i, (m, s) in enumerate(ordered)]
    for m, (name, t) in MODELS.items():
        if t == tier and m not in have:
            rows.append(pending_row(m))
    return "\n          ".join(rows)

def patch_html(scores):
    html = (ROOT / "index.html").read_text()
    for task in ("milu", "indicqa"):
        for tier in ("paid", "value"):
            tag = "MILU" if task == "milu" else "QA"
            html = re.sub(
                rf"(<!--ROWS:{tag}:{tier}-->)(.*?)(<!--/ROWS:{tag}:{tier}-->)",
                lambda m: m.group(1) + "\n          " + tier_rows(task, scores, tier) + "\n          " + m.group(3),
                html, flags=re.S)
    (ROOT / "index.html").write_text(html)

def charts(scores):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager as fm
    import matplotlib.pyplot as plt
    for p in ("/usr/share/fonts/truetype/noto/NotoSerifTamil-Regular.ttf",
              "/usr/share/fonts/truetype/noto/NotoSerifTamil-Bold.ttf"):
        fm.fontManager.addfont(p)
    plt.rcParams.update({
        "font.family": ["Noto Serif Tamil", "DejaVu Serif"],
        "figure.facecolor": "#f7f2e7", "axes.facecolor": "#f7f2e7",
        "text.color": "#2b2117", "axes.edgecolor": "#2b2117",
        "axes.labelcolor": "#2b2117", "xtick.color": "#2b2117",
        "ytick.color": "#2b2117", "svg.fonttype": "none",
    })
    OUT = ROOT / "assets"; OUT.mkdir(exist_ok=True)
    INK, RED, SOFT, RULE = "#2b2117", "#b3382c", "#6f6457", "#d9cfbd"

    # Tamil titles: matplotlib cannot shape Indic scripts (no HarfBuzz), so we
    # render the two title lines with Pillow+Raqm and composite them on top.
    from PIL import Image, ImageDraw, ImageFont
    import io as _io
    TAMIL_FONT = "/usr/share/fonts/truetype/noto/NotoSerifTamil-Bold.ttf"
    EN_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
    PAPER_RGB = (247, 242, 231)
    INK_RGB = (43, 33, 23)

    def compose(out_path, tamil, english, fig):
        buf = _io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        chart = Image.open(buf).convert("RGBA")
        W = chart.width
        tf = ImageFont.truetype(TAMIL_FONT, max(28, round(W * .021)), layout_engine=ImageFont.Layout.RAQM)
        ef = ImageFont.truetype(EN_FONT, max(18, round(W * .0125)))
        tmp = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
        tb = tmp.textbbox((0, 0), tamil, font=tf)
        eb = tmp.textbbox((0, 0), english, font=ef)
        pad, gap = round(W * .012), round(W * .008)
        th, eh = tb[3] - tb[1], eb[3] - eb[1]
        H = pad + th + gap + eh + round(W * .01) + chart.height + pad
        canvas = Image.new("RGBA", (W, H), PAPER_RGB + (255,))
        d = ImageDraw.Draw(canvas)
        y = pad - tb[1]
        d.text(((W - (tb[2] - tb[0])) // 2 - tb[0], y), tamil, font=tf, fill=INK_RGB + (255,))
        y += th + gap
        d.text(((W - (eb[2] - eb[0])) // 2 - eb[0], y), english, font=ef, fill=INK_RGB + (255,))
        y += eh + round(W * .01)
        canvas.paste(chart, (0, y), chart)
        canvas.convert("RGB").save(out_path)
    today = date.today().isoformat()

    def col_for(m):
        return RED if MODELS.get(m, ("", "value"))[1] == "value" else INK

    # Paper 1 — MILU bars with CI whiskers
    milu = sorted(scores["milu"].items(), key=lambda kv: kv[1]["accuracy"])
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    ys = range(len(milu))
    for y, (m, s) in zip(ys, milu):
        ax.barh(y, s["accuracy"], height=.58, color=col_for(m), zorder=3)
        ax.errorbar(s["accuracy"], y,
                    xerr=[[s["accuracy"] - s["ci"][0]], [s["ci"][1] - s["accuracy"]]],
                    fmt="none", ecolor=SOFT, elinewidth=1.4, capsize=4, zorder=4)
        ax.text(s["accuracy"] + 2.6, y, f'{s["accuracy"]}%', va="center",
                fontsize=11, fontweight="bold")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([MODELS[m][0] for m, _ in milu], fontsize=11)
    ax.set_xlim(0, 108); ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(left=False)
    ax.grid(axis="x", color=RULE, lw=.7, zorder=0)
    import matplotlib.patches as mp
    ax.legend(handles=[mp.Patch(color=INK, label="Paid flash tier"),
                       mp.Patch(color=RED, label="Free / open-weight")],
              loc="lower right", frameon=False, fontsize=10)
    fig.text(.995, .01, f"tamil-bench · {today} · 0-shot, temp 0 · github.com/LogicIncZo/tamil-bench",
             ha="right", fontsize=8, color=SOFT)
    fig.tight_layout(rect=(0, .03, 1, 1))
    compose(OUT / "chart-milu.png",
            "தமிழ் பெஞ்ச் — பொது அறிவுத் தேர்வு",
            "Paper 1 · MILU — Tamil exam-style MCQs, accuracy with 95% CI", fig)
    plt.close(fig)

    # Paper 2 — IndicQA EM↔F1 dumbbells
    qa = sorted(scores["indicqa"].items(), key=lambda kv: kv[1]["f1"])
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    ys = range(len(qa))
    for y, (m, s) in zip(ys, qa):
        c = col_for(m)
        ax.plot([s["em"], s["f1"]], [y, y], color=c, lw=2.4, alpha=.55, zorder=2)
        ax.scatter([s["em"]], [y], s=64, color=c, zorder=3)
        ax.scatter([s["f1"]], [y], s=110, facecolors="#f7f2e7", edgecolors=c, linewidths=2.2, zorder=3)
        ax.text(s["f1"] + 1.8, y, f'F1 {s["f1"]}', va="center", fontsize=10, color=c, fontweight="bold")
        ax.text(s["em"] - 1.8, y, f'EM {s["em"]}%', va="center", ha="right", fontsize=9.5, color=SOFT)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([MODELS[m][0] for m, _ in qa], fontsize=11)
    ax.set_xlim(-8, 100); ax.set_xticks(range(0, 101, 20))
    ax.set_xticklabels([f"{v}%" for v in range(0, 101, 20)])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.tick_params(left=False)
    ax.grid(axis="x", color=RULE, lw=.7, zorder=0)
    fig.text(.995, .01, f"tamil-bench · {today} · n=100, seed 42 · github.com/LogicIncZo/tamil-bench",
             ha="right", fontsize=8, color=SOFT)
    fig.tight_layout(rect=(0, .03, 1, 1))
    compose(OUT / "chart-indicqa.png",
            "வாசிப்புப் புரிதல் தேர்வு",
            "Paper 2 · IndicQA — exact match (●) vs F1 (○): models read Tamil, but paraphrase", fig)
    plt.close(fig)

def main():
    push = "--no-push" not in sys.argv
    data = load()
    scores = compute(data)
    summary = {
        "generated": date.today().isoformat(),
        "bench": "tamil-bench",
        "tasks": {
            "milu": {"n_target": 199, "models": scores["milu"]},
            "indicqa": {"n_target": 100, "models": scores["indicqa"]},
        },
        "tiers": {m: t for m, (_, t) in MODELS.items()},
        "pending": [m for m in MODELS if m not in scores["milu"] or m not in scores["indicqa"]],
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    patch_html(scores)
    charts(scores)
    print(f"milu models: {len(scores['milu'])}, indicqa models: {len(scores['indicqa'])}")
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
