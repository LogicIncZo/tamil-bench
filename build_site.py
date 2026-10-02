#!/usr/bin/env python3
"""tamil-bench site refresh: recompute scores, regenerate summary.json + charts, patch index.html.

Usage: python3 build_site.py [--no-push]
Idempotent — safe to re-run after each new results file lands.
"""
import html, json, math, re, subprocess, sys
from functools import lru_cache
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
FULL_MIN = {"milu": 150, "indicqa": 90, "xnli": 150}

MODELS = {
    "google/gemini-3.8-flash": "Gemini 3.8 Flash",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "openai/gpt-6-luna": "GPT-6 Luna",
    "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1 Flash",
    "z-ai/glm-5.3-flash": "GLM 5.3 Flash",
    "qwen/qwen3.8-flash": "Qwen 3.8 Flash",
    "xiaomi/mimo-v2.5": "Xiaomi MiMo v2.5",
    "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B A4B",
    "inclusionai/ling-3.0-flash-vl:free": "Ling 3.0 Flash VL",
    "nvidia/nemotron-3.5-lightning:free": "Nemotron 3.5 Lightning",
    "poolside/laguna-s-2.1:free": "Poolside Laguna-S 2.1",
    "nex-agi/nex-n2.5-mini": "Nex N2.5 Mini",
    "openai/gpt-oss-20b": "GPT-OSS 20B",
    "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super 120B A12B",
    "nvidia/nemotron-3-ultra-550b-a55b:free": "Nemotron 3 Ultra 550B A55B",
    "anthropic/claude-sonnet-5.5": "Claude Sonnet 5.5",
    "stealth/space-bunny-alpha": "Space Bunny Alpha",
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
        out[task][model].append((n, rows, f.name))
    return out

@lru_cache(maxsize=None)
def sheet_date(filename):
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cs", "--", f"results/{filename}"],
        cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() or "Unknown"

def run_cost(rows):
    """Total OpenRouter cost recorded for a sheet, and how many of its rows
    carried a cost at all. Sheets written before usage accounting was enabled
    report nothing, so the coverage count is what makes a total trustworthy."""
    total = 0.0
    covered = 0
    for r in rows:
        usage = r.get("usage")
        if isinstance(usage, dict) and usage.get("cost") is not None:
            total += float(usage["cost"])
            covered += 1
    return round(total, 4), covered

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
        n, rows, filename = max(runs, key=lambda r: r[0])
        valid = valid_rows(rows)
        if not valid:
            continue
        m = len(valid)
        k = sum(1 for r in valid if r.get("correct"))
        lo, hi = wilson(k / m, m)
        cost, cost_covered = run_cost(rows)
        scores["milu"][model] = {
            "n": m, "n_errors": len(rows) - m, "accuracy": round(100 * k / m, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)], "tested_on": sheet_date(filename),
            "cost": cost, "cost_covered": cost_covered, "cost_rows": len(rows),
        }
    for model, runs in data["indicqa"].items():
        n, rows, filename = max(runs, key=lambda r: r[0])
        valid = valid_rows(rows)
        if not valid:
            continue
        m = len(valid)
        em = sum(float(r["em"]) for r in valid) / m
        f1 = 100 * sum(float(r["f1"]) for r in valid) / m
        lo, hi = wilson(em, m)
        cost, cost_covered = run_cost(rows)
        scores["indicqa"][model] = {
            "n": m, "n_errors": len(rows) - m, "em": round(100 * em, 1), "f1": round(f1, 1),
            "em_ci": [round(100 * lo, 1), round(100 * hi, 1)], "tested_on": sheet_date(filename),
            "cost": cost, "cost_covered": cost_covered, "cost_rows": len(rows),
        }
    for model, runs in data.get("xnli", {}).items():
        n, rows, filename = max(runs, key=lambda r: r[0])
        valid = valid_rows(rows)
        if not valid:
            continue
        m = len(valid)
        k = sum(1 for r in valid if r.get("correct"))
        lo, hi = wilson(k / m, m)
        cost, cost_covered = run_cost(rows)
        scores["xnli"][model] = {
            "n": m, "n_errors": len(rows) - m, "accuracy": round(100 * k / m, 1),
            "ci": [round(100 * lo, 1), round(100 * hi, 1)], "tested_on": sheet_date(filename),
            "cost": cost, "cost_covered": cost_covered, "cost_rows": len(rows),
        }
    return scores


COST_UNRECORDED = 1e6


def cost_display(score):
    """(sort key, label) for one run's cost. Sheets written before OpenRouter
    usage accounting carried no charge at all, so a blank is reported as
    "not recorded" and pushed to the end of a numeric sort rather than
    silently reading as free."""
    cost = score.get("cost")
    covered = score.get("cost_covered", 0)
    if cost is None or covered == 0:
        return COST_UNRECORDED, "not recorded"
    label = f"${cost:.4f}"
    if covered < score.get("cost_rows", 0):
        label += "*"
    return cost, label


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
        n, rows, _ = max(runs, key=lambda r: r[0])
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
        n, rows, _ = max(runs, key=lambda r: r[0])
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
            f'<td class="model">{model.replace(":free", "")}<small>{name}</small></td>{cells}'
            f'<td class="num cost">{cost_display(s)[1]}</td></tr>')

PARKED = {
    "nvidia/nemotron-3.5-lightning:free",
    "inclusionai/ling-3.0-flash-vl:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
}

# Columns left over after rank+model, now that every scored table also carries
# a run-cost column. indicqa is widest: EM, F1, bar, CI, cost.
PENDING_SPAN = {"milu": 4, "indicqa": 5, "xnli": 4}


def pending_row(model, task):
    name = MODELS[model]
    label = "parked — endpoint congested" if model in PARKED else "running\u2026"
    return (f'<tr class="pending"><td class="rank">·</td>'
            f'<td class="model">{model.replace(":free", "")}<small>{name}</small></td>'
            f'<td class="num score" colspan="{PENDING_SPAN[task]}">{label}</td></tr>')

def rows(task, scores):
    key = "f1" if task == "indicqa" else "accuracy"
    have = [(m, s) for m, s in scores[task].items()]
    have.sort(key=lambda kv: -kv[1][key])
    out = [row_html(task, i + 1, m, s) for i, (m, s) in enumerate(have)]
    out += [pending_row(m, task) for m in MODELS if m not in scores[task]]
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

def test_page(task, title, title_ta, score_columns, scores):
    headers = ["Model · மாதிரி", "Tested date · சோதனை தேதி", "Valid n · சரியான விடைகள்", "Errors · பிழைகள்"] + [label for _, label in score_columns] + ["95% CI · நம்பிக்கை வரம்பு", "Cost USD · செலவு"]
    header_html = "".join(f'<th scope="col" tabindex="0" aria-sort="none">{label} ↕</th>' for label in headers)
    rows = []
    for model, score in scores[task].items():
        ci = score.get("ci", score.get("em_ci"))
        values = [score["tested_on"], score["n"], score["n_errors"]]
        values.extend(score[key] for key, _ in score_columns)
        values.append(f"{ci[0]:.1f}–{ci[1]:.1f}%")
        cells = [
            f'<td class="model" data-sort="{html.escape(MODELS[model], quote=True)}" data-type="text">'
            f'<strong>{html.escape(MODELS[model])}</strong><small>{html.escape(model.replace(":free", ""))}</small></td>'
        ]
        for index, value in enumerate(values, start=1):
            kind = "number" if 2 <= index < len(values) else ("date" if index == 1 else "text")
            cells.append(f'<td data-sort="{html.escape(str(value), quote=True)}" data-type="{kind}">{html.escape(str(value))}</td>')
        cost_key, cost_label = cost_display(score)
        cells.append(f'<td data-sort="{cost_key:g}" data-type="number">{html.escape(cost_label)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    nav = " · ".join(f'<a href="{path}">{label}</a>' for path, label in (
        ("index.html", "Home / முகப்பு"), ("milu.html", "MILU"),
        ("indicqa.html", "IndicQA"), ("indicxnli.html", "IndicXNLI"), ("contribute.html", "Contribute / பங்களிப்பு")))
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · Tamil Bench</title><style>
:root{{--paper:#f3ead6;--ink:#201a10;--muted:#6b6150;--line:#d6c7a6;--red:#c22b2b}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1100px;margin:auto;padding:clamp(20px,5vw,56px)}}a{{color:#174f72}}nav{{margin-bottom:36px}}nav a{{margin-right:12px;white-space:nowrap}}
h1{{font: bold clamp(2rem,6vw,3.6rem)/1.05 Georgia,serif;margin:.2em 0}}.ta{{font-size:.65em;color:var(--red)}}p{{color:var(--muted)}}
.table-wrap{{overflow-x:auto;margin-top:24px;border:1px solid var(--line);background:#fbf7ee}}table{{border-collapse:collapse;width:100%;min-width:760px}}
th,td{{padding:12px 14px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}}th{{white-space:normal;background:#ece0c4;cursor:pointer;user-select:none}}
th:hover,th:focus{{background:#e3d4b2}}tbody tr:hover{{background:var(--paper)}}td[data-type="number"]{{text-align:right;font-variant-numeric:tabular-nums}}
td.model strong{{display:block}}td.model small{{display:block;color:var(--muted);font-size:.72em;letter-spacing:.02em}}
.note{{font-size:.9rem;margin-top:14px}}@media(max-width:600px){{main{{padding:22px 14px}}}}
</style></head><body><main><nav aria-label="Test pages">{nav}</nav>
<h1>{title}<br><span class="ta">{title_ta}</span></h1>
<p>Sortable results · மாதிரி, சோதனை தேதி, மாதிரி அளவு, பிழைகள், மதிப்பெண், செலவு ஆகியவற்றை வரிசைப்படுத்தலாம். Click a column heading to sort.</p>
<div class="table-wrap"><table><thead><tr>{header_html}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p class="note">Tested date is the result sheet's repository record date; older sheets do not contain a run timestamp. Scores use each model's latest qualifying sheet. Run cost is the total charge OpenRouter recorded in that sheet's own answer rows — the price of one run of this test. * marks a sheet where only some calls reported a charge, and “not recorded” means the sheet predates usage accounting. செலவு என்பது விடைத்தாள் வரிசைகளில் பதிவான கட்டணத்தின் தொகை; * என்பது சில வரிசைகளில் மட்டும் கட்டணம் பதிவானதைக் குறிக்கிறது. சோதனைத் தேதி என்பது விடைத்தாள் களஞ்சியத்தில் பதிவான தேதி; பழைய விடைத்தாள்களில் இயக்க நேரமுத்திரை இல்லை. <a href="https://github.com/LogicIncZo/tamil-bench/tree/main/results">View answer sheets · விடைத்தாள்கள்</a>.</p>
</main><script>
document.querySelectorAll('th').forEach((header,column)=>{{const sort=()=>{{const body=header.closest('table').tBodies[0];const ascending=header.getAttribute('aria-sort')!=='ascending';const type=body.rows[0]?.cells[column]?.dataset.type||'text';const rows=Array.from(body.rows);rows.sort((left,right)=>{{const a=left.cells[column].dataset.sort||left.cells[column].textContent;const b=right.cells[column].dataset.sort||right.cells[column].textContent;const result=type==='number'?Number(a)-Number(b):a.localeCompare(b,undefined,{{numeric:true,sensitivity:'base'}});return (ascending?1:-1)*result}});rows.forEach(row=>body.appendChild(row));header.closest('tr').querySelectorAll('th').forEach(cell=>cell.setAttribute('aria-sort','none'));header.setAttribute('aria-sort',ascending?'ascending':'descending')}};header.addEventListener('click',sort);header.addEventListener('keydown',event=>{{if(event.key==='Enter'||event.key===' '){{event.preventDefault();sort()}}}})}});
</script></body></html>'''

def generate_test_pages(scores):
    pages = (
        ("milu.html", "milu", "MILU · Exam MCQs", "MILU · பல்தேர்வு", [("accuracy", "Accuracy % · துல்லியம் %")]),
        ("indicqa.html", "indicqa", "IndicQA · Reading Comprehension", "IndicQA · வாசிப்புப் புரிதல்", [("em", "Exact Match % · முழுப் பொருத்தம் %"), ("f1", "F1 % · சொல் ஒற்றுமை %")]),
        ("indicxnli.html", "xnli", "IndicXNLI · Three-way Inference", "IndicXNLI · மும்முனை அனுமானம்", [("accuracy", "Accuracy % · துல்லியம் %")]),
    )
    for filename, task, title, title_ta, columns in pages:
        (ROOT / filename).write_text(test_page(task, title, title_ta, columns, scores))

UPI_ID = "logic@ybl"
UPI_INTENT = "upi://pay?pa=logic@ybl&pn=CashlessConsumer&cu=INR&tn=Tamil+Bench+contribution"
REQUEST_MAIL = "cashlessconsumerin@gmail.com"
REPO_URL = "https://github.com/LogicIncZo/tamil-bench"
NEW_ISSUE_URL = REPO_URL + "/issues/new"
CONTRIBUTING_URL = REPO_URL + "/blob/main/CONTRIBUTING.md"
NEW_TEST_ISSUE_URL = REPO_URL + "/issues/new?template=new_test.yml"

TASK_LABELS = {
    "milu": "MILU (199 exam MCQs)",
    "indicqa": "IndicQA (100 reading questions)",
    "xnli": "IndicXNLI (200 three-way logic items)",
}
TASK_LABELS_TA = {
    "milu": "MILU (199 பல்தேர்வு வினாக்கள்)",
    "indicqa": "IndicQA (100 வாசிப்பு வினாக்கள்)",
    "xnli": "IndicXNLI (200 மும்முனை அனுமான வினாக்கள்)",
}


def project_spend(data):
    """Total OpenRouter charge behind the published scoreboard.

    Only the max-n qualifying sheet per (task, model) is summed -- the same
    sheet the scores come from -- so re-runs and n=5 smoke tests do not inflate
    the bill. Sheets written before usage accounting was enabled carry no
    charge at all, so `uncosted_sheets` is what stops a partial sum from
    reading as the whole bill.
    """
    per_task = defaultdict(float)
    per_model = defaultdict(float)
    uncosted = []
    total = 0.0
    for task, models in data.items():
        for model, runs in models.items():
            n, rows, filename = max(runs, key=lambda r: r[0])
            cost, covered = run_cost(rows)
            if covered == 0:
                uncosted.append(filename)
                continue
            per_task[task] += cost
            per_model[model] += cost
            total += cost
    return {
        "total": round(total, 4),
        "per_task": {t: round(v, 4) for t, v in per_task.items()},
        "per_model": {m: round(v, 4) for m, v in per_model.items()},
        "uncosted_sheets": sorted(uncosted),
    }


def all_sheets_spend():
    """Every charge the project has actually recorded, across every sheet in
    results/ -- not only the ones that qualify for the leaderboard.

    project_spend() answers "what did the published scores cost", which is
    the right basis for a per-run cost column but the wrong basis for a
    project total: smoke tests, superseded re-runs and abandoned partial
    sheets all spent money too. Two extra buckets come out of this scan:

      free_sheets      -- `:free` endpoints reporting no charge. Genuinely
                          $0.00, not a gap in the record.
      unrecorded_paid  -- sheets for a commercial model that carry no usage
                          row at all. These were billed, but the sheet does
                          not say by how much, so the project total is a
                          floor and these are the size of the hole in it.
    """
    recorded = 0.0
    recorded_sheets = 0
    free_sheets = 0
    unrecorded_paid = []
    for f in sorted(RESULTS.glob("*.jsonl")):
        p = parse_stem(f.stem)
        if not p:
            continue
        task, model, n = p
        rows = [json.loads(l) for l in
                f.open(encoding="utf-8", errors="replace") if l.strip()]
        cost, covered = run_cost(rows)
        if covered:
            recorded += cost
            recorded_sheets += 1
        elif model.endswith(":free"):
            free_sheets += 1
        else:
            unrecorded_paid.append({"file": f.name, "model": model, "rows": len(rows)})
    return {
        "recorded": round(recorded, 4),
        "recorded_sheets": recorded_sheets,
        "free_sheets": free_sheets,
        "unrecorded_paid": unrecorded_paid,
        "unrecorded_paid_calls": sum(s["rows"] for s in unrecorded_paid),
    }


CONTRIBUTE_CSS = """
:root{--paper:#f3ead6;--ink:#201a10;--muted:#6b6150;--line:#d6c7a6;--red:#c22b2b;--green:#1f6b46}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:900px;margin:auto;padding:clamp(20px,5vw,56px)}a{color:#174f72}
nav{margin-bottom:36px}nav a{margin-right:12px;white-space:nowrap}
h1{font:bold clamp(2rem,6vw,3.4rem)/1.05 Georgia,serif;margin:.2em 0}
.ta{font-size:.62em;color:var(--red)}h2{font:bold 1.5rem/1.2 Georgia,serif;margin:2.2em 0 .2em}
p{color:var(--muted)}section{background:#fbf7ee;border:1px solid var(--line);padding:clamp(18px,3vw,32px);margin:26px 0}
.bignum{font:bold clamp(2.6rem,9vw,4.6rem)/1 Georgia,serif;color:var(--ink);letter-spacing:-.02em}
.sub{font-size:.95rem;color:var(--muted);margin-top:.4em}
table{border-collapse:collapse;width:100%;margin-top:14px;min-width:420px}
th,td{padding:9px 12px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{background:#ece0c4;font-size:.85rem;letter-spacing:.03em;text-transform:uppercase}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.table-wrap{overflow-x:auto}
.qr{display:block;width:min(320px,72vw);height:auto;margin:0 auto;border:6px solid #fff;box-shadow:0 2px 10px rgba(0,0,0,.08)}
.upi-id{font:bold 1.5rem/1.2 ui-monospace,Menlo,Consolas,monospace;background:#fff;border:1px dashed var(--line);padding:12px 16px;display:inline-block;margin:14px 0}
button,.btn{font:inherit;cursor:pointer;border-radius:6px;padding:12px 18px;border:1px solid var(--ink);background:var(--ink);color:#fbf7ee;text-decoration:none;display:inline-block}
button.secondary,.btn.secondary{background:transparent;color:var(--ink)}
label{display:block;font-weight:600;margin:16px 0 4px}
input[type=text],textarea{width:100%;padding:10px 12px;font:inherit;border:1px solid var(--line);background:#fff;color:var(--ink)}
.hint{font-size:.85rem;color:var(--muted);font-weight:400;margin-top:2px}
fieldset{border:1px solid var(--line);margin:16px 0;padding:12px 16px}legend{font-weight:600;padding:0 6px}
.checks{display:flex;flex-wrap:wrap;gap:14px}checks{display:flex;gap:8px;align-items:center}
.checks label{font-weight:400;display:flex;gap:6px;align-items:center;margin:0}
.actions{display:flex;flex-wrap:wrap;gap:12px;margin-top:20px}
#req-preview{white-space:pre-wrap;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;background:#fff;border:1px solid var(--line);padding:12px;margin-top:18px;max-height:260px;overflow:auto}
.note{font-size:.88rem}.ok{color:var(--green)}@media(max-width:600px){main{padding:22px 14px}}
"""


def _esc(s):
    return html.escape(str(s))


def contribute_page(data):
    spend = project_spend(data)
    total = spend["total"]
    uncosted_n = len(spend["uncosted_sheets"])
    allc = all_sheets_spend()
    all_recorded = allc["recorded"]
    n_unrec = len(allc["unrecorded_paid"])
    n_unrec_calls = allc["unrecorded_paid_calls"]
    # Off-board spend: smoke tests, superseded re-runs and partial sheets that
    # a real charge was recorded against but which never reach the leaderboard.
    off_board = round(all_recorded - total, 4)

    # --- cost cards -------------------------------------------------------
    if all_recorded >= 1:
        total_str = f"${all_recorded:,.2f}"
    elif all_recorded > 0:
        total_str = f"${all_recorded:,.4f}"
    else:
        total_str = "$0.00"
    all_exact = f"${all_recorded:,.4f}" if all_recorded > 0 else "$0.00"
    # Full precision for the lower-bound claim -- rounding $1.5518 to $1.55 and
    # then calling $1.55 the floor would understate what is already proven.
    total_exact = f"${total:,.4f}" if total > 0 else "$0.00"

    # per-task rows, biggest first
    task_rows = []
    for task in ("milu", "indicqa", "xnli"):
        v = spend["per_task"].get(task)
        if v is None:
            continue
        task_rows.append(
            f"<tr><td><strong>{_esc(TASK_LABELS[task])}</strong>"
            f"<br><small>{_esc(TASK_LABELS_TA[task])}</small></td>"
            f'<td class="num">${v:,.4f}</td></tr>')

    # per-model rows, biggest first
    model_rows = []
    for model, v in sorted(spend["per_model"].items(), key=lambda kv: -kv[1]):
        model_rows.append(
            f'<tr><td><strong>{_esc(MODELS.get(model, model))}</strong>'
            f"<br><small>{_esc(model.replace(':free', ''))}</small></td>"
            f'<td class="num">${v:,.4f}</td></tr>')

    n_models_all = len({m for task in data for m in data[task]})
    n_models_costed = len(spend["per_model"])
    n_free = sum(1 for m in spend["per_model"] if m.endswith(":free"))
    n_paid = sum(1 for v in spend["per_model"].values() if v > 0)

    # Kept as a fragment, not a <p>: it is appended inside another <p> and a
    # nested <p> would close the parent early in the browser's parser.
    coverage = ""
    if uncosted_n:
        coverage = (
            f"<br><br>Board coverage: {n_models_costed} of {n_models_all} models on "
            f"the board reported a charge; {uncosted_n} qualifying sheets predate usage "
            f"accounting and record none. Across the whole results/ directory the "
            f"accounted total is {all_exact}, with {n_unrec} further paid-model "
            f"sheets unaccounted. <a href=\"{REPO_URL}/tree/main/results\">Check the raw "
            f"sheets · விடைத்தாள்களைப் பார்க்கவும்</a>.")

    # --- model id suggestions for the request form ------------------------
    known = sorted(MODELS, key=lambda m: MODELS[m])
    datalist = "".join(f'<option value="{_esc(m)}"></option>' for m in known)

    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Contribute to Tamil Bench · பங்களிப்பு</title>
<meta name="description" content="Contribute to Tamil Bench, the open API-only Tamil LLM benchmark: fund API runs, improve the runner, or add a new test.">
<style>{CONTRIBUTE_CSS}</style></head><body><main>
<nav aria-label="Site pages"><a href="index.html">Home / முகப்பு</a> · <a href="milu.html">MILU</a> · <a href="indicqa.html">IndicQA</a> · <a href="indicxnli.html">IndicXNLI</a> · <a href="{REPO_URL}">GitHub</a></nav>

<h1>Contribute to Tamil Bench<br><span class="ta">தமிழ் பெஞ்சைக்கு பங்களியுங்கள்</span></h1>
<p>Tamil Bench runs open-weight and commercial language models through the same
three Tamil exams, and publishes every score with the raw answer sheet behind
it. The runs are paid for out of pocket, and this page shows exactly what they
cost.</p>
<p><strong>Money is only one of three ways in, and it is the least demanding.</strong>
Anyone can help, none of them require you to run a single command:</p>
<ul>
<li><strong>Contribute money</strong> — every rupee pays for API calls that put
new models on the board. Free and <code>:free</code> endpoints are run at zero
cost either way.</li>
<li><strong>Contribute code</strong> — new tasks, better scoring, Tamil
transliteration, a dataset loader, a fix for a failing endpoint. The runner is
MIT-licensed and small enough to read in an afternoon.</li>
<li><strong>Add a new test</strong> — the highest-value contribution there is.
See the section below for exactly what a new task has to satisfy before it can
appear on the board.</li>
</ul>

<section>
<h2 style="margin-top:0">What the benchmark has cost so far<br><span class="ta">இதுவரை செலவானதன் மொத்தம்</span></h2>
<div class="bignum">{_esc(total_str)}</div>
<p class="sub">Every OpenRouter charge this project can account for, summed from
the per-answer <code>usage.cost</code> field across all {allc["recorded_sheets"]}
answer sheets that recorded one. No salaries, no servers, no paid datasets —
just API calls.</p>
<p class="sub">Of that, <strong>${total:,.4f}</strong> is the runs standing on
the leaderboard and <strong>${off_board:,.4f}</strong> is off-board work —
n=5 smoke tests, superseded re-runs and partial sheets that were still billed
but never reached the board. A further <strong>{n_unrec}</strong> paid-model
sheets covering <strong>{n_unrec_calls:,}</strong> API calls record no charge at
all; they were certainly billed, but the sheets do not say by how much, so this
figure is a <em>floor</em>, not a closed total. {allc["free_sheets"]} free-tier
<code>:free</code> sheets are excluded because they genuinely cost nothing.</p>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Test · சோதனை</th><th scope="col" class="num">Spend · செலவு</th></tr></thead>
<tbody>{''.join(task_rows) or '<tr><td colspan="2">No charged runs yet · இதுவரை கட்டணம் இல்லை</td></tr>'}</tbody>
</table></div>
<p class="note">{n_models_costed} of {n_models_all} board models reported a charge,
but only {n_paid} of those were actually billed -- the other {n_models_costed - n_paid}
recorded $0.00 ({n_free} of them were free-tier `:free` endpoints, which are $0.00
by definition). Machines marked *free* cost nothing to run; the bill is made up
of the commercial endpoints. {coverage}</p>
</section>

<section>
<h2 style="margin-top:0">Per-model spend<br><span class="ta">மாதிரி வாரியான செலவு</span></h2>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Model · மாதிரி</th><th scope="col" class="num">Spend · செலவு</th></tr></thead>
<tbody>{''.join(model_rows) or '<tr><td colspan="2">No charged runs yet · இதுவரை கட்டணம் இல்லை</td></tr>'}</tbody>
</table></div>
</section>

<section aria-labelledby="upi-h">
<h2 id="upi-h" style="margin-top:0">Contribute money · UPI<br><span class="ta">UPI மூலம் பங்களிப்பு</span></h2>
<p>Scan with any UPI app (GPay, PhonePe, Paytm, BHIM) and enter whatever amount
you want. Every rupee goes toward API calls that keep new models on the board.
There is no minimum and no subscription — a one-time amount is fine.</p>
<img class="qr" src="assets/upi-qr.png" alt="UPI QR code for the contribution UPI ID logic@ybl" width="320" height="320">
<p style="text-align:center;margin-bottom:4px">or type the ID · அல்லது ID-ஐத் தட்டச்சு செய்யவும்</p>
<p style="text-align:center"><span class="upi-id" id="upi-id">{UPI_ID}</span></p>
<p class="note" style="text-align:center">A one-tap UPI link: <a id="upi-link" href="{UPI_INTENT}">{UPI_INTENT[:34]}…</a></p>
</section>

<section aria-labelledby="code-h">
<h2 id="code-h" style="margin-top:0">Contribute code<br><span class="ta">கோடு பங்களிப்பு</span></h2>
<p>The whole benchmark is three Python scripts and a folder of answer sheets, MIT
licensed. <a href="{CONTRIBUTING_URL}">CONTRIBUTING.md</a> is the full guide;
this is the short version of what is actually useful.</p>
<ul>
<li><strong>Add a task runner.</strong> Each test is one
<code>run_&lt;task&gt;()</code> function in <code>bench.py</code> plus a subcommand in
<code>main()</code>. This is the main way in.</li>
<li><strong>Fix or extend scoring.</strong> IndicQA uses exact match and token
F1; MILU and IndicXNLI use accuracy over valid rows. A better Tamil-aware
normaliser or a confidence interval that reflects the real sampling is a
genuine improvement.</li>
<li><strong>Handle the awkward cases.</strong> Models that answer in Tamil script,
in English, in transliteration, or in a mix of all three. A model that fails on
format rather than knowledge is a measurement bug, and the fix belongs here.</li>
<li><strong>Report a broken run.</strong> A dead endpoint, a sheet with fewer
rows than expected, a score that looks wrong. <a href="{NEW_ISSUE_URL}">Open an
issue</a> &mdash; no run needed to do this.</li>
</ul>
<p>Small, reviewable pull requests are easier to land than large ones. If you are
unsure whether something is wanted, open the issue first &mdash; that costs you a
paragraph and saves you a wasted afternoon.</p>
</section>

<section aria-labelledby="test-h">
<h2 id="test-h" style="margin-top:0">Add a new test<br><span class="ta">புதிய சோதனை சேர்க்கவும்</span></h2>
<p>This is the contribution with the most leverage: every test added widens what
the board can detect. A task qualifies if it is Tamil-specific, has gold labels
that are not guessable from the prompt alone, and is not already covered by
MILU, IndicQA or IndicXNLI.</p>
<ol>
<li><strong>Propose it first.</strong> <a href="{NEW_TEST_ISSUE_URL}">Open a
new-test issue</a> with the dataset source, licence, language, size, and what a
good score means. Datasets that cannot be redistributed belong in
<code>data/</code>, which is gitignored &mdash; do not commit a copy.</li>
<li><strong>Write the runner.</strong> A <code>run_&lt;task&gt;()</code>
function in <code>bench.py</code> that samples with
<code>SEED = 42</code>, calls the model at temperature 0, and writes
<code>results/&lt;task&gt;_&lt;org&gt;_&lt;model-slug&gt;_n&lt;N&gt;.jsonl</code>
(<code>/</code> becomes <code>_</code> in the model id). Register the
subcommand in <code>main()</code>.</li>
<li><strong>Gate it.</strong> Add a <code>FULL_MIN</code> entry in
<code>build_site.py</code> so a smoke-test sheet can never reach the leaderboard.
MILU needs 150 rows, IndicQA 90, IndicXNLI 150; pick a floor at least as large as
your real task and say why in the PR.</li>
<li><strong>Label it.</strong> Add entries to <code>TASK_LABELS</code> (English)
and <code>TASK_LABELS_TA</code> (Tamil), and register the page in
<code>generate_test_pages()</code> so it gets its own sortable scoreboard.</li>
<li><strong>Show it works.</strong> Run at least two models end to end and commit
the two sheets. Scores computed from sheets nobody can re-derive are not
evidence.</li>
</ol>
<p>Three rules are not negotiable, because breaking any of them makes the board
quietly wrong rather than obviously broken: <strong>sampling stays at seed
42</strong> so models stay comparable; <strong>failed API calls are never
scored as 0%</strong> &mdash; they are counted and reported as
<code>n_errors</code>; and <strong>result sheets are append-only evidence</strong>,
never edited to fix a number.</p>
</section>

<section aria-labelledby="req-h">
<h2 id="req-h" style="margin-top:0">Request a run for any model<br><span class="ta">ஏதேனும் மாதிரிக்கான ஓட்டைக் கோரவும்</span></h2>
<p>Name any OpenRouter model id (or a free/open endpoint you want compared) and
which exams to run. Runs cost roughly a few dollars each for frontier models and
nothing for <code>:free</code> endpoints; a request does not commit you to
anything. If nobody funds it, the run simply goes to the back of the queue.</p>
<form id="req-form">
  <label for="model">OpenRouter model id <span class="hint">e.g. google/gemma-4-26b-a4b-it:free</span></label>
  <input type="text" id="model" list="known-models" placeholder="vendor/model-name" required>
  <datalist id="known-models">{datalist}</datalist>

  <fieldset><legend>Which tests? · எந்த சோதனைகள்?</legend>
  <div class="checks">
    <label><input type="checkbox" name="task" value="milu" checked> MILU</label>
    <label><input type="checkbox" name="task" value="indicqa" checked> IndicQA</label>
    <label><input type="checkbox" name="task" value="xnli" checked> IndicXNLI</label>
  </div>
  </fieldset>

  <label for="who">Your name / handle <span class="hint">optional — for credit in the changelog</span></label>
  <input type="text" id="who" placeholder="@you or your name">

  <label for="note">Anything else? <span class="hint">optional</span></label>
  <textarea id="note" rows="3" placeholder="Reason for the request, deadline, or a specific endpoint variant."></textarea>

  <div class="actions">
    <button type="submit">Send request · கோரிக்கை அனுப்பு</button>
    <button type="button" class="secondary" id="copy-btn">Copy request text · நகலெடு</button>
    <a class="btn secondary" id="gh-link" href="{NEW_ISSUE_URL}">Open on GitHub · GitHub-ல் திற</a>
  </div>
</form>
<pre id="req-preview" aria-live="polite"></pre>
<p class="note" id="req-status"></p>
</section>

<section class="note">
<h2 style="margin-top:0">Where the money goes<br><span class="ta">பணம் எங்கே செல்கிறது</span></h2>
<ul>
<li><strong>API calls.</strong> Every paid run is metered through OpenRouter and
each answer sheet stores its own charge, so the totals above are auditable line
by line rather than estimated.</li>
<li><strong>Free endpoints stay free.</strong> <code>:free</code> routes and
open models are run on the free tier; they expand the board at zero cost.</li>
<li><strong>Open source.</strong> The runner, the scoring, and every result
sheet live in <a href="{REPO_URL}">{REPO_URL}</a> under an open licence.
Contributions buy runs and compute, not secrecy.</li>
<li><strong>Code counts as much as cash.</strong> A new test, a scoring fix, or
a corrected sheet is worth exactly as much to this project as a rupee, and
costs you nothing but the work.</li>
</ul>
<p>Prefer to wire the money differently? Email
<a href="mailto:{REQUEST_MAIL}">{REQUEST_MAIL}</a>.</p>
</section>
</main>
<script>
(function(){{
  var form=document.getElementById('req-form');
  var model=document.getElementById('model');
  var who=document.getElementById('who');
  var note=document.getElementById('note');
  var preview=document.getElementById('req-preview');
  var status=document.getElementById('req-status');
  var gh=document.getElementById('gh-link');

  function selected(){{return Array.prototype.slice.call(form.querySelectorAll('input[name=task]:checked')).map(function(c){{return c.value;}});}}
  function compose(){{
    var id=model.value.trim();
    var tasks=selected();
    var lines=['### Benchmark run request','','**Model:** '+ (id||'(not set)'),'**Tests:** '+(tasks.length?tasks.join(', '):'(none selected)')];
    if(who.value.trim()) lines.push('**Requested by:** '+who.value.trim());
    if(note.value.trim()) lines.push('','**Notes:**\\n'+note.value.trim());
    return lines.join('\\n');
  }}
  function refresh(){{
    var body=compose();
    preview.textContent=body;
    gh.href='{NEW_ISSUE_URL}?title='+encodeURIComponent('Run request: '+(model.value.trim()||'model'))+'&body='+encodeURIComponent(body);
    return body;
  }}
  form.addEventListener('input',refresh);
  form.addEventListener('submit',function(e){{
    e.preventDefault();
    if(!model.value.trim()){{status.textContent='Please enter a model id first. · முதலில் மாதிரி ID-ஐ உள்ளிடவும்.';return;}}
    var body=refresh();
    var subj='Tamil Bench run request: '+model.value.trim();
    window.location.href='mailto:{REQUEST_MAIL}?subject='+encodeURIComponent(subj)+'&body='+encodeURIComponent(body);
    status.innerHTML='Opening your mail app… if nothing happens, use “Copy request text” and email it to <a href="mailto:{REQUEST_MAIL}">{REQUEST_MAIL}</a> or open the GitHub link. · உங்கள் மின்னஞ்சல் திறக்கிறது…';
    status.className='note ok';
  }});
  document.getElementById('copy-btn').addEventListener('click',function(){{
    var body=refresh();
    (navigator.clipboard?navigator.clipboard.writeText(body):Promise.reject()).then(function(){{
      status.textContent='Request text copied. · நகலெடுக்கப்பட்டது.';status.className='note ok';
    }}).catch(function(){{
      status.textContent='Select the text above and copy it manually. · மேலே உள்ள உரையைத் தேர்ந்தெடுத்து நகலெடுங்கள்.';status.className='note';
    }});
  }});
  refresh();
}})();
</script></body></html>'''
    return page


REDIRECT = """<!doctype html>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url=contribute.html">
<link rel="canonical" href="contribute.html">
<title>Contribute to Tamil Bench · பங்களிப்பு</title>
<p>This page moved to <a href="contribute.html">Contribute / பங்களிப்பு</a>.</p>
"""


def generate_contribute_page(data):
    (ROOT / "contribute.html").write_text(contribute_page(data))
    # The page used to be sponsor.html; keep the old URL alive so shared links
    # do not 404, and point search engines at the new one.
    (ROOT / "sponsor.html").write_text(REDIRECT)


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
        "openai/gpt-6-luna": "GPT-6 Luna",
        "deepseek/deepseek-v4.1-flash": "DeepSeek V4.1",
        "z-ai/glm-5.3-flash": "GLM 5.3",
        "xiaomi/mimo-v2.5": "MiMo v2.5",
        "qwen/qwen3.8-flash": "Qwen 3.8",
        "inclusionai/ling-3.0-flash-vl:free": "Ling 3.0",
        "google/gemma-4-26b-a4b-it:free": "Gemma 4 26B",
        "poolside/laguna-s-2.1:free": "Laguna-S 2.1",
        "nvidia/nemotron-3.5-lightning:free": "Nemotron 3.5",
        "nex-agi/nex-n2.5-mini": "Nex N2.5",
        "openai/gpt-oss-20b": "GPT-OSS 20B",
        "nvidia/nemotron-3-super-120b-a12b:free": "Nemotron 3 Super",
        "nvidia/nemotron-3-ultra-550b-a55b:free": "Nemotron 3 Ultra",
        "anthropic/claude-sonnet-5.5": "Sonnet 5.5",
        "stealth/space-bunny-alpha": "Space Bunny",
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
    fig.suptitle(f"Tamil Bench \u2014 how {len(ORDER)} AI models score on three exams written in Tamil",
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
    generate_test_pages(scores)
    generate_contribute_page(data)
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
        "spend": {
            "accounted_usd": project_spend(data)["total"],
            "all_sheets_recorded_usd": all_sheets_spend()["recorded"],
            "note": (
                "accounted_usd covers only leaderboard-qualifying sheets. "
                "all_sheets_recorded_usd is every usage.cost in results/, including "
                "off-board smoke tests and superseded re-runs. Both are floors: "
                "paid-model sheets with no usage row are billed but unrecorded."
            ),
        },
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
