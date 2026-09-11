#!/usr/bin/env python3
"""Wire Paper 2b (bluff catch) + Paper 3 (IndicXNLI) into index.html, dedupe build_site, update charts."""
import re
from pathlib import Path

ROOT = Path(__file__).parent
idx = ROOT / "index.html"
html = idx.read_text()

# 1. two exams -> three exams (en + ta)
html = html.replace("We give AI models two exams", "We give AI models three exams")
html = html.replace("இரண்டு தேர்வுகளை AI மாதிரிகளுக்குக் கொடுத்து", "மூன்று தேர்வுகளை AI மாதிரிகளுக்குக் கொடுத்து")

# 2. 528 -> 527 (en + ta)
html = html.replace("1,276 answerable; 528 deliberately unanswerable", "1,276 answerable; 527 deliberately unanswerable")
html = html.replace("1,276 கேள்விகளுக்கு விடை உண்டு, 528 கேள்விகள் விடையற்றவை", "1,276 கேள்விகளுக்கு விடை உண்டு, 527 கேள்விகள் விடையற்றவை")

# 3. renumber later sections (do highest first)
html = html.replace('<span class="qta">வினா 6</span><h2><span data-lang="en">Run it yourself', '<span class="qta">வினா 8</span><h2><span data-lang="en">Run it yourself')
html = html.replace('<span class="qta">வினா 5</span><h2><span data-lang="en">What the marks tell us', '<span class="qta">வினா 7</span><h2><span data-lang="en">What the marks tell us')
html = html.replace('<span class="qta">வினா 4</span><h2><span data-lang="en">How to read the marks', '<span class="qta">வினா 6</span><h2><span data-lang="en">How to read the marks')

# 4. new sections before rubric
NEW = """<section id="bluff">
  <div class="wrap">
    <div class="qnum"><span class="qta">வினா 4</span><h2><span data-lang="en">Scoreboard · Paper 2b: the bluff catch</span><span class="ta-title" data-lang="ta">மதிப்பெண் அட்டணை · தாள் 2b: பொய்விடைப் பிடிக்கும் பந்தயம்</span></h2><span class="mark">[traps]</span></div>
    <p data-lang="en">Roughly 27 of each model's 100 reading-comprehension questions were <b>unanswerable traps</b> — the passage simply never says. The only honest answer is “I don't know.” <b>Bluff %</b> is the share of traps the model answered anyway — <b>lower is better</b>. The verified sample answer sheet in வினா 6 shows a real one.</p>
    <p class="ta-copy" data-lang="ta">ஒவ்வொரு மாதிரியின் 100 வாசிப்புக் கேள்விகளில் சுமார் 27 விடையற்றப் பொய்விடைப் பொறிகள் — பத்தியில் பதிலே இல்லை. நேர்மையான பதில் “தெரியவில்லை” மட்டுமே. <b>பொய் %</b> என்பது மாதிரி எத்தனை பொறிகளில் பதில் சொன்னது என்பது — <b>குறைவு சிறந்தது</b>.</p>
    <div class="board">
      <table>
        <thead><tr><th></th><th><span data-lang="en">Model</span><span class="ta-mini" data-lang="ta">மாதிரி</span></th><th class="num"><span data-lang="en">Bluff % (lower = honest)</span><span class="ta-mini" data-lang="ta">பொய் % (குறைவு = நேர்மை)</span></th><th class="barcell"><span data-lang="en">Bar</span><span class="ta-mini" data-lang="ta">பட்டை</span></th><th class="num"><span data-lang="en">Traps</span><span class="ta-mini" data-lang="ta">பொறிகள்</span></th></tr></thead>
        <tbody>
          <!--ROWS:BLUFF-->
          <!--/ROWS:BLUFF-->
        </tbody>
      </table>
    </div>
    <p class="note" data-lang="en">Measured on the same answer sheets as Paper 2 — no extra API calls. Abstention phrases matched in Tamil and English (தெரியவில்லை / no answer / not mentioned …). F1 on answerable questions is unaffected: traps carry empty golds, so they only ever subtract.</p>
  </div>
</section>

<section id="xnli">
  <div class="wrap">
    <div class="qnum"><span class="qta">வினா 5</span><h2><span data-lang="en">Scoreboard · Paper 3: IndicXNLI</span><span class="ta-title" data-lang="ta">மதிப்பெண் அட்டணை · தாள் 3: IndicXNLI</span></h2><span class="mark">[200 marks]</span></div>
    <p data-lang="en">A three-way reading-logic exam. Given a <b>premise</b> and a <b>hypothesis</b>, the model must pick: <b>A</b> definitely true, <b>B</b> definitely false, or <b>C</b> cannot be decided — entirely in Tamil. A blind guess scores ~33%.</p>
    <p class="ta-copy" data-lang="ta">முப்பிரிவு வாசிப்பு-தர்க்கத் தேர்வு. ஒரு <b>முன்னுரை</b>யும் ஒரு <b>கூற்று</b>ம் கொடுக்கப்பட்டால், கூற்று <b>A</b> கண்டிப்பாக உண்மை, <b>B</b> கண்டிப்பாக தவறு, அல்லது <b>C</b> முடிவு செய்ய முடியாது என முடிவு செய்ய வேண்டும் — முழுவதும் தமிழில். சூதாட்டமாகவே கணித்தால் ~33% கிடைக்கும்.</p>
    <div class="board">
      <table>
        <thead><tr><th></th><th><span data-lang="en">Model</span><span class="ta-mini" data-lang="ta">மாதிரி</span></th><th class="num"><span data-lang="en">Accuracy</span><span class="ta-mini" data-lang="ta">சரியான விடை %</span></th><th class="barcell"><span data-lang="en">Bar</span><span class="ta-mini" data-lang="ta">பட்டை</span></th><th class="num"><span data-lang="en">95% CI</span><span class="ta-mini" data-lang="ta">95% நம்பிக்கை வரம்பு</span></th></tr></thead>
        <tbody>
          <!--ROWS:XNLI-->
          <!--/ROWS:XNLI-->
        </tbody>
      </table>
    </div>
    <p class="note" data-lang="en">0-shot, temperature 0, single-letter parse; n=200 per model from the 5,010-question Tamil test split (seed 42). Data: <a href="https://huggingface.co/datasets/AdaMLLab/indicxnli_repaired">IndicXNLI (repaired)</a> — XNLI premises/hypotheses translated to 11 Indic languages, AI4Bharat lineage. Machine-readable: <span style="font-family:var(--mono)">results/summary.json</span>.</p>
  </div>
</section>

<section id="rubric">"""
html = html.replace('<section id="rubric">', NEW, 1)

# 5. findings: mark update + bluff/xnli bullets
html = html.replace("[4 × 2 = 8 marks]", "[5 × 2 = 10 marks]")
old_li = '      <li data-lang="en"><b>Model variety matters.</b>'
new_bullets = """      <li data-lang="en"><b>Everyone bluffs.</b> On unanswerable questions the honest move is to say “I don't know” — yet every model answered most traps anyway. Gemma 4 26B bluff least (32%), Gemini 3.8 around 59%; at the top of the leaderboard, bluffing is still the default behaviour.</li>
      <li class="ta-copy" data-lang="ta"><b>அனைவரும் பொய்விடை சொல்கிறார்கள்.</b> விடையற்ற கேள்விகளில் நேர்மையான பதில் “தெரியவில்லை” என்பதுதான் — ஆனால் அனைத்து மாதிரிகளும் பெரும்பாலான பொறிகளில் பதில் சொல்கின்றன. Gemma 4 26B குறைவாகப் பொய் சொல்கிறது (32%); Gemini 3.8 சுமார் 59%.</li>
      <li data-lang="en"><b>Model variety matters.</b>"""
html = html.replace(old_li, new_bullets, 1)

# 6. chart caption
html = html.replace(
    'alt="Tamil Bench comparison chart: 9 models across MILU accuracy, IndicQA exact match and F1"',
    'alt="Tamil Bench comparison chart: 9 models across MILU accuracy, IndicQA exact match, F1 and IndicXNLI accuracy"')
html = html.replace("all three metrics side by side", "all metrics side by side")

idx.write_text(html)
print("index.html: sections wired, renumbered, claims fixed")

# ---- build_site.py: dedupe bluff_scores ----
bs = ROOT / "build_site.py"
s = bs.read_text()
m1 = s.find("def bluff_scores(data):")
m2 = s.find("def bluff_scores(data):", m1 + 10)
if m2 != -1:
    # remove the FIRST (data-file based) definition; keep the golds-based one + ABSTAIN_MARKERS
    first = s[m1:s.find("ABSTAIN_MARKERS", m1)]
    s = s.replace(first, "", 1)
    bs.write_text(s)
    print("build_site.py: duplicate bluff_scores removed")
else:
    print("build_site.py: no duplicate found")
