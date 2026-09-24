"""Render the README charts from assets/scores.json (measured numbers; nothing typed in by hand).

    python assets/make_charts.py   -> assets/hero-light.svg, assets/hero-dark.svg
"""
import json
from pathlib import Path
from xml.sax.saxutils import escape

A = Path(__file__).resolve().parent
TARGET = ["tool_choice", "tool_relevance", "cmd_intent", "cmd_destructive", "cmd_success", "ext_swe_rebench", "ext_swe_zero",
          "pi_spml", "pi_bipia", "pi_deepset_test"]
SHORT = {"tool_choice": "Which tool to call · BFCL", "tool_relevance": "Does any tool fit · BFCL",
         "cmd_intent": "Command matches the request · NL2Bash", "cmd_destructive": "Deletes / overwrites files · sandbox",
         "cmd_success": "Did it succeed · sandbox output", "ext_swe_rebench": "Did it succeed · SWE-rebench agent logs",
         "ext_swe_zero": "Did it succeed · SWE-Zero agent logs", "pi_spml": "Prompt injection · SPML",
         "pi_bipia": "Hidden injection · BIPIA", "pi_deepset_test": "Prompt injection · deepset"}
THEMES = {
    "light": {"bg": "#ffffff", "ink": "#17161f", "ink2": "#454456", "ink3": "#6b6a7d", "grid": "#e6e6ef", "gyra": "#6552e0", "laya": "#b86b00"},
    "dark": {"bg": "#0d1117", "ink": "#f1f0f7", "ink2": "#c6c5d6", "ink3": "#9c9bb0", "grid": "#2a2a36", "gyra": "#7965f0", "laya": "#cf7b00"},
}


def chart(rows, t):
    W, left, right, top, rowh = 920, 300, 70, 92, 46
    H = top + rowh * len(rows) + 46
    x0, x1 = left, W - right
    sx = lambda v: x0 + (x1 - x0) * v / 100                # full 0-100 % axis: no exaggerated gaps
    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">' % (W, H, W, H),
         '<rect width="100%%" height="100%%" rx="14" fill="%s"/>' % t["bg"],
         '<text x="28" y="40" font-size="21" font-weight="700" fill="%s">Gyra vs Laya on coding-agent decisions</text>' % t["ink"],
         '<text x="28" y="64" font-size="13.5" fill="%s">Accuracy %%, held-out tests with true answers. Same 421M architecture, same speed. Laya = the better of its two checkpoints per test.</text>' % t["ink3"]]
    for v in range(0, 101, 20):
        x = sx(v)
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="%s" stroke-width="1"/>' % (x, top - 8, x, H - 40, t["grid"]))
        o.append('<text x="%.1f" y="%d" font-size="11.5" fill="%s" text-anchor="middle">%d</text>' % (x, H - 22, t["ink3"], v))
    for i, r in enumerate(rows):
        y = top + i * rowh
        laya = max(r["laya-english"], r["laya-typed-decisions"])
        o.append('<text x="%d" y="%d" font-size="13.5" fill="%s" text-anchor="end">%s</text>' % (left - 14, y + 20, t["ink2"], escape(SHORT[r["suite"]])))
        for j, (v, c) in enumerate(((laya, t["laya"]), (r["ours-H8"], t["gyra"]))):
            yy = y + 4 + j * 15
            o.append('<rect x="%.1f" y="%d" width="%.1f" height="12" rx="4" fill="%s"/>' % (x0, yy, sx(v) - x0, c))
            o.append('<text x="%.1f" y="%d" font-size="11.5" font-weight="%d" fill="%s">%.1f</text>' % (sx(v) + 6, yy + 10, 700 if j else 500, t["ink"] if j else t["ink2"], v))
    lx = W - right - 190
    for j, (name, c) in enumerate((("Laya", t["laya"]), ("Gyra", t["gyra"]))):
        o.append('<rect x="%d" y="30" width="12" height="12" rx="3" fill="%s"/><text x="%d" y="40" font-size="13" fill="%s">%s</text>' % (lx + j * 90, c, lx + j * 90 + 18, t["ink2"], name))
    o.append("</svg>")
    return "\n".join(o)


def main():
    rows = {r["suite"]: r for r in json.load(open(A / "scores.json"))}
    rows = [rows[s] for s in TARGET]
    for name, t in THEMES.items():
        (A / ("hero-%s.svg" % name)).write_text(chart(rows, t))
    print("wrote hero-light.svg, hero-dark.svg")


if __name__ == "__main__":
    main()
