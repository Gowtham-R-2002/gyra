"""Render the H17 audit summary as a PNG for the README and model card.

Run: python assets/make_h17_chart.py
Requires: matplotlib
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "h17-audit-summary.json").read_text())
rows = data["rows"]

bg = "#0c1119"
white = "#f4f2ed"
muted = "#aeb8c9"
grid = "#303b4b"
orange = "#d58a15"
violet = "#8467f2"

fig, ax = plt.subplots(figsize=(15, 9), dpi=120)
fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(.045, .935, "GYRA  /  H17 AUDIT", color=violet, fontsize=12, weight="bold", va="center")
ax.text(.045, .87, "Gyra vs Laya on coding-agent decisions", color=white,
        fontsize=30, weight="bold", va="center")
ax.text(.045, .815, "Share of cases handled correctly in a frozen 365-case audit",
        color=muted, fontsize=15, va="center")

ax.add_patch(FancyBboxPatch((.69, .918), .012, .012, boxstyle="round,pad=0.002", color=orange))
ax.text(.711, .924, "Laya English", color=muted, fontsize=13, va="center")
ax.add_patch(FancyBboxPatch((.82, .918), .012, .012, boxstyle="round,pad=0.002", color=violet))
ax.text(.842, .924, "Gyra v0.2", color=white, fontsize=13, va="center")

x0, xw = .37, .54
for pct in (0, 25, 50, 75, 100):
    x = x0 + xw * pct / 100
    ax.plot((x, x), (.17, .765), color=grid, linewidth=1, zorder=0)
    ax.text(x, .147, str(pct), color=muted, fontsize=11, ha="center", va="center")

ys = [.69, .535, .38, .225]
for row, y in zip(rows, ys):
    ax.text(.045, y + .022, row["label"], color=white, fontsize=17,
            weight="medium", va="center")
    ax.text(.045, y - .015, f'{row["total"]} cases', color=muted, fontsize=12, va="center")
    for n, dy, color in ((row["laya"], .019, orange), (row["gyra"], -.019, violet)):
        length = xw * n / row["total"]
        if n:
            ax.add_patch(FancyBboxPatch((x0, y + dy - .012), length, .024,
                                        boxstyle="round,pad=0,rounding_size=0.006",
                                        linewidth=0, facecolor=color))
        text_x = min(x0 + length + .009, .944)
        ax.text(text_x, y + dy, f'{n}/{row["total"]}', color=white,
                fontsize=12, weight="bold", va="center", ha="left")

ax.text(.045, .085,
        "Standalone models at hook thresholds: 0.5 for destructive/secret, 0.7 for hang.",
        color=muted, fontsize=11, va="center")
ax.text(.045, .058,
        "The deployed Gyra hook also uses rules: 17/17 direct file/secret cases, 0 rule false alarms in a separate fixed audit.",
        color=muted, fontsize=11, va="center")
ax.text(.045, .025, "Source: assets/h17-audit-summary.json  ·  Results are specific to these audit sets.",
        color="#8290a4", fontsize=10, va="center")

out = HERE / "gyra-v02-vs-laya.png"
fig.savefig(out, dpi=120, facecolor=bg, bbox_inches=None, pad_inches=0)
plt.close(fig)
print(out)
