#!/usr/bin/env python3
"""Generates countdown.svg: an animated digital countdown for a GitHub README.
Seconds tick live (SMIL loop); minutes/hours/days are pre-computed for the next
WINDOW seconds. A scheduled GitHub Action regenerates the file to stay in sync."""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import os

TZ = ZoneInfo(os.getenv("COUNTDOWN_TZ", "Asia/Kolkata"))
TARGET = datetime.fromisoformat(os.getenv("COUNTDOWN_TARGET", "2026-12-01T00:00:00")).replace(tzinfo=TZ)
TITLE = os.getenv("COUNTDOWN_TITLE", "WINTER COUNTDOWN  ·  1 DEC")
WINDOW = 1800  # seconds of pre-computed animation (30 min safety margin)

now = datetime.now(TZ)
total = max(int((TARGET - now).total_seconds()), 0)

def split(rem):
    rem = max(rem, 0)
    return rem // 86400, rem % 86400 // 3600, rem % 3600 // 60, rem % 60

def segments(idx):
    """Consecutive runs of equal values for unit idx over the window."""
    segs, prev, start = [], None, 0
    for t in range(WINDOW):
        v = split(total - t)[idx]
        if v != prev:
            if prev is not None:
                segs.append((prev, start, t))
            prev, start = v, t
    segs.append((prev, start, WINDOW))
    return segs

BOX_W, GAP, X0, Y0, BOX_H = 120, 20, 30, 55, 90
CX = lambda i: X0 + i * (BOX_W + GAP) + BOX_W / 2
LABELS = ["DAYS", "HOURS", "MINUTES", "SECONDS"]

def fmt(n): return f"{n:02d}"

def one_shot(text, cx, start, end, last):
    """Visible during [start,end) of the WINDOW timeline; holds if last."""
    T = WINDOW
    if start == 0 and last:
        vals, keys = "1", None
    elif start == 0:
        vals, keys = "1;0", f"0;{end/T:.6f}"
    elif last:
        vals, keys = "0;1", f"0;{start/T:.6f}"
    else:
        vals, keys = "0;1;0", f"0;{start/T:.6f};{end/T:.6f}"
    anim = "" if keys is None else (
        f'<animate attributeName="opacity" values="{vals}" keyTimes="{keys}" '
        f'dur="{T}s" calcMode="discrete" fill="freeze" begin="0s"/>')
    init = "1" if start == 0 else "0"
    return f'<text class="d" x="{cx}" y="{Y0+62}" opacity="{init}">{text}{anim}</text>'

parts = []
for i in range(3):  # days, hours, minutes
    segs = segments(i)
    for n, (v, s, e) in enumerate(segs):
        parts.append(one_shot(fmt(v), CX(i), s, e, n == len(segs) - 1))

# seconds: 60 elements looping every 60s, phase-aligned to generation time
s0 = total % 60
for v in range(60):
    a = (s0 - v) % 60
    if a == 0:   vals, keys = "1;0", f"0;{1/60:.6f}"
    elif a == 59: vals, keys = "0;1", f"0;{59/60:.6f}"
    else:        vals, keys = "0;1;0", f"0;{a/60:.6f};{(a+1)/60:.6f}"
    init = "1" if a == 0 else "0"
    parts.append(
        f'<text class="d" x="{CX(3)}" y="{Y0+62}" opacity="{init}">{fmt(v)}'
        f'<animate attributeName="opacity" values="{vals}" keyTimes="{keys}" '
        f'dur="60s" calcMode="discrete" repeatCount="indefinite"/></text>')

boxes = "".join(
    f'<rect x="{X0+i*(BOX_W+GAP)}" y="{Y0}" width="{BOX_W}" height="{BOX_H}" rx="14" class="box"/>'
    f'<text class="lbl" x="{CX(i)}" y="{Y0+BOX_H+22}">{LABELS[i]}</text>'
    for i in range(4))

colons = "".join(
    f'<text class="col" x="{X0+(i+1)*(BOX_W+GAP)-GAP/2}" y="{Y0+60}">:'
    f'<animate attributeName="opacity" values="1;0.15;1" dur="1s" repeatCount="indefinite"/></text>'
    for i in range(3))


import random
random.seed(7)
def snowflakes(n=34):
    out = []
    for i in range(n):
        x = random.randint(5, 595)
        sc = random.choice([0.5, 0.7, 0.9, 1.2, 1.5])
        fall = random.uniform(6, 14)
        sway = random.uniform(3, 6)
        amp = random.randint(8, 18)
        delay = -random.uniform(0, fall)
        op = round(0.35 + sc * 0.4, 2)
        shape = ('<path d="M0,-4V4M-3.5,-2L3.5,2M-3.5,2L3.5,-2" stroke="#fff" stroke-width="1" stroke-linecap="round" fill="none"/>'
                 if i % 3 == 0 else '<circle r="2.2" fill="#fff"/>')
        out.append(
            f'<g transform="translate({x} 0)" opacity="{op}">'
            f'<g><animateTransform attributeName="transform" type="translate" values="0 -15;0 235" dur="{fall:.1f}s" begin="{delay:.1f}s" repeatCount="indefinite"/>'
            f'<g><animateTransform attributeName="transform" type="translate" values="0 0;{amp} 0;0 0;-{amp} 0;0 0" dur="{sway:.1f}s" repeatCount="indefinite"/>'
            f'<g transform="scale({sc})">{shape}</g></g></g></g>')
    return "".join(out)
SNOW = snowflakes()
PILE = ('<path d="M0,208 Q40,198 90,206 T200,204 T320,207 T440,203 T540,206 T600,202 V220 H0Z" fill="#e8f6ff" opacity="0.92"/>'
        '<path d="M0,214 Q60,208 130,213 T280,212 T430,214 T600,211 V220 H0Z" fill="#cfeaff"/>')

generated = now.strftime("%d %b %Y %H:%M IST")
svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 220" width="600" height="220" role="img" aria-label="Winter countdown to 1 Dec">
<defs>
<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0a1628"/><stop offset="1" stop-color="#17405f"/></linearGradient>
<linearGradient id="bx" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#24507a"/><stop offset="1" stop-color="#102a47"/></linearGradient>
<filter id="glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
</defs>
<style>
.box{{fill:url(#bx);fill-opacity:.78;stroke:#9fe0ff;stroke-width:1.5}}
.d{{font:700 52px "SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;fill:#e6f8ff;text-anchor:middle;filter:url(#glow)}}
.col{{font:700 44px monospace;fill:#7fd4ff;text-anchor:middle}}
.lbl{{font:600 12px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;fill:#a8d8f0;text-anchor:middle;letter-spacing:3px}}
.ttl{{font:700 15px -apple-system,"Segoe UI",Helvetica,Arial,sans-serif;fill:#ffffff;text-anchor:middle;letter-spacing:4px}}
.sub{{font:10px monospace;fill:#6b8ba6;text-anchor:middle}}
</style>
<rect width="600" height="220" rx="18" fill="url(#bg)"/>
<text class="ttl" x="300" y="34">{TITLE}</text>
{SNOW}
{boxes}
{"".join(parts)}
{colons}
<text class="sub" x="300" y="193">synced {generated}</text>
{PILE}
</svg>
'''
out = os.getenv("COUNTDOWN_OUT", "countdown.svg")
open(out, "w", encoding="utf-8").write(svg)
print(f"Wrote {out} ({len(svg)//1024} KB) — remaining: {split(total)}")
