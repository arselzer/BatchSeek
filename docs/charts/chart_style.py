"""Shared look for the post's charts (standard library only): tokens, type scale, SVG helpers.

Every chart is a static SVG with a transparent background, so it sits on the page's own surface. Colours are CSS
variables: light values by default, dark values under prefers-color-scheme: dark. PaperMod sets color-scheme on
:root for its theme toggle, and browsers resolve an <img>-embedded SVG's prefers-color-scheme from the embedding
element's color-scheme, so the toggle reaches the charts. Inlined, [data-theme=dark] / body.dark also switch them.

Series colours: the dataviz reference palette's first three categorical slots, validated (adjacent pairs, CVD and
normal vision) against PaperMod's surfaces #ffffff and #1d1e20. Aqua is below 3:1 on white, so it always carries a
direct label or a key entry. A neutral grey is the de-emphasis colour (the plain engine / earlier configuration).
"""
import math
from xml.sax.saxutils import escape

TOK = {   # name: (light, dark)
    "ink": ("#1e1e1e", "#e4e4e5"), "ink2": ("#4a4a48", "#c3c3c4"), "muted": ("#76767a", "#9b9c9d"),
    "grid": ("#ececea", "#2e2f32"), "axis": ("#c9c8c2", "#4a4b4f"), "surf": ("#ffffff", "#1d1e20"),
    "s1": ("#2a78d6", "#3987e5"), "s2": ("#eb6834", "#d95926"), "s3": ("#1baf7a", "#199e70"),
    "neutral": ("#a3a29c", "#77767c"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
# type scale (px): chart title, panel title, label, tick / note
TITLE, HEAD, LABEL, SMALL = 15, 12.5, 12, 11
W = 760          # every chart's viewBox width; blog column is ~720 px, so 1 unit ~ 1 px


def style():
    light = ";".join(f"--{k}:{v[0]}" for k, v in TOK.items())
    dark = ";".join(f"--{k}:{v[1]}" for k, v in TOK.items())
    return ("<style>"
            f"svg.ch{{{light};font-family:{FONT};font-variant-numeric:tabular-nums}}"
            f"@media (prefers-color-scheme: dark){{svg.ch{{{dark}}}}}"
            f"[data-theme=dark] svg.ch,body.dark svg.ch{{{dark}}}"
            f"[data-theme=light] svg.ch{{{light}}}"
            "svg.ch text{dominant-baseline:auto}"
            "</style>")


def doc(w, h, body, title, desc):
    return (f"<svg xmlns='http://www.w3.org/2000/svg' class='ch' viewBox='0 0 {w} {h:.0f}' width='100%' role='img' "
            f"aria-labelledby='t d'><title id='t'>{escape(title)}</title><desc id='d'>{escape(desc)}</desc>"
            f"{style()}{body}</svg>\n")


def text(x, y, s, size=LABEL, fill="ink2", anchor="start", weight="normal", extra=""):
    return (f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' fill='var(--{fill})' text-anchor='{anchor}' "
            f"font-weight='{weight}'{extra}>{escape(s)}</text>")


def text_w(s, size=LABEL):
    """Rough rendered width of s in px (system sans, tabular digits)."""
    return sum(0.56 if c.isdigit() else 0.28 if c in " .,:;il|!'()t" else 0.66 if c.isupper() or c in "mw" else 0.5
               for c in s) * size


def hline(x1, x2, y, col="grid", width=1, dash=None, opacity=None):
    d = f" stroke-dasharray='{dash}'" if dash else ""
    o = f" stroke-opacity='{opacity}'" if opacity else ""
    return f"<line x1='{x1:.1f}' x2='{x2:.1f}' y1='{y:.1f}' y2='{y:.1f}' stroke='var(--{col})' stroke-width='{width}'{d}{o}/>"


def path_d(pts):
    return " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(pts))


def line(pts, col, width=2, opacity=None, dash=None):
    o = f" stroke-opacity='{opacity}'" if opacity else ""
    d = f" stroke-dasharray='{dash}'" if dash else ""
    return (f"<path d='{path_d(pts)}' fill='none' stroke='var(--{col})' stroke-width='{width}' stroke-linejoin='round' "
            f"stroke-linecap='round'{o}{d}/>")


def area(upper, lower, col, opacity=0.12):
    return f"<path d='{path_d(upper + lower[::-1])} Z' fill='var(--{col})' fill-opacity='{opacity}' stroke='none'/>"


def dot(x, y, col, r=4.5):
    return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='var(--{col})' stroke='var(--surf)' stroke-width='2'/>"


def key(items, x, y, anchor="start"):
    """One-line key: (label, colour, kind) with kind line / box / dash. Returns (svg, width)."""
    parts, cx = [], 0.0
    for lab, col, kind in items:
        if kind == "box":
            parts.append((cx, f"<rect x='{{x}}' y='{y-8.5:.1f}' width='10' height='10' rx='2' fill='var(--{col})' fill-opacity='0.55'/>"))
            sw = 10
        elif kind == "dash":
            parts.append((cx, f"<line x1='{{x}}' x2='{{x2}}' y1='{y-4:.1f}' y2='{y-4:.1f}' stroke='var(--{col})' stroke-width='1.5' stroke-dasharray='4 3'/>"))
            sw = 16
        else:
            parts.append((cx, f"<line x1='{{x}}' x2='{{x2}}' y1='{y-4:.1f}' y2='{y-4:.1f}' stroke='var(--{col})' stroke-width='2' stroke-linecap='round'/>"))
            sw = 16
        parts.append((cx + sw + 5, ("T", lab)))
        cx += sw + 5 + text_w(lab, SMALL) + 14
    width = cx - 14
    x0 = x - width if anchor == "end" else x
    out = []
    for ox, p in parts:
        if isinstance(p, tuple):
            out.append(text(x0 + ox, y, p[1], SMALL, "ink2"))
        else:
            out.append(p.replace("{x}", f"{x0+ox:.1f}").replace("{x2}", f"{x0+ox+16:.1f}"))
    return "".join(out), width


def nice_step(span, target=4):
    raw = span / target
    p = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if m * p >= raw:
            return m * p
    return 10 * p


def nice_max(v, target=3):
    """Axis top = v itself (the caller pads it); gridlines at a nice step below it, so the plot keeps few lines."""
    return v, nice_step(v, target)
