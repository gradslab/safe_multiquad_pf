r"""Layout quality gate for every figure.

Catches what makes a figure unreadable and is easy to miss at thumbnail size:

  * text drawn on top of plotted data (any axes' data, not just its own)
  * text overlapping other text, INCLUDING across neighbouring axes -- a y-label colliding with the
    tick labels of the panel to its left is the classic multi-column failure
  * tick labels colliding with each other
  * text or labels overrunning a neighbouring axes' plotting area
  * legends sitting on the curves they describe
  * axes overlapping each other

It inspects the live matplotlib figure, so it reasons about real artist bounding boxes rather than
pixels. Call check(fig, name) just before saving.
"""
import numpy as np
from matplotlib.transforms import Bbox


def _bb(artist, rend):
    try:
        b = artist.get_window_extent(renderer=rend)
        return b if (b.width > 1 and b.height > 1) else None
    except Exception:
        return None


def _frac(a, b):
    """Fraction of a's area covered by b."""
    x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
    y0, y1 = max(a.y0, b.y0), min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0) / max(a.width * a.height, 1e-9)


def _ink_under(box, ax, rend):
    """Number of actually-drawn vertices of `ax` falling inside `box` (display coords).

    Points outside the view are clipped by the axes and never reach the canvas, so they are
    excluded here; counting them once made a suptitle look as though it sat on a curve.
    """
    n = 0
    ab = ax.get_window_extent(renderer=rend)
    for d in ax.get_lines():
        if not d.get_visible():
            continue
        xy = d.get_xydata()
        if xy is None or len(xy) == 0:
            continue
        step = max(1, len(xy) // 3000)
        try:
            pts = ax.transData.transform(xy[::step])
        except Exception:
            continue
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            continue
        inside_axes = ((pts[:, 0] >= ab.x0 - 1) & (pts[:, 0] <= ab.x1 + 1) &
                       (pts[:, 1] >= ab.y0 - 1) & (pts[:, 1] <= ab.y1 + 1))
        n += int((inside_axes &
                  (pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1) &
                  (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1)).sum())
    return n


def _tick_in_view(ax, axis, lbl, rend):
    """matplotlib keeps tick labels for locations outside the current view; they are never drawn.
    Judging layout on them reports collisions that do not exist in the saved file."""
    try:
        b = lbl.get_window_extent(renderer=rend)
    except Exception:
        return False
    ab = ax.get_window_extent(renderer=rend)
    if axis is ax.xaxis:
        c = 0.5 * (b.x0 + b.x1)
        return ab.x0 - 3 <= c <= ab.x1 + 3
    c = 0.5 * (b.y0 + b.y1)
    return ab.y0 - 3 <= c <= ab.y1 + 3


def _texts_of(ax, rend):
    """Every visible text artist an axes owns, with a role tag."""
    out = []
    for t in ax.texts:
        if t.get_visible() and t.get_text().strip():
            out.append(("annotation", t))
    pairs = [("title", ax.title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]
    if hasattr(ax, "zaxis"):
        pairs.append(("zlabel", ax.zaxis.label))
    for role, art in pairs:
        if art.get_text().strip():
            out.append((role, art))
    if getattr(ax, "name", "") != "3d":
        for axis in (ax.xaxis, ax.yaxis):
            for lbl in axis.get_ticklabels():
                if lbl.get_visible() and lbl.get_text().strip() and _tick_in_view(ax, axis, lbl, rend):
                    out.append(("tick", lbl))
    return [(r, t, _bb(t, rend)) for r, t in out if _bb(t, rend) is not None]


def check(fig, name="figure", verbose=True):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    axes = [a for a in fig.get_axes() if _bb(a, rend) is not None]
    bad = []

    def label(i):
        if i < 0:
            return "figure"
        ttl = axes[i].get_title(loc="left") or axes[i].get_title()
        return ttl if ttl else f"ax{i}"

    all_text = []                       # (axes_index, role, artist, bbox)
    for i, ax in enumerate(axes):
        for role, t, tb in _texts_of(ax, rend):
            all_text.append((i, role, t, tb))
    # FIGURE-level artists belong to no axes and were invisible to an axes-only sweep: a fig.legend
    # or fig.suptitle printed straight over the panel titles once passed as "clean".
    for leg in getattr(fig, "legends", []):
        for t in leg.get_texts():
            tb = _bb(t, rend)
            if tb is not None and t.get_text().strip():
                all_text.append((-1, "figure legend", t, tb))
        lb = _bb(leg, rend)
        if lb is not None:
            for j, ax in enumerate(axes):
                if _ink_under(lb, ax, rend) > 25:
                    bad.append(f"{name}/{label(j)}: figure legend covers plotted data")
                    break
    for t in fig.texts:
        tb = _bb(t, rend)
        if tb is not None and t.get_text().strip():
            all_text.append((-1, "figure text", t, tb))

    # 1. text over plotted data, in ANY axes
    for i, role, t, tb in all_text:
        if role in ("xlabel", "ylabel", "tick"):
            continue                    # these live outside the data area by construction
        if i < 0 and role == "figure legend":
            continue                    # handled as a block above
        for j, ax in enumerate(axes):
            if _ink_under(tb, ax, rend) > 20:
                bad.append(f"{name}/{label(i)}: {role} {t.get_text()[:26]!r} sits on plotted data")
                break

    # 2. text vs text, including across neighbouring axes
    for a in range(len(all_text)):
        ia, ra, ta, ba = all_text[a]
        for b in range(a + 1, len(all_text)):
            ib, rb, tb_, bb = all_text[b]
            f = max(_frac(ba, bb), _frac(bb, ba))
            if f > 0.03:
                where = f"{label(ia)}" if ia == ib else f"{label(ia)} vs {label(ib)}"
                bad.append(f"{name}/{where}: {ra} {ta.get_text()[:18]!r} overlaps "
                           f"{rb} {tb_.get_text()[:18]!r}")

    # 3. a label overrunning a NEIGHBOURING axes' plotting area
    for i, role, t, tb in all_text:
        if role not in ("ylabel", "xlabel", "title", "tick"):
            continue
        for j, ax in enumerate(axes):
            if j == i:
                continue
            if _frac(tb, _bb(ax, rend)) > 0.35:
                bad.append(f"{name}/{label(i)}: {role} {t.get_text()[:22]!r} runs into {label(j)}")
                break

    # 4. legend over the curves it describes, or over a neighbouring panel
    for i, ax in enumerate(axes):
        leg = ax.get_legend()
        lb = _bb(leg, rend) if leg is not None else None
        if lb is None:
            continue
        if _ink_under(lb, ax, rend) > 25:
            bad.append(f"{name}/{label(i)}: legend covers plotted data")
        for j, other in enumerate(axes):
            if j == i:
                continue
            ob = _bb(other, rend)
            if ob is not None and _frac(lb, ob) > 0.06:
                bad.append(f"{name}/{label(i)}: legend runs into {label(j)}")
                break

    # 5. axes overlapping each other
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            if _frac(_bb(axes[i], rend), _bb(axes[j], rend)) > 0.02:
                bad.append(f"{name}: {label(i)} and {label(j)} overlap")

    bad = sorted(set(bad))
    if verbose:
        if bad:
            print(f"    QC {name}: {len(bad)} issue(s)")
            for m in bad[:14]:
                print("       -", m)
            if len(bad) > 14:
                print(f"       ... and {len(bad)-14} more")
        else:
            print(f"    QC {name}: clean")
    return bad
