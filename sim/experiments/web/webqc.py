r"""Publication gate for the figures and animations that go on the project page.

A figure is drawn much larger than it is shown, so a panel that looks fine in an editor can be
unreadable in a browser column. Nothing is written to the site unless it passes.

Figures (checked live, before saving): the layout checks in experiments/qc_figures.py, plus text
that overruns the canvas, text too small once scaled to the display width, data clipped by the view
limits, empty axes, and axes missing labels or ticks.

Media (checked on disk, after writing): decodes, meets the pixel floor for its slot on the page, is
not blank, keeps a margin from the canvas edge, and for a GIF has enough frames that actually differ.

    issues = webqc.gate_figure(fig, "W_feasibility_map", display_px=1180)
    webqc.save_gated(fig, outdir, "W_feasibility_map")      # raises on failure
    webqc.gate_media("docs/anim/circles_proposed.gif", slot="gif")
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
import qc_figures as base                                     # the layout checks already in the repo

MIN_PX = 11.0          # smallest legible rendered text height, in CSS pixels, at page display width
MIN_PX_TICK = 10.0     # ticks may be a shade smaller than titles and axis labels
EDGE_PAD = 1.0         # a text artist must sit this many pixels inside the canvas


class GateFailure(RuntimeError):
    pass


def _bb(artist, rend):
    try:
        b = artist.get_window_extent(renderer=rend)
        return b if (b.width > 0.5 and b.height > 0.5) else None
    except Exception:
        return None


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


def _texts(fig, rend):
    """(owner_label, role, artist, bbox) for every visible text in the figure, axes and figure level."""
    out = []
    for i, ax in enumerate(fig.get_axes()):
        who = ax.get_title(loc="left") or ax.get_title() or f"ax{i}"
        pairs = [("title", ax.title), ("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label)]
        if hasattr(ax, "zaxis"):          # 3d: the z label was invisible to every check below
            pairs.append(("zlabel", ax.zaxis.label))
        for role, art in pairs:
            if art.get_text().strip():
                out.append((who, role, art))
        if getattr(ax, "name", "") != "3d":     # see the note in qc_figures._texts_of
            for axis in (ax.xaxis, ax.yaxis):
                for lbl in axis.get_ticklabels():
                    if lbl.get_visible() and lbl.get_text().strip() and _tick_in_view(ax, axis, lbl, rend):
                        out.append((who, "tick", lbl))
        for t in ax.texts:
            if t.get_visible() and t.get_text().strip():
                out.append((who, "annotation", t))
        leg = ax.get_legend()
        if leg is not None:
            for t in leg.get_texts():
                out.append((who, "legend", t))
    for leg in getattr(fig, "legends", []):
        for t in leg.get_texts():
            out.append(("figure", "legend", t))
    for t in fig.texts:
        if t.get_visible() and t.get_text().strip():
            out.append(("figure", "figure text", t))
    return [(w, r, t, _bb(t, rend)) for w, r, t in out if _bb(t, rend) is not None]


def _data_outside_view(ax):
    """Fraction of finite plotted points that fall outside the axes view, per line."""
    worst, who = 0.0, ""
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    x0, x1 = min(x0, x1), max(x0, x1); y0, y1 = min(y0, y1), max(y0, y1)
    xs = 1e-9 + 0.002 * (x1 - x0); ys = 1e-9 + 0.002 * (y1 - y0)
    for ln in ax.get_lines():
        if not ln.get_visible() or ln.get_linestyle() == "None" and ln.get_marker() in ("", "None"):
            continue
        if ln.get_transform() is not ax.transData:
            continue          # axhline / axvline: blended transform, its xydata is not data coords
        xy = ln.get_xydata()
        if xy is None or len(xy) == 0:
            continue
        xy = np.asarray(xy, float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy) == 0:
            continue
        inside = ((xy[:, 0] >= x0 - xs) & (xy[:, 0] <= x1 + xs) &
                  (xy[:, 1] >= y0 - ys) & (xy[:, 1] <= y1 + ys))
        frac = 1.0 - inside.mean()
        if frac > worst:
            worst, who = frac, (ln.get_label() or "line")
    return worst, who


def gate_figure(fig, name="figure", display_px=1100, verbose=True, allow_zoom=(), require_labels=True,
                tight=True, min_px=None, min_px_tick=None):
    """Run every check against a live figure. Returns a list of issue strings (empty means pass).

    display_px: the CSS width, in pixels, at which the page will show this figure. Text legibility is
    judged after scaling the authored figure down to that width, which is what a reader actually sees.
    allow_zoom: names of axes titles whose view deliberately crops data (a zoom inset, a clipped axis).
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    issues = list(base.check(fig, name, verbose=False))

    fig_px = fig.get_size_inches()[0] * fig.dpi
    scale = display_px / max(fig_px, 1.0)                    # authored pixels -> displayed pixels
    W, H = fig_px, fig.get_size_inches()[1] * fig.dpi

    # 6. nothing cut off at the canvas edge. A tight save grows the canvas to the union of all
    #    artists, so an overhanging label is preserved there and only a gross overhang is a problem;
    #    the saved file is checked again pixel-side by gate_media().
    slop = 0.06 * W if tight else EDGE_PAD
    for who, role, t, b in _texts(fig, rend):
        if b.x0 < -slop or b.y0 < -slop or b.x1 > W + slop or b.y1 > H + slop:
            issues.append(f"{name}/{who}: {role} {t.get_text()[:24]!r} is cut off by the canvas edge")

    # 7. legible once the page scales it down
    for who, role, t, b in _texts(fig, rend):
        px = t.get_fontsize() * fig.dpi / 72.0 * scale
        floor = (min_px_tick or MIN_PX_TICK) if role == "tick" else (min_px or MIN_PX)
        if px < floor - 1e-6:
            issues.append(f"{name}/{who}: {role} {t.get_text()[:20]!r} renders at {px:.1f}px on the "
                          f"page, below the {floor:.0f}px floor")

    # 7b. a three-dimensional panel's axis furniture landing in the panel next to it. Its tick
    #     labels are skipped by the layout checks because their boxes are projected, but their
    #     positions still say plainly whether they have wandered into a neighbour.
    boxes = [(i, ax, _bb(ax, rend)) for i, ax in enumerate(fig.get_axes())]
    for i, ax in enumerate(fig.get_axes()):
        if getattr(ax, "name", "") != "3d":
            continue
        arts = []
        for a in (ax.xaxis, ax.yaxis, ax.zaxis):
            lo, hi = sorted(a.get_view_interval())
            locs = a.get_ticklocs()
            for loc, l in zip(locs, a.get_ticklabels()):
                # matplotlib keeps labels for locations outside the view; their positions are stale
                if lo - 1e-9 <= loc <= hi + 1e-9 and l.get_visible() and l.get_text().strip():
                    arts.append(("tick", l))
        arts += [(r, a) for r, a in (("xlabel", ax.xaxis.label), ("ylabel", ax.yaxis.label),
                                     ("zlabel", ax.zaxis.label)) if a.get_text().strip()]
        for role, art in arts:
            b = _bb(art, rend)
            if b is None:
                continue
            for j, other, ob in boxes:
                if j == i or ob is None:
                    continue
                if base._frac(b, ob) > 0.25:
                    issues.append(f"{name}/ax{i}: 3d {role} {art.get_text()[:16]!r} sits inside ax{j}")
                    break

    # 8..11 per-axes checks
    for i, ax in enumerate(fig.get_axes()):
        who = ax.get_title(loc="left") or ax.get_title() or f"ax{i}"
        if getattr(ax, "_qc_colorbar", False):
            continue
        n_art = len(ax.get_lines()) + len(ax.collections) + len(ax.patches) + len(ax.images)
        if n_art == 0 and not ax.get_children():
            issues.append(f"{name}/{who}: axes is empty")
            continue
        if getattr(ax, "name", "") == "3d":
            continue        # projected data: xydata is not in the axes' data frame
        if who in allow_zoom or getattr(ax, "_qc_zoom", False):
            pass
        else:
            frac, ln = _data_outside_view(ax)
            if frac > 0.02:
                issues.append(f"{name}/{who}: {frac*100:.0f}% of '{ln}' falls outside the view limits "
                              f"and is clipped")
        if require_labels and not getattr(ax, "_qc_shared_x", False) and not ax.get_xlabel().strip():
            issues.append(f"{name}/{who}: no x label")
        if require_labels and not getattr(ax, "_qc_shared_y", False) and not ax.get_ylabel().strip():
            issues.append(f"{name}/{who}: no y label")
        if getattr(ax, "name", "") == "3d":
            continue
        for axis, tag in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            vis = [l for l in axis.get_ticklabels()
                   if l.get_visible() and l.get_text().strip() and _tick_in_view(ax, axis, l, rend)]
            if len(vis) < 2 and not getattr(ax, f"_qc_no_{tag}ticks", False):
                issues.append(f"{name}/{who}: fewer than two {tag} tick labels")

    issues = sorted(set(issues))
    if verbose:
        if issues:
            print(f"    GATE {name}: {len(issues)} issue(s)")
            for m in issues:
                print("        -", m)
        else:
            print(f"    GATE {name}: clean")
    return issues


def save_gated(fig, outdir, name, display_px=1100, formats=("png", "pdf"), dpi=200, strict=True,
               **kw):
    """Gate, then write. Nothing reaches the site directory unless the gate is clean."""
    issues = gate_figure(fig, name, display_px=display_px, tight=True, **kw)
    if issues and strict:
        raise GateFailure(f"{name} failed the figure gate:\n  " + "\n  ".join(issues))
    os.makedirs(outdir, exist_ok=True)
    for ext in formats:
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02,
                    **({"dpi": dpi} if ext == "png" else {}))
    return issues


def make_room(ax, tries=10, grow=0.14, log=False):
    """Grow the vertical view until the legend no longer sits on any curve.

    Placing a legend by eye and hoping is what produces the one occluded panel nobody notices, so the
    space is measured against the drawn data and opened up until it is genuinely clear.
    """
    fig = ax.figure
    leg = ax.get_legend()
    if leg is None:
        return
    for _ in range(tries):
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        lb = _bb(leg, rend)
        if lb is None or base._ink_under(lb, ax, rend) <= 20:
            return
        lo, hi = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(lo, hi * (1.0 + 3 * grow))
        else:
            ax.set_ylim(lo, hi + grow * (hi - lo))


def gate_print(fig, name, printed_in, min_pt=6.0, **kw):
    """Gate a figure that goes into the manuscript rather than the page.

    LaTeX scales the file to `printed_in`, so a label's size on paper is its size on this canvas
    times that ratio. Everything is judged after that scaling.
    """
    px = printed_in * 96.0
    return gate_figure(fig, name, display_px=px, min_px=min_pt * 96.0 / 72.0,
                       min_px_tick=(min_pt - 0.5) * 96.0 / 72.0, **kw)


# ----------------------------------------------------------------------------- media on disk

SLOT_MIN = {"hero": (900, 380), "wide": (900, 340), "half": (620, 340), "third": (460, 300),
            "gif": (480, 300)}


def _content_bbox(arr, bg_tol=6):
    """Bounding box of pixels that differ from the corner background colour."""
    a = arr.astype(np.int16)
    if a.ndim == 3:
        a = a[..., :3]
        bg = np.median(np.stack([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]]), axis=0)
        d = np.abs(a - bg).max(axis=2)
    else:
        bg = np.median([a[0, 0], a[0, -1], a[-1, 0], a[-1, -1]])
        d = np.abs(a - bg)
    mask = d > bg_tol
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return xs.min(), ys.min(), xs.max(), ys.max()


def gate_media(path, slot="half", min_frames=8, verbose=True, edge_margin=0):
    """Check a PNG or GIF that is about to be published. Returns a list of issue strings."""
    from PIL import Image, ImageSequence
    issues = []
    tag = os.path.basename(path)
    if not os.path.exists(path):
        return [f"{tag}: missing"]
    if os.path.getsize(path) < 2048:
        issues.append(f"{tag}: file is only {os.path.getsize(path)} bytes")
    try:
        im = Image.open(path)
    except Exception as e:
        return [f"{tag}: will not decode ({e})"]

    w, h = im.size
    mw, mh = SLOT_MIN.get(slot, SLOT_MIN["half"])
    if w < mw or h < mh:
        issues.append(f"{tag}: {w}x{h}px is below the {mw}x{mh}px floor for a '{slot}' slot")

    frames = []
    for k, fr in enumerate(ImageSequence.Iterator(im)):
        frames.append(np.asarray(fr.convert("RGB")))
        if k > 400:
            break
    n = len(frames)

    if n == 0:
        return issues + [f"{tag}: no frames"]
    sizes = {f.shape[:2] for f in frames}
    if len(sizes) > 1:
        issues.append(f"{tag}: frame size changes mid-animation {sorted(sizes)}")

    for idx in ({0, n // 2, n - 1} if n > 1 else {0}):
        f = frames[idx]
        if float(f.std()) < 3.0:
            issues.append(f"{tag}: frame {idx} is blank or near-uniform")
        bb = _content_bbox(f)
        if bb is None:
            issues.append(f"{tag}: frame {idx} has no drawn content")
            continue
        x0, y0, x1, y1 = bb
        if edge_margin and (x0 < edge_margin or y0 < edge_margin or
                            x1 > f.shape[1] - 1 - edge_margin or y1 > f.shape[0] - 1 - edge_margin):
            issues.append(f"{tag}: frame {idx} content touches the canvas edge, so it is cropped")
        area = (x1 - x0) * (y1 - y0) / float(f.shape[0] * f.shape[1])
        if area < 0.05:
            issues.append(f"{tag}: frame {idx} content fills only {area*100:.1f}% of the canvas")

    if path.lower().endswith(".gif"):
        if n < min_frames:
            issues.append(f"{tag}: only {n} frames, below the {min_frames}-frame floor")
        if n > 1:
            # a duplicate frame is one that is pixel-identical, not one whose mean difference is
            # small: a small subject on a wide plain background moves a lot and averages to nothing
            same = sum(1 for k in range(1, n)
                       if np.abs(frames[k].astype(np.int16) - frames[k - 1].astype(np.int16)).max() <= 2)
            if same > 0.6 * (n - 1):
                issues.append(f"{tag}: {same}/{n-1} consecutive frames are identical, the animation is "
                              f"effectively frozen")

    if verbose:
        print(f"    GATE {tag}: {'clean' if not issues else str(len(issues)) + ' issue(s)'}"
              f" [{w}x{h}, {n} frame(s)]")
        for m in issues:
            print("        -", m)
    return issues
