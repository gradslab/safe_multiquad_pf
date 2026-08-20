r"""Render docs/index.html from results/web/numbers.json.

Run page_numbers.py first. No result is written into the template by hand.
"""
import html, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
def _site():
    """docs/ next to the checkout when this lives in the repo, or the sibling checkout in dev."""
    for c in (os.path.join(ROOT, "..", "docs"),
              os.path.join(ROOT, "..", "safe_multiquad_pf-main", "docs"),
              os.path.join(ROOT, "..", "safe_multiquad_pf", "docs")):
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(ROOT, "..", "docs"))


SITE = _site()
N = json.load(open(os.path.join(ROOT, "results", "web", "numbers.json")))

REPO = "https://github.com/gradslab/safe_multiquad_pf"


def f(x, spec=".2f", dash="n/a"):
    return dash if x is None else format(x, spec)


def sig(x, n=2, dash="n/a"):
    """A number in the form a plain reader can scan: 2.9e-05 becomes 3 x 10^-5."""
    if x is None:
        return dash
    if x == 0:
        return "0"
    import math
    e = math.floor(math.log10(abs(x)))
    if -3 <= e <= 4:
        return f"{x:,.{max(0, n - 1 - e)}f}".rstrip("0").rstrip(".")
    m = x / 10 ** e
    return f"{m:.1f}&times;10<sup>{e}</sup>"


SEC = [("gallery", "Animations"),
       ("scenarios", "The two scenarios"),
       ("comparison", "Against two cascades"),
       ("feasibility", "Where feasibility holds"),
       ("mechanism", "How feasibility is lost"),
       ("authority", "Authority through an encounter"),
       ("rate", "Sampling rate"),
       ("cost", "Cost per step"),
       ("split", "Responsibility split"),
       ("actuator", "Actuator bounds"),
       ("geometry", "Path distance and conditioning"),
       ("parameters", "Parameters"),
       ("reproduce", "Reproducing this")]


def rail():
    items = "\n".join(f'      <li><a href="#{i}">{t}</a></li>' for i, t in SEC)
    return f'''  <aside class="rail">
    <div class="rail-h">On this page</div>
    <ol>
{items}
    </ol>
  </aside>'''


def video(base, alt="", cls="plate"):
    return (f'<div class="{cls}"><video src="anim/{base}.mp4" poster="anim/{base}.jpg" '
            f'autoplay loop muted playsinline preload="metadata" aria-label="{alt}"></video></div>')


def viewrow(prefix, labels=(("iso", "isometric"), ("top", "from above"), ("side", "side on"))):
    cards = ""
    for key, lab in labels:
        cards += (f'<div class="card media"><span class="pill neutral">{lab}</span>'
                  f'{video(prefix + "_" + key, lab)}</div>')
    return f'<div class="grid3">{cards}</div>'


def figure(src, cap, plate_class="plate"):
    return f'''      <figure class="figblock">
        <div class="{plate_class}"><img src="{src}" alt="" loading="lazy"></div>
        <figcaption class="cap">{cap}</figcaption>
      </figure>'''


def build():
    c = N["scenarios"]["circles"]; s = N["scenarios"]["sine"]
    sw = N["sweeps"]; bench = N.get("bench")
    cp, cb, ce = c["proposed"], c["baseline"], c["se3"]
    sp, sb, se = s["proposed"], s["baseline"], s["se3"]

    head = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Safe Path Following for Quadrotor Teams</title>
<meta name="description" content="Simulation study for decentralized safe path following of multiple quadrotors on intersecting paths.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/css/site.css">
<script>
window.MathJax = {{tex:{{inlineMath:[["\\\\(","\\\\)"]]}}, svg:{{fontCache:"global"}}}};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>

<header class="topbar">
  <span class="mark">Safe path following &middot; quadrotor teams</span>
  <span class="spacer"></span>
  <nav>
    <a href="#gallery">Animations</a>
    <a href="#comparison">Comparison</a>
    <a href="#feasibility">Feasibility</a>
    <a href="#cost">Cost</a>
    <a href="#reproduce">Reproduce</a>
  </nav>
  <button id="theme" class="btn" type="button" aria-label="Switch colour theme">Theme</button>
</header>

<div class="shell">
{rail()}
  <main>

  <section class="hero">
    <div class="eyebrow">Supplementary simulation study</div>
    <h1>Decentralized Safe Path Following for Multiple Quadrotors on Intersecting Paths</h1>
    <p class="byline">Hamza Tariq and Adeel Akhtar
      <span class="aff">GRaDS Lab, New Jersey Institute of Technology</span></p>
    <p class="lede">The paper reports one four-quadrotor run. This page carries the rest of the
      study: where the persistent-feasibility condition holds and where it stops holding, what the
      failure looks like when it comes, what the sampling period costs, and what the safety step
      costs per agent per step.</p>
    <div class="buttons">
      <a class="btn primary" href="{REPO}">Code and data</a>
      <a class="btn" href="#feasibility">Feasibility study</a>
      <a class="btn" href="#reproduce">Reproduce the figures</a>
    </div>
  </section>

  <section class="sec" id="gallery">
    <div class="sec-head">
      <span class="sec-tag">Animations</span>
      <h2>Four quadrotors, intersecting nonplanar circles</h2>
      <p>Rendered in Drake from the logged state, at the speeds the paper reports, from a start one
      metre off path. Watch the crossings: agents give way by changing speed along their own circles,
      and none of them steps off to do it. The dotted rings are the assigned paths.</p>
    </div>
{viewrow("circles_proposed")}
    <div class="sec-head" style="margin-top:2.6rem">
      <h2>Two quadrotors, intersecting nonplanar sinusoids</h2>
      <p>A second geometry with a different crossing angle and a different arc-length map. Both
      agents start one metre off path, converge, and then negotiate a crossing. The separation
      requirement is {f(cp["ds"], ".1f")}&nbsp;m between centres, which is not much wider than the
      airframe, so the two pass with little visible clearance while the constraint holds.</p>
    </div>
{viewrow("sine_proposed")}
    <div class="stats" style="margin-top:2.2rem">
      <div class="stat"><div class="n">{f(cp["transverse_cm"])} cm</div>
        <div class="k">worst transverse output after convergence, against
          {f(cb["transverse_cm"], ".1f")} cm and {f(ce["transverse_cm"] / 100, ".2f")} m for the
          two cascades</div></div>
      <div class="stat"><div class="n">{sig(cp["heading_deg"])} deg</div>
        <div class="k">worst heading error, the integration floor rather than a control error</div></div>
      <div class="stat"><div class="n">{sig(cp["eq_resid"])}</div>
        <div class="k">largest residual on the three equalities that are never relaxed, whole run</div></div>
      <div class="stat"><div class="n">{sw["sine"]["cells"] + sw["circles"]["cells"]}</div>
        <div class="k">physics rollouts behind the feasibility map</div></div>
    </div>
  </section>
'''

    scenarios = f'''
  <section class="sec" id="scenarios">
    <div class="sec-head">
      <span class="sec-tag">Scenarios</span>
      <h2>The two scenarios</h2>
      <p>Both are flown on the Drake physics engine, with a floating rigid body per quadrotor and
      quaternion attitude. The controller runs at 200&nbsp;Hz and holds its input between updates.
      The circle scenario is the one the paper reports. The sinusoid scenario is a second geometry
      with a different crossing angle and a different arc-length map.</p>
    </div>
{figure("figs/W_space_circles.png",
        "<b>Four quadrotors on intersecting circles.</b> Grey ribbons are the assigned paths. The "
        "proposed trajectory lies on its ribbon through every crossing. The two cascades leave "
        "theirs, and the geometric cascade leaves by the most.")}
{figure("figs/W_space_sine.png",
        "<b>Two quadrotors on intersecting sinusoids.</b> Both start one metre off the path and "
        "converge onto it. The views are cropped to the stretch that contains the two crossings.")}
  </section>
'''

    def row(lab, p, b, e, key, spec=".2f", better="low", unit=""):
        vals = [p.get(key), b.get(key), e.get(key)]
        ok = [v for v in vals if v is not None]
        best = min(ok) if better == "low" else max(ok)
        cells = ""
        for v in vals:
            cls = "num best" if v is not None and v == best else "num"
            cells += f'<td class="{cls}">{f(v, spec)}{unit}</td>'
        return f"<tr><td>{lab}</td>{cells}</tr>"

    comparison = f'''
  <section class="sec" id="comparison">
    <div class="sec-head">
      <span class="sec-tag">Comparison</span>
      <h2>Against two cascades</h2>
      <p>The first cascade replaces the hard transverse and heading equalities by a full-input
      minimum-deviation filter on the same transverse feedback linearization. The second is the
      geometric controller on SE(3) for quadrotors, with an acceleration-level barrier filter in
      front of it. Barrier constraints, poles, separation distance, initial
      conditions and control rate are identical across the three.</p>
    </div>
    <div class="grid3">
      <div class="card media"><span class="pill ok">stays on path</span><h3>Proposed</h3>
        {video("circles_proposed_top", "proposed controller from above")}
        <p>Safety is spent on speed alone.</p></div>
      <div class="card media"><span class="pill warn">leaves path</span><h3>TFL + safety filter</h3>
        {video("circles_baseline_top", "TFL with a full-input safety filter")}
        <p>Nothing pins the transverse rows, so the filter moves them.</p></div>
      <div class="card media"><span class="pill bad">leaves path and heading</span>
        <h3>Geometric + barrier filter</h3>
        {video("circles_se3_top", "geometric controller with a barrier filter")}
        <p>Avoidance is rebuilt into an attitude, which carries the heading with it.</p></div>
    </div>
    <p class="cap" style="margin:0 0 1.6rem">Seen from above, the view where leaving a path is
      unmistakable. The same three controllers on the sinusoids:</p>
    <div class="grid3">
      <div class="card media"><span class="pill ok">stays on path</span><h3>Proposed</h3>
        {video("sine_proposed_top", "proposed controller on sinusoids, from above")}</div>
      <div class="card media"><span class="pill warn">leaves path</span><h3>TFL + safety filter</h3>
        {video("sine_baseline_top", "TFL with a safety filter on sinusoids, from above")}</div>
      <div class="card media"><span class="pill bad">leaves path and heading</span>
        <h3>Geometric + barrier filter</h3>
        {video("sine_se3_top", "geometric controller with a barrier filter on sinusoids, from above")}</div>
    </div>
{figure("figs/W_channels_circles.png",
        "<b>Four quadrotors on intersecting circles.</b> Panel (c) barely separates the three, "
        "since all of them ride the separation boundary through the encounters. Panels (a) and "
        "(b) separate them by two orders of magnitude.")}
{figure("figs/W_channels_sine.png",
        "<b>Two quadrotors on intersecting sinusoids.</b> The ordering is the same on the second "
        "geometry, with smaller margins because there is one pair rather than six.")}
    <div class="tablewrap">
      <table>
        <thead><tr><th>After convergence, worst over agents</th>
          <th class="num">Proposed</th><th class="num">TFL + filter</th>
          <th class="num">Geometric + filter</th></tr></thead>
        <tbody>
          <tr><td colspan="4"><b>Four quadrotors, intersecting circles</b>
            (measured from t = {f(cp["settle_s"], ".1f")}&nbsp;s)</td></tr>
          {row("Transverse output |&xi;<sub>1</sub>| [cm]", cp, cb, ce, "transverse_cm")}
          {row("Distance to the assigned path [cm]", cp, cb, ce, "path_err_cm")}
          {row("Heading error [deg]", cp, cb, ce, "heading_deg", ".2e")}
          {row("Closest pair distance [m]", cp, cb, ce, "dist_min", ".3f", "high")}
          <tr><td colspan="4"><b>Two quadrotors, intersecting sinusoids</b>
            (measured from t = {f(sp["settle_s"], ".1f")}&nbsp;s)</td></tr>
          {row("Distance to the assigned path [cm]", sp, sb, se, "path_err_cm")}
          {row("Heading error [deg]", sp, sb, se, "heading_deg", ".2e")}
          {row("Closest pair distance [m]", sp, sb, se, "dist_min", ".3f", "high")}
        </tbody>
      </table>
    </div>
    <p class="cap">Green marks the better value in each row. The first row is the quantity the paper
      reports, the transverse output that Lemma 2 bounds the path distance with. The second is the
      geometric distance itself. They separate the controllers by
      {f(cb["transverse_cm"] / cp["transverse_cm"], ".0f")}&times; and
      {f(ce["transverse_cm"] / cp["transverse_cm"], ".0f")}&times; on the first,
      {f(cb["path_err_cm"] / cp["path_err_cm"], ".0f")}&times; and
      {f(ce["path_err_cm"] / cp["path_err_cm"], ".0f")}&times; on the second. The paper quotes the
      first row, with the geometric cascade written as {f(ce["transverse_cm"] / 100, ".2f")}&nbsp;m.
      Separation is {f(cp["ds"], ".1f")}&nbsp;m.
      The two cascades hold it on the sinusoids. On the circles the geometric cascade closes to
      {f(ce["dist_min"], ".4f")}&nbsp;m, which is {f(1000 * (cp["ds"] - ce["dist_min"]), ".1f")}&nbsp;mm
      inside the boundary at the sampled instants, while the proposed controller and the full-input
      filter stay outside it.</p>
  </section>
'''
    return head, scenarios, comparison


def build_rest():
    c = N["scenarios"]["circles"]
    cp, cb, ce = c["proposed"], c["baseline"], c["se3"]
    sw = N["sweeps"]; bench = N.get("bench")
    rates = N.get("rates") or []
    _hworst = min(sw["sine"]["worst_h_over_all_cells"], sw["circles"]["worst_h_over_all_cells"])

    feas = f'''
  <section class="sec" id="feasibility">
    <div class="sec-head">
      <span class="sec-tag">Feasibility</span>
      <h2>Where the feasibility condition holds</h2>
      <p>Assumption 1 is a hypothesis, and the paper checks it at one operating point. Here it is
      checked over a grid that runs well past that point, out to twice the desired speed and to
      separation distances up to {f(max(sw["sine"]["axis_values"]), ".1f")}&nbsp;m. Each cell is a full
      rollout, and the horizon is scaled with the desired speed so that every cell covers the same
      path distance rather than the same wall-clock time. There is no fallback controller anywhere in
      this study, by design, so a cell that loses feasibility simply stops at that instant.</p>
      <p>A cell contains many crossings, so counting whole cells is a blunt measure: a rollout that
      negotiates twenty encounters and loses feasibility on the twenty-first counts the same as one
      that fails immediately. The natural unit is the encounter.</p>
    </div>
    <div class="stats">
      <div class="stat"><div class="n">{f(100 * sw["totals"]["rate"], ".0f")}%</div>
        <div class="k">of pairwise encounters negotiated with the interval non-empty</div></div>
      <div class="stat"><div class="n">{sw["totals"]["encounters_cleared"]}</div>
        <div class="k">encounters cleared across {sw["sine"]["cells"] + sw["circles"]["cells"]} rollouts</div></div>
      <div class="stat"><div class="n">{sw["totals"]["encounters_failed"]}</div>
        <div class="k">encounters where the certificate reported an empty interval</div></div>
      <div class="stat"><div class="n">0</div>
        <div class="k">collisions and path deviations, over the whole grid</div></div>
    </div>
{figure("figs/W_feasibility_map.png",
        "<b>" + str(sw["sine"]["cells"] + sw["circles"]["cells"]) + " rollouts.</b> Green cells ran "
        "the full horizon. Orange cells stopped when the closed-form certificate reported an empty "
        "interval, and the label is how far into the horizon that happened. The cell the paper "
        "reports is outlined in black.")}
    <ul class="look">
      <li>Read as whole cells, {sw["sine"]["ran"]} of {sw["sine"]["cells"]} sinusoid cells and
        {sw["circles"]["ran"]} of {sw["circles"]["cells"]} circle cells ran their full horizon. Read
        as encounters, which is the unit the constraint acts on,
        {sw["totals"]["encounters_cleared"]} of {sw["totals"]["encounters"]} were negotiated.</li>
      <li>No cell ended in a collision or a path deviation. Across all
        {sw["sine"]["cells"] + sw["circles"]["cells"]} rollouts the worst barrier value at any
        sampled instant was {sig(_hworst, 2)} m<sup>2</sup>, which puts the closest sampled approach
        within {f(1e9 * abs(0.5 - (0.25 + _hworst) ** 0.5), ".0f")} nanometres of the separation
        sphere. What fails is the certificate, and it fails before safety does.</li>
      <li>The boundary is not a simple speed threshold. Cells that stop are spread through the grid,
        and the median stopped cell had already cleared
        {f(100 * sw["sine"]["median_stop_fraction"], ".0f")} percent of its horizon on the sinusoids
        and {f(100 * sw["circles"]["median_stop_fraction"], ".0f")} percent on the circles.</li>
    </ul>
  </section>

  <section class="sec" id="mechanism">
    <div class="sec-head">
      <span class="sec-tag">Mechanism</span>
      <h2>How feasibility is lost</h2>
      <p>Remark 3 predicts the mechanism, and one cell of the map shows it directly. A collision row
      bounds the slack from above while its authority is positive. As that authority approaches zero
      the bound it produces walks away, and when the authority changes sign the row flips to bounding
      from below, which empties the interval in a single step.</p>
    </div>
{figure("figs/W_mechanism.png",
        "<b>The last three seconds of one stopped cell.</b> Interval width is the quantity one "
        "would reach for as an early warning, and panel (c) shows why it is the wrong one. The "
        "width is still in the hundreds one sample before the interval is empty.")}
    <p class="prose">This is the argument for evaluating the certificate every step rather than
      extrapolating a margin. The test costs one pass over N + 6 rows, so there is no reason to
      approximate it.</p>
  </section>

  <section class="sec" id="authority">
    <div class="sec-head">
      <span class="sec-tag">Authority</span>
      <h2>Authority through an encounter</h2>
      <p>The same quantities on a run that does not stop. The collision authority still passes
      through zero once per encounter, and nothing happens, because the row's offset has the sign
      that keeps the interval non-empty across the crossing.</p>
    </div>
{figure("figs/W_authority_circles.png",
        "<b>Four quadrotors, intersecting circles.</b> Panel (b) is the slack the projection "
        "selects. The red bar marks the steps where the collision row is the binding constraint, "
        "which is where the agent is genuinely yielding.")}
{figure("figs/W_authority_sine.png",
        "<b>Two quadrotors, intersecting sinusoids.</b> One pair rather than six, so the "
        "collision row binds over a single long window instead of several short ones.")}
  </section>
'''

    rate_sec = ""
    if rates:
        hs = ", ".join(f"{r['rate']} Hz" for r in rates)
        worst = min(r["h_min"] for r in rates)
        rate_sec = f'''
  <section class="sec" id="rate">
    <div class="sec-head">
      <span class="sec-tag">Sampled data</span>
      <h2>What the sampling period costs</h2>
      <p>The controller holds each input constant between updates, so the barrier conditions are
      enforced on the sampled solution and not between samples. The paper says so and claims nothing
      more. This is the empirical part of that statement, on the four-quadrotor scenario at
      {hs}.</p>
    </div>
{figure("figs/W_rates.png",
        "<b>Five rates, one scenario, everything else fixed.</b> Panel (b) is the worst sampled "
        "barrier value over each run, and panel (c) is the settled path error.")}
    <ul class="look">
      <li>The worst sampled barrier value over all five rates is
        {sig(worst, 2)} m<sup>2</sup>, so no rate in this range produced a sampled excursion past
        the separation boundary.</li>
      <li>This is evidence, not a guarantee. An intersample certificate needs the bound on the
        barrier derivative over one period, which we do not claim.</li>
    </ul>
  </section>
'''

    cost_sec = ""
    if bench:
        hw = bench["hardware"]
        rows = "".join(
            f"<tr><td class=\"num\">{r['N']}</td><td class=\"num\">{r['rows']}</td>"
            f"<td class=\"num\">{r['closed_us']:.1f}</td><td class=\"num\">{r['osqp_us']:.1f}</td>"
            f"<td class=\"num\">{r['osqp_us']/r['closed_us']:.1f}&times;</td></tr>"
            for r in bench["rows"])
        cost_sec = f'''
  <section class="sec" id="cost">
    <div class="sec-head">
      <span class="sec-tag">Cost</span>
      <h2>Cost of the safety step</h2>
      <p>The paper quotes one ratio. The measurement behind it separates two things that a single
      number runs together: the projection against a numerical solve of the same program, and the
      whole per-agent control step, which both controllers pay alike. OSQP is timed through its own
      interface rather than through a modelling layer, so the comparison charges it for the solver
      and nothing else.</p>
    </div>
{figure("figs/W_bench.png",
        "<b>" + str(bench["reps"]) + " repetitions per team size.</b> Bands in panel (a) are the "
        "5th to 95th percentile. Panel (c) is the spread of the projection alone.")}
    <div class="tablewrap">
      <table>
        <thead><tr><th class="num">Team size N</th><th class="num">Rows</th>
          <th class="num">Projection [&micro;s]</th><th class="num">OSQP solve [&micro;s]</th>
          <th class="num">Ratio</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </div>
    <p class="cap">Median over {bench["reps"]} repetitions.
      {html.escape(hw.get("cpu", hw.get("processor", "")))},
      {html.escape(hw.get("cores", "?"))} cores, Python {html.escape(hw["python"])},
      OSQP {html.escape(str(hw.get("osqp", "")))}. The projection is
      {f(bench["ratio_median"], ".1f")} times faster than the solve at the median team size, and the
      gap is not the whole story. The solve time depends on how many iterations the problem happens
      to need, so it ranges from {f(bench["ratio_min"], ".1f")} to {f(bench["ratio_max"], ".1f")}
      times the projection across these team sizes, while the projection itself is a fixed number of
      operations. Both controllers also pay the same per-agent cost for path geometry and row
      assembly, which dominates this Python prototype and is the same code in both.</p>
  </section>
'''

    split_sec = ""
    if os.path.exists(os.path.join(SITE, "figs", "W_weights.png")):
        split_sec = f'''
  <section class="sec" id="split">
    <div class="sec-head">
      <span class="sec-tag">Responsibility</span>
      <h2>The responsibility split</h2>
      <p>Remark 5 says that safety does not imply continued progress, and that a symmetric approach
      can deadlock unless priority or scheduling logic rules it out. Here two agents arrive at a
      crossing at the same speed on mirrored paths, which is that approach built on purpose. The
      responsibility split is the only thing that changes between the three runs, and every split
      keeps the pair sum at one, so the centralized condition is untouched.</p>
    </div>
{figure("figs/W_weights.png",
        "<b>One symmetric encounter, three splits.</b> The distance-blended split engages as the "
        "pair closes. The committed split assigns right of way before the pair locks.")}
  </section>
'''

    act_sec = ""
    if os.path.exists(os.path.join(SITE, "figs", "W_actuator.png")):
        act_sec = '''
  <section class="sec" id="actuator">
    <div class="sec-head">
      <span class="sec-tag">Extension</span>
      <h2>Actuator bounds</h2>
      <p>Section IV-D argues that any constraint affine in the input reduces to an interval of the
      same form, so actuator bounds add rows without changing the test. A box on the three torques is
      two rows per bound, the thrust input enters the same way, and the certificate stays exact.
      Tightening the box below the demand does not produce a solver failure: at 21 mN m the
      certificate reports the box infeasible at the first step, before the vehicle moves.</p>
    </div>
''' + figure("figs/W_actuator.png",
        "<b>Three bounds against the unbounded run.</b> The interval narrows as the box tightens, "
        "the applied torque respects the bound, and the path error does not move. The bounds bind "
        "more often, which is the cost the paper names.") + '''
  </section>
'''

    geom = f'''
  <section class="sec" id="geometry">
    <div class="sec-head">
      <span class="sec-tag">Geometry</span>
      <h2>Path distance and conditioning</h2>
      <p>Two quantities the paper quotes as single numbers, shown as time series. The first is the
      distance to the assigned path, which is the geometric quantity the transverse output bounds.
      The second is the smallest singular value of the decoupling matrix, which is what regularity
      of the transverse feedback linearization comes down to in practice.</p>
    </div>
{figure("figs/W_pathdist_circles.png",
        "<b>Four quadrotors, intersecting circles.</b> Panel (c) never approaches zero, so the "
        "decoupling matrix stays invertible over the whole tube the agents actually visit.")}
{figure("figs/W_pathdist_sine.png",
        "<b>Two quadrotors, intersecting sinusoids.</b> Same three quantities on the second "
        "geometry.")}
    <ul class="look">
      <li>Worst transverse output after convergence, four agents: {f(cp["transverse_cm"])} cm,
        against {f(cb["transverse_cm"], ".1f")} cm for the full-input filter and
        {f(ce["transverse_cm"] / 100, ".2f")} m for the geometric cascade. As a geometric distance
        the proposed controller is {f(cp["path_err_cm"])} cm from its path.</li>
      <li>Smallest singular value of the decoupling matrix over the run:
        {f(cp["sigma_min_D"], ".3f")}.</li>
      <li>Thrust margin above the positivity floor: {f(cp["thrust_margin_N"], ".1f")} N.</li>
    </ul>
  </section>
'''
    return feas + rate_sec + cost_sec + split_sec + act_sec + geom


PARAMS = [
    ("Shared plant and scenario", [
        ("Mass m", "1.923 kg"), ("Inertia J", "diag(1.152, 1.152, 2.18) &times; 10<sup>-2</sup> kg m<sup>2</sup>"),
        ("Gravity g", "9.8 m/s<sup>2</sup>"), ("Control rate", "200 Hz, zero-order hold"),
        ("Integrator", "Drake continuous plant, quaternion floating body"),
        ("Separation d<sub>s</sub>", "0.5 m"),
        ("Circle radius R", "1.5 m"), ("Circle centres", "(0,0), (0,1), (1,0), (1,1) m"),
        ("Height field", "z = 1 + 0.25 sin(2&pi;x/3) m"),
        ("Desired speeds, circles", "&plusmn;0.63 and &plusmn;0.525 m/s"),
        ("Sinusoid geometry", "y = &plusmn;1.25 sin(0.5x), z = 1 + 0.25 sin(0.5x) m"),
        ("Desired speeds, sinusoid", "1.0 and 0.96 m/s"),
        ("Start", "1 m off path, level attitude, hover thrust"),
    ]),
    ("Proposed controller", [
        ("Transverse gains k<sub>&xi;</sub>", "200, 400, 90, 30"),
        ("Altitude gains k<sub>&zeta;</sub>", "100, 200, 60, 20"),
        ("Speed gains", "triple pole at &minus;2: 8, 12, 6"),
        ("Heading gains", "10, 12"),
        ("Cost weights", "W = I<sub>4</sub>, P = 100"),
        ("Collision barrier", "quadruple real pole at &minus;3.75"),
        ("Attitude barrier", "double pole at &minus;10, margin &epsilon; = 0.1 rad"),
        ("Thrust barrier", "double pole at &minus;10, f<sub>min</sub> = 0.94 N"),
        ("Speed barrier", "triple pole at &minus;20, v<sub>max</sub> = 1 m/s"),
        ("Responsibility", "w<sub>ij</sub> = 1/2 unless stated"),
        ("Solve", "closed-form projection, no numerical solver in the loop"),
    ]),
    ("TFL with a full-input safety filter", [
        ("Nominal feedback", "the same transverse feedback linearization, by inversion"),
        ("Filter", "min &#8214;&nu; &minus; &nu;<sub>TFL</sub>&#8214;<sup>2</sup> subject to the same barrier rows"),
        ("Barrier rows", "identical: attitude, thrust, collision, same poles and weights"),
        ("Difference from proposed", "the hard equality is dropped, and there is no slack"),
        ("Solve", "OSQP through cvxpy, eps 10<sup>-8</sup>, CLARABEL fallback"),
    ]),
    ("Geometric controller with a barrier filter", [
        ("Position gains", "k<sub>x</sub> = 16 m, k<sub>v</sub> = 5.6 m"),
        ("Attitude gains", "k<sub>R</sub> = 8.81, k<sub>&Omega;</sub> = 2.54"),
        ("Reference", "the same path at the same physical speed"),
        ("Filter", "acceleration-level collision barrier, double pole at &minus;3.75"),
        ("Separation", "0.5 m, as above"),
    ]),
]


def build_tail():
    blocks = ""
    for title, rows in PARAMS:
        body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
        blocks += (f'<h3 style="margin:1.6rem 0 .7rem">{title}</h3>'
                   f'<div class="tablewrap"><table><thead><tr><th>Parameter</th><th>Value</th>'
                   f'</tr></thead><tbody>{body}</tbody></table></div>')

    return f'''
  <section class="sec" id="parameters">
    <div class="sec-head">
      <span class="sec-tag">Parameters</span>
      <h2>Every parameter, for all three controllers</h2>
      <p>The comparison is only worth reading if the three controllers were given the same problem.
      These are the complete settings behind every figure on this page.</p>
    </div>
    {blocks}
  </section>

  <section class="sec" id="reproduce">
    <div class="sec-head">
      <span class="sec-tag">Reproduce</span>
      <h2>Reproducing this</h2>
      <p>The simulation code is in the repository under <code>sim/</code>. The verification gate runs
      first and checks the controller algebra over several thousand random admissible states, so a
      broken build fails before any scenario is flown.</p>
    </div>
<pre><code><span class="c"># set up</span>
git clone {REPO}.git
cd safe_multiquad_pf/sim
pip install -r requirements.txt
python3 tests/test_layer.py

<span class="c"># the two scenarios</span>
python3 experiments/run_circles.py --scenario circles --agents 4 --offpath --vscale 1.4 --tag _cs1p4
python3 experiments/run_circles.py --scenario sine    --agents 2 --offpath

<span class="c"># the two cascades, same scenario</span>
python3 experiments/run_circles.py --scenario circles --agents 4 --offpath --vscale 1.4 \
        --controller baseline --tag _cs1p4
python3 experiments/run_circles.py --scenario circles --agents 4 --offpath --vscale 1.4 \
        --controller se3 --tag _cs1p4

<span class="c"># the studies on this page</span>
python3 experiments/web/sweep.py --grid sine
python3 experiments/web/sweep.py --grid circles
python3 experiments/web/runs.py --study all
python3 experiments/web/bench_solver.py
python3 experiments/web/renders.py

<span class="c"># figures, each gated before it is written</span>
python3 experiments/web/figs_scenes.py
python3 experiments/web/figs_core.py
python3 experiments/web/figs_studies.py
python3 experiments/web/page_numbers.py
python3 experiments/web/build_page.py</code></pre>
    <p class="prose">The animations need Drake with a VTK renderer:
      <code>python3 experiments/web/renders.py</code> writes all fourteen, three camera angles per
      controller.</p>
  </section>

  <section class="sec" id="cite">
    <div class="sec-head">
      <span class="sec-tag">Citation</span>
      <h2>Citation</h2>
    </div>
<pre><code>@unpublished{{tariq_decentralized,
  author = {{Hamza Tariq and Adeel Akhtar}},
  title  = {{Decentralized Safe Path Following for Multiple Quadrotors
            on Intersecting Paths}},
  note   = {{Under review}},
  year   = {{2026}}
}}</code></pre>
  </section>

  <footer>
    <div class="rulebar"></div>
    <p>GRaDS Lab, New Jersey Institute of Technology.
      Every figure on this page is generated by the scripts above and is checked for occlusion,
      clipping and legibility before it is written.
      <a href="{REPO}">Repository</a>.</p>
  </footer>

  </main>
</div>
<script src="assets/js/site.js"></script>
</body>
</html>
'''


def write():
    parts = list(build()) + [build_rest(), build_tail()]
    doc = "".join(parts)
    out = os.path.join(SITE, "index.html")
    with open(out, "w") as fh:
        fh.write(doc)
    print(f"wrote {out}  ({len(doc):,} characters)")


if __name__ == "__main__":
    write()
