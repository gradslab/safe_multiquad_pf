r"""Copy gated figures and animations into the project page.

Each file is checked again at the size the page shows it. Anything that fails does not ship.
"""
import json, os, shutil, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import webqc

def _site():
    """docs/ next to the checkout when this lives in the repo, or the sibling checkout in dev."""
    for c in (os.path.join(ROOT, "..", "docs"),
              os.path.join(ROOT, "..", "safe_multiquad_pf-main", "docs"),
              os.path.join(ROOT, "..", "safe_multiquad_pf", "docs")):
        if os.path.isdir(c):
            return os.path.abspath(c)
    return os.path.abspath(os.path.join(ROOT, "..", "docs"))


SITE = _site()
FIGS = os.path.join(SITE, "figs")
ANIM = os.path.join(SITE, "anim")
WEB = os.path.join(ROOT, "results", "web")
RES = os.path.join(ROOT, "results")

FIGURES = [
    ("W_space_circles", "wide"), ("W_space_sine", "wide"),
    ("W_channels_circles", "wide"), ("W_channels_sine", "wide"),
    ("W_authority_circles", "wide"), ("W_authority_sine", "wide"),
    ("W_pathdist_circles", "wide"), ("W_pathdist_sine", "wide"),
    ("W_feasibility_map", "wide"), ("W_mechanism", "wide"),
    ("W_rates", "wide"), ("W_bench", "wide"),
    ("W_weights", "wide"), ("W_actuator", "wide"),
]

ANIMS = [
    ("drake_render_circles_N4_proposed_web_iso.gif", "circles_proposed_iso", "half"),
    ("drake_render_circles_N4_proposed_web_top.gif", "circles_proposed_top", "half"),
    ("drake_render_circles_N4_proposed_web_side.gif", "circles_proposed_side", "half"),
    ("drake_render_circles_N4_baseline_web_iso.gif", "circles_baseline_iso", "half"),
    ("drake_render_circles_N4_baseline_web_top.gif", "circles_baseline_top", "half"),
    ("drake_render_circles_N4_se3_web_iso.gif", "circles_se3_iso", "half"),
    ("drake_render_circles_N4_se3_web_top.gif", "circles_se3_top", "half"),
    ("drake_render_sine_N2_proposed_web_iso.gif", "sine_proposed_iso", "half"),
    ("drake_render_sine_N2_proposed_web_top.gif", "sine_proposed_top", "half"),
    ("drake_render_sine_N2_proposed_web_side.gif", "sine_proposed_side", "half"),
    ("drake_render_sine_N2_baseline_web_iso.gif", "sine_baseline_iso", "half"),
    ("drake_render_sine_N2_baseline_web_top.gif", "sine_baseline_top", "half"),
    ("drake_render_sine_N2_se3_web_iso.gif", "sine_se3_iso", "half"),
    ("drake_render_sine_N2_se3_web_top.gif", "sine_se3_top", "half"),
]


def to_video(src_gif, dst_base):
    """A GIF of this length is several megabytes; the same frames as H.264 are a fraction of that.
    A poster frame is written alongside so the card has something to show before the video loads."""
    mp4 = dst_base + ".mp4"
    poster = dst_base + ".jpg"
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    poster_at = 0.45
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_gif, "-movflags", "+faststart",
                    "-pix_fmt", "yuv420p", "-vf", vf, "-crf", "26", "-preset", "slow", mp4],
                   check=True)
    # poster from the middle of the run, not the first frame: at t=0 the agents are at the edge of
    # frame and the card looks empty until the video starts
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-3", "-i", mp4,
                    "-vf", vf, "-frames:v", "1", "-q:v", "3", poster], check=True)
    return mp4, poster


def main(strict=True):
    os.makedirs(FIGS, exist_ok=True); os.makedirs(ANIM, exist_ok=True)
    manifest, failures = [], []
    for name, slot in FIGURES:
        src = os.path.join(WEB, name + ".png")
        if not os.path.exists(src):
            failures.append(f"{name}.png was never generated")
            continue
        dst = os.path.join(FIGS, name + ".png")
        shutil.copy2(src, dst)
        pdf = os.path.join(WEB, name + ".pdf")
        if os.path.exists(pdf):
            shutil.copy2(pdf, os.path.join(FIGS, name + ".pdf"))
        iss = webqc.gate_media(dst, slot=slot)
        failures += iss
        manifest.append({"file": f"figs/{name}.png", "slot": slot, "issues": iss})
    for src_name, dst_base, slot in ANIMS:
        src = os.path.join(RES, src_name)
        if not os.path.exists(src):
            failures.append(f"{src_name} is missing")
            continue
        iss = webqc.gate_media(src, slot=slot)          # gate the frames, then re-encode
        failures += iss
        mp4, poster = to_video(src, os.path.join(ANIM, dst_base))
        manifest.append({"file": f"anim/{dst_base}.mp4", "slot": slot, "issues": iss,
                         "MB": round(os.path.getsize(mp4) / 1e6, 2)})
    with open(os.path.join(SITE, "manifest.json"), "w") as fh:
        json.dump({"assets": manifest, "failures": failures}, fh, indent=1)
    print(f"\n{len(manifest)} assets published to {SITE}")
    if failures:
        print(f"{len(failures)} gate failure(s):")
        for f in failures:
            print("  -", f)
        if strict:
            sys.exit(1)


if __name__ == "__main__":
    main(strict="--force" not in sys.argv)
