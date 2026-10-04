"""
Export the held-out fixture as a 256x256 RGBA dump for the executed browser
pre-check (frontend/scripts/check_preflight.cjs), together with the figures
the pre-check should report for it, computed here independently.

Why this exists: the executed check ran on synthetic patterns only. Synthetic
patterns have no pixels near the foreground cut-off unless one is put there on
purpose; a photograph's aperture edge has them. Node has no PNG decoder without
a new dependency, so the pixels are exported once, by this script, and the
check reads the dump.

What the dump is and is not: a nearest-neighbour sample of the fixture at the
centres of a 256x256 grid, computed with integer arithmetic so that it does not
depend on any library's resampling filter. It is NOT the browser's own
`drawImage` scaling, which interpolates and differs between browsers; the
figures below describe this dump, not what a given browser would compute for
the same file.

The expected figures are computed with numpy from the definitions in
backend/app/services/validation/gate2_relevance.py (foreground = luminance
above 15; channel means over the foreground), not by running the TypeScript,
so the executed check compares two implementations.

Usage:
    python backend/scripts/export_preflight_fixture.py           # write the dump and its sidecar
    python backend/scripts/export_preflight_fixture.py --check   # exit 1 if either would change
"""
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_REL = "backend/tests/fixtures/aptos_heldout_d1f1ea894da1.png"
DUMP_REL = "frontend/scripts/fixtures/aptos_heldout_d1f1ea894da1_256.rgba"
SIDECAR_REL = "frontend/scripts/fixtures/aptos_heldout_d1f1ea894da1_256.json"
SIZE = 256
FOREGROUND_LUMINANCE_ABOVE = 15


def render():
    with open(os.path.join(REPO_ROOT, FIXTURE_REL), "rb") as fh:
        source_bytes = fh.read()
    image = Image.open(os.path.join(REPO_ROOT, FIXTURE_REL)).convert("RGB")
    width, height = image.size
    src = np.asarray(image, dtype=np.uint8)

    # the source pixel under the centre of each cell of the 256x256 grid
    ys = ((2 * np.arange(SIZE) + 1) * height) // (2 * SIZE)
    xs = ((2 * np.arange(SIZE) + 1) * width) // (2 * SIZE)
    rgb = src[ys][:, xs]
    rgba = np.concatenate([rgb, np.full((SIZE, SIZE, 1), 255, dtype=np.uint8)], axis=2)
    dump = rgba.tobytes()

    r = rgb[..., 0].astype(np.float64)
    g = rgb[..., 1].astype(np.float64)
    b = rgb[..., 2].astype(np.float64)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    foreground = luminance > FOREGROUND_LUMINANCE_ABOVE
    count = int(foreground.sum())
    if count:
        mean_r, mean_g, mean_b = (float(c[foreground].sum()) / count for c in (r, g, b))
    else:
        mean_r, mean_g, mean_b = (float(c.sum()) / (SIZE * SIZE) for c in (r, g, b))

    sidecar = {
        "source": FIXTURE_REL,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_width": width,
        "source_height": height,
        "source_size_bytes": len(source_bytes),
        "dump": DUMP_REL,
        "dump_sha256": hashlib.sha256(dump).hexdigest(),
        "method": "nearest-neighbour sample at the centres of a 256x256 grid (integer arithmetic); "
                  "not the browser's drawImage scaling",
        "expected": {
            "computed_by": "numpy in backend/scripts/export_preflight_fixture.py, from the definitions in "
                           "gate2_relevance.py; not by running the TypeScript",
            "foreground_pixels": count,
            "foreground_coverage": count / (SIZE * SIZE),
            "pixels_with_luminance_in_5_to_40": int(((luminance > 5) & (luminance <= 40)).sum()),
            "red_to_blue_ratio": mean_r / (mean_b + 1e-6),
            "red_share": mean_r / (mean_r + mean_g + mean_b + 1e-6),
        },
    }
    return dump, (json.dumps(sidecar, indent=2) + "\n").encode("utf-8")


def main():
    dump, sidecar = render()
    dump_path = os.path.join(REPO_ROOT, DUMP_REL)
    sidecar_path = os.path.join(REPO_ROOT, SIDECAR_REL)
    if "--check" in sys.argv:
        stale = []
        for path, wanted in ((dump_path, dump), (sidecar_path, sidecar)):
            if not os.path.exists(path):
                stale.append(path + " is missing")
                continue
            with open(path, "rb") as fh:
                have = fh.read()
            if path.endswith(".json"):
                have = have.replace(b"\r\n", b"\n")
            if have != wanted:
                stale.append(path + " differs from what the fixture yields")
        if stale:
            print("STALE: " + "; ".join(stale))
            return 1
        print("OK: the dump and its sidecar are what the fixture yields")
        return 0
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "wb") as fh:
        fh.write(dump)
    with open(sidecar_path, "wb") as fh:
        fh.write(sidecar)
    print("wrote %s (%d bytes) and %s" % (DUMP_REL, len(dump), SIDECAR_REL))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
