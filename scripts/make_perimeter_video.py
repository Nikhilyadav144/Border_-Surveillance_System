#!/usr/bin/env python3
"""Build a synthetic PERIMETER FENCE clip for CAM-02.

WHAT THIS IS
------------
A composited test clip, not real footage. A photographic cut-out of a person
(from the sample image that ships with Ultralytics) is moved across a drawn
fence scene along a scripted path.

The detections the application makes on it are REAL: YOLO genuinely finds the
people, ByteTrack genuinely follows them, and the zone, line-crossing and
loitering logic genuinely fire. Only the scene is synthetic.

WHY IT EXISTS
-------------
CAM-02's job is virtual fence and restricted-zone intrusion. CAM-01's demo clip
has the wrong geometry for that. This clip is built to exercise CAM-02's rules
specifically, including a negative control, so the intrusion logic can be
developed and demonstrated before real footage exists.

It is a test fixture. Never present output from it as a real-world result, and
never quote accuracy figures measured on it. See docs/DATA.md for how to obtain
footage you can legitimately use.

THE SCENARIO (30 s, matched to the `perimeter_fence` zone profile)
------------------------------------------------------------------
    PATROL GUARD   walks the patrol track for the whole clip and never
                   crosses the fence. This is the NEGATIVE CONTROL: if this
                   person raises an intrusion, the rules are wrong.

    INTRUDER       0-7 s    approaches along the outer ground
                   7-9 s    crosses the fence line       -> LINE CROSSING
                                                         -> RESTRICTED INTRUSION
                   9-24 s   stands still inside          -> LOITERING (after 8 s)
                   24-30 s  moves deeper into the compound

USAGE
-----
    python scripts/make_perimeter_video.py
    python scripts/make_perimeter_video.py --output data/input/sample_02.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paths import PROJECT_ROOT, ensure_dir  # noqa: E402

WIDTH, HEIGHT = 960, 540
FPS = 20
DURATION = 30

#: Normalised y of the fence. Must match the `perimeter_fence` zone profile
#: in config/zones.yaml. Change one and you must change the other.
FENCE_Y = 0.46
FENCE_PX = int(HEIGHT * FENCE_Y)


def _ultralytics_assets() -> Path | None:
    try:
        import ultralytics
        assets = Path(ultralytics.__file__).parent / "assets"
        return assets if assets.is_dir() else None
    except ImportError:
        return None


def _extract_person(image: np.ndarray, model, skip: int = 0):
    """Crop exactly ONE person out of a sample image.

    The crop is verified, not assumed. A high-confidence person box can be a
    wide sprawling one that also contains other people: ``zidane.jpg``'s best
    box is 1000x514 and holds three detectable figures. Pasting that as a
    single "person" sprite makes one scripted actor appear as several tracks
    and sends the tracker's id churn through the roof, which then looks like a
    tracking bug rather than a bad fixture.

    So each candidate crop is run back through the detector, and only a crop
    that yields exactly one person is accepted.

    ``skip`` selects among the accepted candidates, so two actors in the same
    clip are visibly different people.
    """
    results = model.predict(image, conf=0.35, verbose=False)
    if not results or results[0].boxes is None:
        return None

    names = model.names
    people = [
        (float(box.conf[0]), box.xyxy[0].tolist())
        for box in results[0].boxes
        if names.get(int(box.cls[0]), "") == "person"
    ]
    people.sort(key=lambda item: -item[0])

    accepted = []
    for _, (x1, y1, x2, y2) in people:
        x1, y1, x2, y2 = (int(round(v)) for v in (x1, y1, x2, y2))
        crop = image[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0 or crop.shape[0] < 60:
            continue
        # A usable sprite is taller than it is wide. A wide box is almost
        # always a group or a badly-fitted one.
        if crop.shape[1] > crop.shape[0]:
            continue
        check = model.predict(crop, conf=0.35, verbose=False)[0]
        found = 0 if check.boxes is None else sum(
            1 for b in check.boxes if names.get(int(b.cls[0]), "") == "person"
        )
        if found == 1:
            accepted.append(crop)

    if not accepted:
        return None
    return accepted[min(skip, len(accepted) - 1)]


def _background() -> np.ndarray:
    """Ground, a patrol track, and a chain-link perimeter fence."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    # Ground: cooler and darker beyond the fence, warmer in front of it.
    for y in range(HEIGHT):
        if y < FENCE_PX:
            blend = y / max(FENCE_PX, 1)
            top = np.array([96, 104, 88], np.float32)
            bottom = np.array([78, 96, 82], np.float32)
        else:
            blend = (y - FENCE_PX) / max(HEIGHT - FENCE_PX, 1)
            top = np.array([88, 108, 122], np.float32)
            bottom = np.array([70, 92, 112], np.float32)
        frame[y, :] = (top * (1 - blend) + bottom * blend).astype(np.uint8)

    # Compacted patrol track running along the near side of the fence.
    cv2.rectangle(frame, (0, FENCE_PX + 74), (WIDTH, FENCE_PX + 168),
                  (96, 116, 138), -1)

    # --- the fence itself -------------------------------------------------
    # Posts, then horizontal wires, then a light mesh. Drawn thin so it does
    # not occlude a person enough to defeat detection: the point of the clip
    # is to test the zone rules, not the detector.
    for x in range(30, WIDTH, 84):
        cv2.line(frame, (x, FENCE_PX - 46), (x, FENCE_PX + 16),
                 (128, 134, 142), 3)
    for offset in (-44, -30, -16, -2, 12):
        cv2.line(frame, (0, FENCE_PX + offset), (WIDTH, FENCE_PX + offset),
                 (118, 126, 136), 1)
    for x in range(0, WIDTH + 40, 20):
        cv2.line(frame, (x, FENCE_PX - 46), (x - 40, FENCE_PX + 14),
                 (104, 112, 122), 1)
        cv2.line(frame, (x, FENCE_PX + 14), (x - 40, FENCE_PX - 46),
                 (104, 112, 122), 1)

    # Scattered vegetation so the scene is not perfectly uniform.
    rng = np.random.default_rng(11)
    for _ in range(160):
        x = int(rng.integers(0, WIDTH))
        y = int(rng.integers(0, HEIGHT))
        if FENCE_PX + 70 < y < FENCE_PX + 170:
            continue
        radius = int(rng.integers(3, 9))
        shade = int(rng.integers(60, 96))
        cv2.circle(frame, (x, y), radius, (shade - 8, shade + 14, shade), -1)

    noise = rng.normal(0, 4, (HEIGHT, WIDTH, 3))
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _paste(canvas: np.ndarray, sprite: np.ndarray,
           cx: int, feet_y: int, height_px: int) -> None:
    """Paste *sprite* so its FEET land on ``feet_y``.

    Anchoring on the feet matters: the zone engine uses the bottom-centre of a
    person's box as their position, so the scripted path and the rule that
    reads it agree about where the person is standing.
    """
    if sprite is None or sprite.size == 0 or height_px < 8:
        return
    scale = height_px / sprite.shape[0]
    w = max(4, int(sprite.shape[1] * scale))
    resized = cv2.resize(sprite, (w, height_px), interpolation=cv2.INTER_AREA)

    x1 = int(cx - w / 2)
    y1 = int(feet_y - height_px)
    x2, y2 = x1 + w, y1 + height_px

    sx1, sy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] = resized[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]


def _perspective_height(feet_y_norm: float) -> int:
    """Taller near the camera, shorter far away. Crude but it reads correctly."""
    return int(72 + 96 * max(0.0, min(1.0, feet_y_norm)))


def _intruder_path(t: float) -> tuple[float, float]:
    """(x, feet_y) in normalised coordinates, for the scripted intruder."""
    if t < 7.0:                                   # approach from the outside
        p = t / 7.0
        return 0.30 + p * 0.22, 0.86 - p * 0.34
    if t < 9.0:                                   # cross the fence
        p = (t - 7.0) / 2.0
        return 0.52 + p * 0.02, 0.52 - p * 0.13
    if t < 24.0:                                  # stand still: loitering
        wobble = 0.004 * np.sin((t - 9.0) * 1.6)
        return 0.54 + wobble, 0.39 + wobble * 0.4
    p = min(1.0, (t - 24.0) / 6.0)                # move deeper inside
    return 0.54 - p * 0.16, 0.39 - p * 0.14


def _guard_path(t: float) -> tuple[float, float]:
    """The negative control: patrols the track, never crosses the fence."""
    # Paces from left to right and back, always well outside the fence.
    cycle = (t % 20.0) / 20.0
    x = 0.08 + (cycle * 2 if cycle < 0.5 else (1 - cycle) * 2) * 0.62
    return x, 0.80 + 0.01 * np.sin(t * 1.1)


def build(output: Path, show_progress: bool = True) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is required to build the clip.\n"
            "Run: pip install -r requirements.txt"
        )

    assets = _ultralytics_assets()
    if assets is None:
        raise SystemExit("Could not locate the Ultralytics sample images.")

    weights = PROJECT_ROOT / "models" / "yolov8n.pt"
    model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")

    zidane = cv2.imread(str(assets / "zidane.jpg"))
    bus = cv2.imread(str(assets / "bus.jpg"))
    if zidane is None or bus is None:
        raise SystemExit("Could not read the Ultralytics sample images.")

    # bus.jpg yields clean, well-separated single-person crops; zidane.jpg's
    # best box is a group. Take both actors from bus.jpg, different instances,
    # and fall back only if that image somehow yields nothing.
    intruder = _extract_person(bus, model, skip=0)
    guard = _extract_person(bus, model, skip=1)
    if intruder is None:
        intruder = _extract_person(zidane, model, skip=0)
    if guard is None:
        guard = _extract_person(zidane, model, skip=1)
    if intruder is None:
        raise SystemExit(
            "No person found in the sample image, so the clip cannot be built. "
            "Supply your own footage instead - see docs/DATA.md."
        )

    ensure_dir(output.parent)
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        raise SystemExit(f"OpenCV could not open {output} for writing.")

    background = _background()
    total = FPS * DURATION

    for i in range(total):
        t = i / FPS
        frame = background.copy()

        if guard is not None:
            gx, gy = _guard_path(t)
            _paste(frame, guard, int(gx * WIDTH), int(gy * HEIGHT),
                   _perspective_height(gy))

        ix, iy = _intruder_path(t)
        _paste(frame, intruder, int(ix * WIDTH), int(iy * HEIGHT),
               _perspective_height(iy))

        writer.write(frame)
        if show_progress and i % (FPS * 3) == 0:
            print(f"  {t:5.1f}s / {DURATION}s")

    writer.release()

    print(f"\nPerimeter clip written to {output}")
    print(f"  {WIDTH}x{HEIGHT}, {FPS} fps, {DURATION}s, {total} frames")
    print(f"  fence line at y={FENCE_Y} "
          f"(must match the perimeter_fence zone profile)")
    print("\nExpected on CAM-02 with zones_profile: perimeter_fence")
    print("  ~7-9s   line crossing + restricted zone intrusion (intruder)")
    print("  ~17s    loitering, once 8s of dwell has accumulated")
    print("  never   any incident for the patrol guard - that is the control")
    print("\nThis is a synthetic composite for testing. The detections the "
          "application\nmakes on it are real, the scene is not.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data/input/sample_02.mp4"),
        help="Where to write the clip",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    build(Path(args.output), show_progress=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
