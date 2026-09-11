#!/usr/bin/env python3
"""Build a synthetic demo clip so the application can be tested immediately.

WHAT THIS IS
------------
A composited test clip, not real border footage. Photographic cut-outs of a
person and a vehicle (from the sample images that ship with Ultralytics) are
moved across a plain synthetic background along a scripted path.

The detections the application makes on this clip are **real**: YOLO genuinely
finds the person and the vehicle, the tracker genuinely follows them, and the
zone, loitering and threat logic genuinely fire. Only the scene is synthetic.

WHY IT EXISTS
-------------
So that a fresh clone can be verified end to end without first hunting for
footage. It is a test fixture. Never present output from this clip as a
real-world result, and never quote accuracy figures measured on it.

For anything meaningful, use your own footage. See docs/DATA.md.

USAGE
-----
    python scripts/make_demo_video.py
    python scripts/make_demo_video.py --output data/input/demo_border.mp4

The scripted scenario, matched to the default zone profile:

    0-3s    person walks up from the bottom of the frame
    3-4s    person crosses the Forward Boundary line
    4-16s   person stands still inside the Restricted Compound  -> loitering
    6-14s   a vehicle drives through the Vehicle Exclusion Area
    16-20s  person walks on into the Border Strip
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
DURATION = 20


def _ultralytics_assets() -> Path | None:
    try:
        import ultralytics
        assets = Path(ultralytics.__file__).parent / "assets"
        return assets if assets.is_dir() else None
    except ImportError:
        return None


def _extract_sprite(image: np.ndarray, want: str, model) -> np.ndarray | None:
    """Crop the highest-confidence instance of *want* out of an image."""
    results = model.predict(image, conf=0.35, verbose=False)
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None

    names = model.names
    best = None
    best_conf = 0.0
    for box in results[0].boxes:
        label = names.get(int(box.cls[0]), "")
        if label != want:
            continue
        conf = float(box.conf[0])
        if conf > best_conf:
            best_conf = conf
            best = box.xyxy[0].tolist()

    if best is None:
        return None
    x1, y1, x2, y2 = (int(round(v)) for v in best)
    crop = image[max(0, y1):y2, max(0, x1):x2]
    return crop if crop.size else None


def _background() -> np.ndarray:
    """A plain outdoor-looking scene: sky band, ground, a track and a fence."""
    frame = np.zeros((HEIGHT, WIDTH, 3), np.uint8)

    # Vertical gradient from a hazy sky to dry ground.
    for y in range(HEIGHT):
        blend = y / HEIGHT
        top = np.array([150, 140, 120], np.float32)
        bottom = np.array([70, 92, 112], np.float32)
        frame[y, :] = (top * (1 - blend) + bottom * blend).astype(np.uint8)

    # Fence posts along the border strip.
    for x in range(20, WIDTH, 70):
        cv2.line(frame, (x, 70), (x, 105), (95, 105, 120), 3)
    cv2.line(frame, (0, 78), (WIDTH, 78), (95, 105, 120), 2)
    cv2.line(frame, (0, 96), (WIDTH, 96), (95, 105, 120), 2)

    # A dirt track for the vehicle.
    cv2.rectangle(frame, (0, 300), (260, HEIGHT), (78, 96, 118), -1)

    # Texture, so optical-flow-like measures are not perfectly uniform.
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 5, (HEIGHT, WIDTH, 3))
    return np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _paste(canvas: np.ndarray, sprite: np.ndarray,
           cx: int, cy: int, height_px: int) -> None:
    """Paste *sprite* scaled to *height_px*, centred on (cx, cy)."""
    if sprite is None or sprite.size == 0 or height_px < 8:
        return
    scale = height_px / sprite.shape[0]
    w = max(4, int(sprite.shape[1] * scale))
    resized = cv2.resize(sprite, (w, height_px), interpolation=cv2.INTER_AREA)

    x1 = int(cx - w / 2)
    y1 = int(cy - height_px / 2)
    x2, y2 = x1 + w, y1 + height_px

    sx1, sy1 = max(0, -x1), max(0, -y1)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(canvas.shape[1], x2), min(canvas.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    canvas[y1:y2, x1:x2] = resized[sy1:sy1 + (y2 - y1), sx1:sx1 + (x2 - x1)]


def build(output: Path, show_progress: bool = True) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "ultralytics is required to build the demo clip.\n"
            "Run: pip install -r requirements.txt"
        )

    assets = _ultralytics_assets()
    if assets is None:
        raise SystemExit("Could not locate the Ultralytics sample images.")

    weights = PROJECT_ROOT / "models" / "yolov8n.pt"
    model = YOLO(str(weights) if weights.exists() else "yolov8n.pt")

    person_src = cv2.imread(str(assets / "zidane.jpg"))
    vehicle_src = cv2.imread(str(assets / "bus.jpg"))
    if person_src is None or vehicle_src is None:
        raise SystemExit("Could not read the Ultralytics sample images.")

    person = _extract_sprite(person_src, "person", model)
    vehicle = _extract_sprite(vehicle_src, "bus", model)
    second_person = _extract_sprite(vehicle_src, "person", model)

    if person is None:
        raise SystemExit(
            "No person was found in the sample image, so the demo clip "
            "cannot be built. Supply your own footage instead - see docs/DATA.md."
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

        # --- vehicle: drives up the dirt track, 6s to 14s ---------------
        if vehicle is not None and 6.0 <= t <= 14.0:
            progress = (t - 6.0) / 8.0
            vy = int(HEIGHT - 40 - progress * (HEIGHT - 200))
            _paste(frame, vehicle, cx=130, cy=vy, height_px=140)

        # --- second person: crosses left to right, 8s to 18s ------------
        if second_person is not None and 8.0 <= t <= 18.0:
            progress = (t - 8.0) / 10.0
            px = int(360 + progress * 480)
            _paste(frame, second_person, cx=px, cy=430, height_px=100)

        # --- primary person: approach, loiter, then enter the border ----
        if t < 3.0:                       # walking up towards the line
            progress = t / 3.0
            py = int(500 - progress * 200)
            person_h = int(150 - progress * 25)
        elif t < 4.0:                     # crossing the boundary line
            progress = (t - 3.0)
            py = int(300 - progress * 60)
            person_h = 125
        elif t < 16.0:                    # standing still: loitering
            py = 240 + int(3 * np.sin(t * 2.0))   # small idle sway
            person_h = 122
        else:                             # moving on into the border strip
            progress = (t - 16.0) / 4.0
            py = int(240 - progress * 150)
            person_h = int(122 - progress * 30)

        _paste(frame, person, cx=520, cy=py, height_px=person_h)

        writer.write(frame)
        if show_progress and i % (FPS * 2) == 0:
            print(f"  {t:4.1f}s / {DURATION}s")

    writer.release()
    print(f"\nDemo clip written to {output}")
    print(f"  {WIDTH}x{HEIGHT}, {FPS} fps, {DURATION}s, {total} frames")
    print("\nThis is a synthetic composite for testing. The detections the "
          "application makes on it are real, the scene is not.")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "data/input/demo_border.mp4"),
        help="Where to write the clip",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    build(Path(args.output), show_progress=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
