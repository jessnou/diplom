#!/usr/bin/env python3
"""
Capture specific frames from video for thesis visualization.
Headless — no PySide6 needed.

Usage examples:
  # Capture specific frames
  python capture_frames.py --video test.mp4 --frames 100 200 300

  # Auto-detect warning frames (LDW / FCW)
  python capture_frames.py --video test.mp4 --scan-warnings

  # Combine: specific frames + warnings
  python capture_frames.py --video test.mp4 --frames 100 200 --scan-warnings

  # Clean road view (no HUD panels)
  python capture_frames.py --video test.mp4 --frames 100 --no-hud

Output per frame:
  frame_NNNNNN_{tag}.jpg   — full annotated frame
  frame_NNNNNN_bev.jpg     — bird's-eye view
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adas_pipeline import ADASProcessor
from main_desktop import default_config
from TrafficLaneDetector.ufldDetector.utils import OffsetType, CurvatureType
from ObjectDetector.utils import CollisionType


WARN_LABELS = {
    OffsetType.LEFT: "LDW_LEFT",
    OffsetType.RIGHT: "LDW_RIGHT",
    CollisionType.WARNING: "FCW_WARNING",
    CollisionType.PROMPT: "FCW_PROMPT",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Capture ADAS frames for thesis visualization (headless)",
    )
    p.add_argument("--video", required=True, help="Path to video file")
    p.add_argument("--output", default="captured_frames",
                   help="Output directory (default: captured_frames)")
    p.add_argument("--frames", type=int, nargs="*", default=[],
                   help="Frame numbers to capture, e.g. --frames 100 200 300")
    p.add_argument("--no-hud", action="store_true",
                   help="Save clean road view without HUD panels")
    p.add_argument("--scan-warnings", action="store_true",
                   help="Auto-detect and save LDW / FCW warning frames")
    p.add_argument("--scan-every", type=int, default=3,
                   help="Scan every Nth frame (default: 3)")
    p.add_argument("--max-warnings", type=int, default=10,
                   help="Max warning frames per type (default: 10)")
    p.add_argument("--yolo-size", choices=["n", "s", "m", "l"], default="n",
                   help="YOLO model size (default: n)")
    p.add_argument("--downscale", type=float, default=1.0,
                   help="Inference downscale factor (default: 1.0)")
    p.add_argument("--bev-crop", type=float, default=0.15,
                   help="Hood crop ratio (default: 0.15)")
    return p.parse_args()


def build_config(args):
    cfg = default_config(yolo_size=args.yolo_size)
    cfg.bev_config.hood_crop_ratio = args.bev_crop
    return cfg


def process_frames(args):
    cfg = build_config(args)
    target_frames = set(args.frames)

    if not target_frames and not args.scan_warnings:
        print("ERROR: specify --frames or --scan-warnings")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Video: {args.video}")
    print(f"  size={width}x{height}  fps={fps:.1f}  total_frames={total_frames}")

    processor = ADASProcessor(
        cfg.lane_config, cfg.object_config,
        allowed_labels={"person", "car", "truck", "bus", "motorbike"},
        parallel=False,
        lane_skip_frames=0,
        downscale=args.downscale,
        bev_config=cfg.bev_config,
    )
    processor.initialize((width, height))

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    warning_counts = {label: 0 for label in WARN_LABELS.values()}
    max_w = args.max_warnings
    all_captured = set()
    metrics_log = []

    print(f"\nProcessing... (use --scan-every N to adjust scan density)\n")

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        want_user = frame_idx in target_frames
        want_scan = args.scan_warnings and (frame_idx % args.scan_every == 0)

        if not want_user and not want_scan:
            frame_idx += 1
            continue

        t0 = time.time()
        frame_show, metrics = processor.process_frame(frame, no_hud=args.no_hud)
        dt = time.time() - t0

        tags = []

        if want_user:
            tags.append("USER")

        if want_scan:
            if metrics.offset in (OffsetType.LEFT, OffsetType.RIGHT):
                label = WARN_LABELS[metrics.offset]
                if warning_counts[label] < max_w:
                    tags.append(label)
                    warning_counts[label] += 1

            if metrics.collision in (CollisionType.WARNING, CollisionType.PROMPT):
                label = WARN_LABELS[metrics.collision]
                if warning_counts[label] < max_w:
                    tags.append(label)
                    warning_counts[label] += 1

        if tags:
            tag = "+".join(tags)
            all_captured.add(frame_idx)

            path_full = out_dir / f"frame_{frame_idx:06d}_{tag}.jpg"
            cv2.imwrite(str(path_full), frame_show)

            if metrics.birdview is not None:
                path_bev = out_dir / f"frame_{frame_idx:06d}_bev.jpg"
                cv2.imwrite(str(path_bev), metrics.birdview)

            metrics_log.append((frame_idx, tag, metrics, dt))

            print(f"  [{frame_idx:>5}] {tag:<28} "
                  f"offset={metrics.offset.name:<8} curve={metrics.curvature.name:<14} "
                  f"collision={metrics.collision.name:<14} {dt:.3f}s")

        frame_idx += 1

        user_done = (not target_frames) or target_frames.issubset(all_captured)
        scan_done = (not args.scan_warnings) or all(v >= max_w for v in warning_counts.values())
        if user_done and scan_done:
            break

    cap.release()
    processor.cleanup()

    n = len(all_captured)
    print(f"\n{'='*60}")
    print(f"Saved {n} frames → {out_dir.resolve()}/")
    if metrics_log:
        print(f"\n{'Frame':>7} {'Tag':<28} {'Offset':<10} {'Curve':<16} {'Collision':<16} {'T':>8}")
        print("-" * 90)
        for idx, tag, m, dt in metrics_log:
            print(f"{idx:>7} {tag:<28} {m.offset.name:<10} {m.curvature.name:<16} {m.collision.name:<16} {dt:.3f}s")

    if args.scan_warnings:
        print(f"\nWarnings found:")
        for label, count in warning_counts.items():
            print(f"  {label}: {count}")


if __name__ == "__main__":
    process_frames(parse_args())
