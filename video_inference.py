import argparse
import importlib.util
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import torch

from data.constant import tusimple_row_anchor
from model.model import parsingNet

CONFIG_FILENAME = Path(__file__).resolve().parent / "configs" / "tusimple.py"
DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "configs" / "tusimple_18.pth"

MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


def load_cfg(path: Path):
    spec = importlib.util.spec_from_file_location("tusimple_cfg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UFld lane detection on a video.")
    parser.add_argument("video", help="path to the input video")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="path to the trained `tusimple_18.pth` checkpoint",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_FILENAME,
        help="path to the config that describes the Tusimple backbone/griding settings",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ufld_output.mp4"),
        help="processed video path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="display the annotated frames while processing",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="skip this many frames before running inference",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="stop after this many frames (after start-frame)",
    )
    return parser.parse_args()


def load_weights(net: torch.nn.Module, path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    cleaned = {}
    for key, value in checkpoint.items():
        if key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    net.load_state_dict(cleaned, strict=False)


def preprocess(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    resized = cv2.resize(frame, (800, 288))
    tensor = torch.from_numpy(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)).float().permute(2, 0, 1) / 255.0
    tensor = (tensor - MEAN[:, None, None]) / STD[:, None, None]
    return tensor.unsqueeze(0).to(device)


def decode_output(out: torch.Tensor, griding_num: int) -> np.ndarray:
    lane_out = out[0, :, :, :]
    lane_loc = torch.argmax(lane_out, dim=0)
    prob = torch.softmax(lane_out[:-1, :, :], dim=0)
    idx = torch.arange(griding_num, device=out.device).view(-1, 1, 1)
    loc = (prob * idx).sum(dim=0)
    loc[lane_loc == griding_num] = griding_num
    return loc.cpu().numpy()


def draw_lanes(
    vis: np.ndarray, lane_locs: np.ndarray, row_anchor: Sequence[int], griding_num: int
) -> None:
    height, width = vis.shape[:2]
    y_scale = height / 288.0
    x_scale = width / max(griding_num - 1, 1)
    for lane_id in range(lane_locs.shape[1]):
        for row_id, loc in enumerate(lane_locs[:, lane_id]):
            if loc >= griding_num:
                continue
            x = int(round((loc + 0.5) * x_scale))
            y = int(round(row_anchor[row_id] * y_scale))
            if 0 <= x < width and 0 <= y < height:
                cv2.circle(vis, (x, y), 5, (0, 255, 0), -1)


def run_video(args: argparse.Namespace) -> None:
    if not args.video:
        raise FileNotFoundError(f"{args.video} not found")
    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = parsingNet(
        pretrained=False,
        backbone=cfg.backbone,
        cls_dim=(cfg.griding_num + 1, 56, cfg.num_lanes),
        use_aux=False,
    ).to(device)
    load_weights(net, args.weights)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {args.video}")
    writer = None
    frame_idx = -1
    processed = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1
        if frame_idx < args.start_frame:
            continue
        if args.max_frames is not None and processed >= args.max_frames:
            break
        tensor = preprocess(frame, device)
        with torch.no_grad():
            out = net(tensor)
            if isinstance(out, tuple):
                out = out[0]
            lane_locs = decode_output(out, cfg.griding_num)
        vis = frame.copy()
        draw_lanes(vis, lane_locs, tusimple_row_anchor, cfg.griding_num)
        if writer is None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            height, width = frame.shape[:2]
            writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        writer.write(vis)
        if args.show:
            cv2.imshow("UFld lanes", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        processed += 1
    cap.release()
    if writer:
        writer.release()
    if args.show:
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    run_video(args)
    print(args.video)


if __name__ == "__main__":
    main()
