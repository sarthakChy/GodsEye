#!/usr/bin/env python3
"""Run DEVA over a video, emitting stable track masks for downstream relation scoring."""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


class TracksSaver:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.masks_dir = out_dir / "masks"
        self.frames_dir = out_dir / "frames"
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.tracks = {}
        self._idx = 0

    def save_mask(self, prob, name, need_resize=False, shape=None,
                  image_np=None, prompts=None):
        if prob.dim() == 3:
            ids = prob.argmax(dim=0).cpu().numpy().astype(np.uint16)
        else:
            ids = prob.cpu().numpy().astype(np.uint16)

        if need_resize and shape is not None:
            h, w = shape
            ids = cv2.resize(ids, (w, h), interpolation=cv2.INTER_NEAREST)

        idx = self._idx
        cv2.imwrite(str(self.masks_dir / f"{idx:05d}.png"), ids)
        if image_np is not None:
            cv2.imwrite(str(self.frames_dir / f"{idx:05d}.jpg"),
                        cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

        frame_tracks = []
        for tid in np.unique(ids):
            tid = int(tid)
            if tid == 0:
                continue
            ys, xs = np.where(ids == tid)
            frame_tracks.append({
                "track_id": tid,
                "box": [int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max())],
                "area": int(len(xs)),
            })
        self.tracks[idx] = frame_tracks
        self._idx += 1

    def end(self):
        pass

    def __getattr__(self, name):
        def _noop(*_a, **_k):
            pass
        return _noop

    def write_meta(self, video: Path, fps: float, n_frames: int):
        (self.out_dir / "tracks.json").write_text(
            json.dumps({str(k): v for k, v in self.tracks.items()}, indent=2))
        all_ids = {t["track_id"] for v in self.tracks.values() for t in v}
        (self.out_dir / "meta.json").write_text(json.dumps({
            "video": str(video), "fps": fps, "n_frames": n_frames,
            "n_tracks_total": len(all_ids),
        }, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--deva_repo", type=Path,
                    default=Path.home() / "Desktop/Tracking-Anything-with-DEVA")
    ap.add_argument("--sam_variant", default="mobile", choices=["mobile", "original"])
    ap.add_argument("--detection_every", type=int, default=8)
    ap.add_argument("--temporal_setting", default="online", choices=["online", "semionline"])
    ap.add_argument("--max_num_objects", type=int, default=50)
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--sam_points_per_side", type=int, default=16)
    ap.add_argument("--sam_points_per_batch", type=int, default=16)
    a = ap.parse_args()

    if not a.deva_repo.exists():
        sys.exit(f"DEVA repo not found at {a.deva_repo}")
    sys.path.insert(0, str(a.deva_repo))

    from argparse import ArgumentParser
    from deva.inference.eval_args import add_common_eval_args, get_model_and_config
    from deva.ext.ext_eval_args import add_ext_eval_args, add_auto_default_args

    parser = ArgumentParser()
    add_common_eval_args(parser)
    add_ext_eval_args(parser)
    add_auto_default_args(parser)

    deva_argv = [
        "--img_path", str(a.video),
        "--sam_variant", a.sam_variant,
        "--detection_every", str(a.detection_every),
        "--temporal_setting", a.temporal_setting,
        "--max_num_objects", str(a.max_num_objects),
        "--size", str(a.size),
        "--SAM_NUM_POINTS_PER_SIDE", str(a.sam_points_per_side),
        "--SAM_NUM_POINTS_PER_BATCH", str(a.sam_points_per_batch),
    ]
    if torch.cuda.is_available():
        deva_argv.append("--amp")

    _orig_argv = sys.argv
    sys.argv = ["run_deva.py"] + deva_argv
    try:
        deva_model, cfg, args = get_model_and_config(parser)
    finally:
        sys.argv = _orig_argv
    cfg["temporal_setting"] = a.temporal_setting
    cfg["enable_long_term_count_usage"] = True
    cfg["max_num_objects"] = a.max_num_objects
    cfg["size"] = a.size
    cfg["amp"] = torch.cuda.is_available()
    cfg["detection_every"] = a.detection_every
    cfg["sam_variant"] = a.sam_variant

    from deva.inference.inference_core import DEVAInferenceCore
    from deva.inference.demo_utils import flush_buffer
    from deva.ext.automatic_sam import get_sam_model
    from deva.ext.automatic_processor import process_frame_automatic

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[deva] device={device} sam={a.sam_variant} "
          f"det_every={a.detection_every} mode={a.temporal_setting}")

    sam_model = get_sam_model(cfg, device)
    deva = DEVAInferenceCore(deva_model, config=cfg)
    deva.next_voting_frame = cfg["num_voting_frames"] - 1
    deva.enabled_long_id()

    a.out.mkdir(parents=True, exist_ok=True)
    saver = TracksSaver(a.out)

    cap = cv2.VideoCapture(str(a.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[deva] {a.video.name} — {n_frames} frames @ {fps:.1f} fps")

    ti = 0
    with torch.cuda.amp.autocast(enabled=cfg["amp"]):
        with tqdm(total=n_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                process_frame_automatic(
                    deva, sam_model, "null.png", saver, ti,
                    image_np=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ti += 1
                pbar.update(1)
        flush_buffer(deva, saver)
    cap.release()
    saver.write_meta(a.video, fps, n_frames)
    print(f"[deva] wrote {ti} frames -> {a.out}")


if __name__ == "__main__":
    main()
