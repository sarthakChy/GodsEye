"""REMIND tracker output adapter.

Reads REMIND's `detections.jsonl` (one record per processed frame, each
with a list of detections carrying a persistent `object_id`) and exposes
it as a drop-in replacement for the in-process YOLO + ByteTrack path.

REMIND's persistent `object_id` is used directly as the semantic ID suffix:
    det2semantic[i] = f"{class_name}_{object_id}"

This means a knife tracked by REMIND as object_id=3 becomes `knife_3` for
the entire video -- no registry re-ID needed. The Phase 0 registry layer is
bypassed entirely in this mode.
"""
from __future__ import annotations

import json
from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class RemindCache:
    """Per-frame detections from REMIND's detections.jsonl."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"REMIND JSONL not found: {self.path}")
        self._records: List[Tuple[float, dict]] = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._records.append((float(rec["timestamp"]), rec))
        self._records.sort(key=lambda x: x[0])
        self._ts = [t for t, _ in self._records]

    def __len__(self) -> int:
        return len(self._records)

    def _nearest(self, ts: float, tol: float = 0.06) -> Optional[dict]:
        if not self._ts:
            return None
        i = bisect_left(self._ts, ts)
        candidates = []
        if i < len(self._ts):
            candidates.append(i)
        if i > 0:
            candidates.append(i - 1)
        best_idx = None
        best_dt = tol
        for j in candidates:
            dt = abs(self._ts[j] - ts)
            if dt <= best_dt:
                best_idx = j
                best_dt = dt
        if best_idx is None:
            return None
        return self._records[best_idx][1]

    def det2semantic_for(self, ts: float, tol: float = 0.06):
        """Return (det2semantic, boxes, labels, confs) for the frame nearest
        `ts`, or None if no record is within tol seconds.

        det2semantic maps detection index -> f"{class_name}_{object_id}".
        Detections without an object_id (unassigned) are dropped entirely
        so downstream indices stay dense.
        """
        rec = self._nearest(ts, tol)
        if rec is None:
            return None

        dets = rec.get("detections") or []
        kept = [d for d in dets if d.get("object_id") is not None]
        n = len(kept)

        boxes = np.zeros((n, 4), dtype=float)
        labels: List[str] = []
        confs = np.zeros((n,), dtype=float)
        det2semantic: Dict[int, str] = {}

        for i, d in enumerate(kept):
            boxes[i] = d["bbox_xyxy"]
            labels.append(d["class_name"])
            confs[i] = float(d["confidence"])
            det2semantic[i] = f"{d['class_name']}_{int(d['object_id'])}"

        return det2semantic, boxes, labels, confs
