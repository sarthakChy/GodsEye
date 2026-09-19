"""Convert live styled-app frames into a GodsEye run directory.

Takes the per-frame boxes, labels, and triplets collected by run_video and
writes outputs/<run_name>/{manifest,scene_graph,events}.json in the format
temporal/viz/dashboard_ui.py expects.
"""
from __future__ import annotations
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import networkx as nx


def _group_intervals(observations, gap_sec=1.0, min_frames=2, min_duration=0.5):
    """observations: list of (ts, score). Returns list of (start, end, mean, n)."""
    if len(observations) < min_frames:
        return []
    obs = sorted(observations, key=lambda x: x[0])
    groups = []
    cur = [obs[0]]
    for o in obs[1:]:
        if o[0] - cur[-1][0] <= gap_sec:
            cur.append(o)
        else:
            groups.append(cur)
            cur = [o]
    groups.append(cur)

    out = []
    for g in groups:
        if len(g) < min_frames:
            continue
        dur = g[-1][0] - g[0][0]
        if dur < min_duration:
            continue
        mean_sc = sum(x[1] for x in g) / len(g)
        out.append((g[0][0], g[-1][0], mean_sc, len(g)))
    return out


def save_run(
    video_path: str,
    sample_fps: float,
    frames_records: list[dict[str, Any]],
    score_threshold: float = 0.30,
    outputs_dir: str = "outputs",
) -> str:
    """Write a GodsEye-format run. Returns the run name."""
    # Name the run after the source video plus a short timestamp, so the
    # dashboard dropdown reads "homer_chopping_20260919_1745" instead of a
    # bare epoch. Sanitise the stem to keep the name filesystem-safe.
    stem = Path(video_path).stem
    safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in stem)[:40]
    stamp = time.strftime("%Y%m%d_%H%M")
    run_name = f"{safe}_{stamp}"
    run_dir = Path(outputs_dir) / run_name
    # If a run with this name exists (processed same minute), suffix with -N
    n = 2
    while run_dir.exists():
        run_name = f"{safe}_{stamp}_{n}"
        run_dir = Path(outputs_dir) / run_name
        n += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---- manifest ----
    manifest = {
        "video_path": str(Path(video_path).resolve()),
        "sample_fps": float(sample_fps),
        "frames": [],
    }
    for rec in frames_records:
        det2sem = {}
        for i, lab in enumerate(rec["labels"]):
            det2sem[i] = f"{lab}_{i}"
        manifest["frames"].append({
            "idx": int(rec["idx"]),
            "ts": float(rec["ts"]),
            "boxes": [[float(v) for v in b] for b in rec["boxes"]],
            "det2semantic": det2sem,
        })
    manifest["total_frames"] = len(frames_records)
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- aggregate temporal relations (class-level) ----
    history: dict[tuple[str, str, str], list[tuple[float, float]]] = defaultdict(list)
    for rec in frames_records:
        for s, p, o, sc in rec["triplets"]:
            if float(sc) < score_threshold:
                continue
            if s >= len(rec["labels"]) or o >= len(rec["labels"]):
                continue
            subj = rec["labels"][s]
            obj = rec["labels"][o]
            if subj == obj:                # drop self-loops
                continue
            if s == o:                     # drop same-detection edges
                continue
            history[(subj, str(p), obj)].append((float(rec["ts"]), float(sc)))

    events = []
    for (subj, pred, obj), obs in history.items():
        for start, end, mean_sc, n in _group_intervals(obs):
            events.append({
                "subject_id": subj,
                "predicate": pred,
                "object_id": obj,
                "start_time": start,
                "end_time": end,
                "mean_score": mean_sc,
                "frame_count": n,
                "confidence": mean_sc,
            })
    events.sort(key=lambda e: e["start_time"])
    (run_dir / "events.json").write_text(json.dumps(events, indent=2))

    # ---- scene graph ----
    G = nx.MultiDiGraph()
    for ev in events:
        for nid, cls in (
            (ev["subject_id"], ev["subject_id"]),
            (ev["object_id"], ev["object_id"]),
        ):
            if nid not in G:
                G.add_node(
                    nid,
                    class_name=cls,
                    first_seen=ev["start_time"],
                    last_seen=ev["end_time"],
                )
        G.add_edge(
            ev["subject_id"],
            ev["object_id"],
            predicate=ev["predicate"],
            start_time=ev["start_time"],
            end_time=ev["end_time"],
            score=ev["mean_score"],
            frames=ev["frame_count"],
        )
    (run_dir / "scene_graph.json").write_text(
        json.dumps(nx.node_link_data(G), indent=2)
    )

    print(f"[live_to_godseye] wrote {len(events)} events to {run_dir}")
    return run_name
