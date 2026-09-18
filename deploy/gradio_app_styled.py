"""RelateAnything live demo — styled Gradio app with manual box support.

    python deploy/gradio_app_styled.py --device cpu \
        --ckpt <model.pth> --det checkpoints/detectors/yoloe-11m-seg-pf.pt

Both models re-parameterize live:
  object classes -> YOLOE text prompts
  predicates     -> relation head vocabulary, via the checkpoint's own student

Manual mode: tick "use my boxes", then either draw rectangles on the image
(tick "draw on image" first) or paste x1,y1,x2,y2 lines. The detector is
bypassed entirely.
"""

from __future__ import annotations

import argparse
import html
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
from deploy.pipeline import ParallelScenePipeline, PipelineConfig, _default_predicates


DEFAULT_CLASSES = [
    "person", "face", "hand", "laptop", "keyboard", "mouse", "monitor", "cup",
    "bottle", "phone", "book", "chair", "desk", "backpack", "hat", "glasses",
    "plant", "door", "window", "picture", "lamp", "bag",
]

PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (138, 99, 210),
           (214, 168, 42), (52, 187, 235), (200, 62, 120), (120, 160, 60)]

COLOR_NAMES = ["blue", "orange", "green", "purple", "yellow", "cyan", "pink", "olive"]


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    for cand in (here, here.parent):
        if (cand / "assets").is_dir():
            return cand
    return here


REPO_ROOT = _repo_root()

SAMPLE_IMAGES = [
    REPO_ROOT / "assets" / "reel" / "images" / n
    for n in ("bicycle.jpg", "catlaptop.jpg", "frisbee.jpg",
              "horse.jpg", "skateboard.jpg", "tennis.jpg")
]
SAMPLE_IMAGES = [str(p) for p in SAMPLE_IMAGES if p.is_file()]

PIPE: ParallelScenePipeline | None = None

SPATIAL_EDGE = (120, 200, 255)
SEMANTIC_EDGE = (140, 255, 170)
MERGED_EDGE = (255, 255, 255)


# ---------------------------------------------------------------- helpers

def _color(i: int):
    return PALETTE[i % len(PALETTE)]


def _object_name(res, i: int, label_mode: str = "names") -> str:
    if label_mode == "colours":
        return f"{COLOR_NAMES[i % len(COLOR_NAMES)]} object {i + 1}"
    return res.labels[i] if i < len(res.labels) else f"object {i + 1}"


def _predicate_kind(pred: str) -> str:
    """Spatial vs semantic for one predicate, using the live vocabulary."""
    if PIPE is None:
        return "semantic"
    try:
        i = PIPE.ra.predicates.index(pred)
        return "spatial" if bool(PIPE.is_spatial[i]) else "semantic"
    except Exception:
        return "semantic"


def _parse_boxes(text: str, W: int, H: int):
    """'x1,y1,x2,y2' per line -> [N,4] pixel xyxy. Values <=1.5 are normalised."""
    lines = [l.strip() for l in (text or "").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return None, "no boxes given"
    rows = []
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) != 4:
            return None, f"line {i+1}: expected 4 numbers, got {len(parts)}"
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            return None, f"line {i+1} is not four numbers"
    arr = np.asarray(rows, np.float32)
    if float(arr.max()) <= 1.5:
        arr = arr * np.array([W, H, W, H], np.float32)
    arr[:, [0, 2]] = arr[:, [0, 2]].clip(0, W - 1)
    arr[:, [1, 3]] = arr[:, [1, 3]].clip(0, H - 1)
    return arr, None


# ---------------------------------------------------------------- render

def render(res, show_masks: bool, show_labels: bool,
           mode: str = "both", label_mode: str = "names") -> np.ndarray:
    img = res.frame.copy()
    H, W = img.shape[:2]

    if show_masks and getattr(res, "masks", None) is not None and len(res.masks):
        overlay = img.copy()
        for i, m in enumerate(res.masks[:len(res.boxes_xyxy)]):
            if m.shape[:2] != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[m] = _color(i)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    centers = []
    for i, bb in enumerate(res.boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bb]
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), _color(i), 2)
        if show_labels:
            lab = _object_name(res, i, label_mode)
            sc = float(res.scores[i]) if i < len(res.scores) else 1.0
            txt = f"{i}:{lab}" + (f" {sc:.2f}" if sc < 0.99 else "")
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                          _color(i), -1)
            cv2.putText(img, txt, (x1 + 2, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)

    if mode == "merged":
        streams = [(res.triplets, MERGED_EDGE)]
    elif mode == "spatial":
        streams = [(getattr(res, "triplets_spatial", []) or [], SPATIAL_EDGE)]
    elif mode == "semantic":
        streams = [(getattr(res, "triplets_semantic", []) or [], SEMANTIC_EDGE)]
    else:
        streams = [(getattr(res, "triplets_spatial", []) or [], SPATIAL_EDGE),
                   (getattr(res, "triplets_semantic", []) or [], SEMANTIC_EDGE)]

    for si, (trips, edge_col) in enumerate(streams):
        off = 0 if len(streams) == 1 else (6 if si == 0 else -6)
        for s, pred, o, sc in trips:
            if s >= len(centers) or o >= len(centers):
                continue
            p1, p2 = centers[s], centers[o]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            n = max(1.0, (dx * dx + dy * dy) ** 0.5)
            ox, oy = int(-dy / n * off), int(dx / n * off)
            q1, q2 = (p1[0] + ox, p1[1] + oy), (p2[0] + ox, p2[1] + oy)
            cv2.arrowedLine(img, q1, q2, edge_col, max(1, int(1 + 3 * sc)),
                            cv2.LINE_AA, tipLength=0.03)
            mid = ((q1[0] + q2[0]) // 2, (q1[1] + q2[1]) // 2)
            (tw, th), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (mid[0] - 2, mid[1] - th - 4),
                          (mid[0] + tw + 4, mid[1] + 3), (30, 30, 30), -1)
            cv2.putText(img, pred, (mid[0] + 1, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, edge_col, 1, cv2.LINE_AA)

    t = res.timing
    hud = (f"{t.fps:5.1f} FPS | det {t.det:5.1f} | backbone {t.backbone:5.1f} "
           f"| rel {t.relation:5.1f} | total {t.total:5.1f} ms")
    cv2.rectangle(img, (0, 0), (W, 24), (20, 20, 20), -1)
    cv2.putText(img, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 230, 160), 1, cv2.LINE_AA)
    return img[:, :, ::-1]


# ---------------------------------------------------------------- panels

def _relations_panel(groups, res, label_v, labels_override=None):
    """groups: [(title, trips, edge_kind)]; edge_kind 'auto' colours per edge."""
    def name(i):
        if labels_override is not None:
            return labels_override[i]
        return _object_name(res, i, label_v)

    total = sum(len(t) for _, t, _ in groups)
    out = [f'<div class="relations-header"><span>Predictions</span>'
           f'<span class="relation-count">{total}</span></div>']
    for title, trips, edge_kind in groups:
        cls = "" if edge_kind == "auto" else edge_kind
        out.append(f'<div class="relation-section-title {cls}">{html.escape(title)}</div>')
        if not trips:
            out.append('<div class="empty-relations">No relations above threshold.</div>')
            continue
        for s, pred, o, sc in trips:
            kind = _predicate_kind(pred) if edge_kind == "auto" else edge_kind
            pct = max(0, min(100, int(float(sc) * 100)))
            out.append(
                '<div class="relation-card"><div class="relation-text">'
                f'<span class="entity">{html.escape(str(name(s)))}</span>'
                f'<span class="predicate {kind}">{html.escape(str(pred))}</span>'
                f'<span class="entity">{html.escape(str(name(o)))}</span>'
                '</div><div class="relation-score"><div class="score-track">'
                f'<div class="score-fill {kind}" style="width:{pct}%"></div>'
                f'</div><span>{float(sc):.2f}</span></div></div>'
            )
    return "".join(out)


def _stats_panel(res, n_objects=None):
    t = res.timing
    n = n_objects if n_objects is not None else len(res.boxes_xyxy)
    det = "—" if t.det < 0.01 else f"{t.det:.1f}"
    bb = "—" if t.backbone < 0.01 else f"{t.backbone:.1f}"
    return (
        '<div class="stats-strip">'
        f'<div><strong>{t.fps:.1f}</strong><span>FPS</span></div>'
        f'<div><strong>{det}</strong><span>det ms</span></div>'
        f'<div><strong>{bb}</strong><span>backbone ms</span></div>'
        f'<div><strong>{t.relation:.1f}</strong><span>rel ms</span></div>'
        f'<div><strong>{t.total:.1f}</strong><span>total ms</span></div>'
        f'<div><strong>{n}</strong><span>objects</span></div>'
        '</div>'
    )


def _error_panel(msg):
    return (f'<div class="relations-header"><span>Predictions</span>'
            f'<span class="relation-count">0</span></div>'
            f'<div class="empty-relations">{html.escape(msg)}</div>')


# ---------------------------------------------------------------- runners

def _clean(trips, score_thr, top_k):
    keep = [(int(s), str(p), int(o), float(sc)) for s, p, o, sc in trips
            if float(sc) >= float(score_thr)]
    return keep[:int(top_k)]


def run_frame(frame_rgb, conf_v, top_k_v, score_v, label_v):
    """Detector path."""
    if frame_rgb is None or PIPE is None:
        return None, "", "", ""
    PIPE.cfg.det_conf = float(conf_v)
    res = PIPE(frame_rgb[:, :, ::-1].copy(), top_k=int(top_k_v),
               score_thr=float(score_v), decompose=True,
               spatial_drop_pair=False)

    def cln(trips):
        return _clean(trips, score_v, top_k_v)

    merged = cln(res.triplets)
    spa = cln(getattr(res, "triplets_spatial", []) or [])
    sem = cln(getattr(res, "triplets_semantic", []) or [])
    groups_all = [("all relations", merged, "auto")]
    groups_spa = [("layout · spatial", spa, "spatial")]
    groups_sem = [("content · semantic", sem, "semantic")]
    return (
        render(res, True, True, "both", label_v),
        _relations_panel(groups_all, res, label_v),
        _relations_panel(groups_spa, res, label_v),
        _relations_panel(groups_sem, res, label_v),
    )


def run_manual(frame_rgb, boxes_text, top_k_v, score_v, label_v):
    """Skip the detector: relations between exactly the boxes the user drew."""
    if frame_rgb is None or PIPE is None:
        return None, "", "", ""
    H, W = frame_rgb.shape[:2]
    boxes, err = _parse_boxes(boxes_text, W, H)
    if err is not None:
        msg = _error_panel(err)
        return None, msg, msg, msg
    if len(boxes) < 2:
        msg = _error_panel("Need at least two boxes.")
        return None, msg, msg, msg

    bgr = frame_rgb[:, :, ::-1].copy()
    t0 = time.perf_counter()
    graphs = PIPE.ra.predict(bgr, boxes, topk=int(top_k_v),
                             max_boxes=len(boxes), decompose=True)
    dt = (time.perf_counter() - t0) * 1000.0

    def trips(g):
        return [(t.subject_idx, t.predicate, t.object_idx, float(t.score))
                for t in g]

    spa_all = _clean(trips(graphs.get("spatial", [])), score_v, top_k_v)
    sem_all = _clean(trips(graphs.get("semantic", [])), score_v, top_k_v)
    merged, seen = [], set()
    for t in sorted(spa_all + sem_all, key=lambda x: -x[3]):
        k = (t[0], t[1], t[2])
        if k not in seen:
            seen.add(k)
            merged.append(t)
    merged = merged[:int(top_k_v)]

    class _Res:
        pass
    res = _Res()
    res.frame = bgr
    res.boxes_xyxy = boxes
    res.labels = [f"box {i+1}" for i in range(len(boxes))]
    res.scores = np.ones(len(boxes), np.float32)
    res.masks = None
    res.triplets = merged
    res.triplets_spatial = spa_all
    res.triplets_semantic = sem_all

    class _T:
        pass
    t = _T()
    t.fps = 1000.0 / max(dt, 1e-3)
    t.det = 0.0
    t.backbone = 0.0
    t.relation = dt
    t.total = dt
    res.timing = t

    return (
        render(res, False, True, "both", label_v),
        _relations_panel([("all relations", merged, "auto")], res, label_v),
        _relations_panel([("layout · spatial", spa_all, "spatial")], res, label_v),
        _relations_panel([("content · semantic", sem_all, "semantic")], res, label_v),
    )


def apply_vocab(classes_txt, preds_txt):
    if PIPE is None:
        return "pipeline not ready"
    msgs = []
    try:
        cls = [c.strip() for c in classes_txt.replace("\n", ",").split(",") if c.strip()]
        if cls:
            PIPE.set_object_classes(cls)
            msgs.append(f"detector → {len(cls)} classes")
    except Exception as e:
        msgs.append(f"detector FAILED: {e}")
    try:
        prs = [p.strip() for p in preds_txt.replace("\n", ",").split(",") if p.strip()]
        if prs:
            PIPE.set_predicates(prs)
            msgs.append(f"relations → {len(prs)} predicates")
    except Exception as e:
        msgs.append(f"relations FAILED: {e}")
    return " · ".join(msgs)


# ---------------------------------------------------------------- presets

PREDICATE_PRESETS = {
    "Default": [p.strip() for p in (
        "wearing, riding, playing, sitting on, sitting at, holding, sitting in, "
        "looking at, using, watching, standing on, carrying, talking to, smiling at, "
        "standing beside, walking past, posing with, leaning against, part of, "
        "resting on, on, covering, inside, on top of, contained in, hanging from, "
        "surrounding, attached to, in front of, beside, to the left of, to the right "
        "of, behind, above, below"
    ).split(",")],
    "Spatial": ["on", "inside", "on top of", "in front of", "beside",
                "to the left of", "to the right of", "behind", "above", "below",
                "next to", "near", "under", "over", "underneath"],
    "Semantic": [p.strip() for p in (
        "wearing, riding, playing, sitting on, sitting at, holding, sitting in, "
        "looking at, using, watching, standing on, carrying, talking to, smiling at, "
        "standing beside, walking past, posing with, leaning against, part of, "
        "resting on, covering, contained in, hanging from, surrounding, attached to, "
        "supporting, containing, worn by, illuminating, decorating, resting in, "
        "shading, accompanying, touching, standing behind, resting against, mounted "
        "on, driving past, forming part of, casting shadow on, standing by, standing "
        "in front of, parked near, walking through, growing in, depicting, tucked "
        "under, reflecting, singing into, following, covering head of, standing in, "
        "operating, held by, standing near, comprising, hanging on, embracing, "
        "facing, occupying, passing, pointing at, eating, parked on, towering over, "
        "riding in, parked behind, steering, lying on, growing from, eating from, "
        "hugging, framing, pulling, floating in, parked beside, topping, displaying, "
        "covering eyes of, growing near, speaking into, working at, filling, "
        "driving, obscuring, walking towards, drinking from, swimming in, leaning "
        "over, walking on, serving, reflecting in, leaning on, flying over, cutting, "
        "looking towards, climbing, reaching for, sitting beside, driving on, "
        "pushing, looking through, parked in, mounting, standing next to, showing, "
        "contains, parked in front of, writing on, carried by, playing with, stored "
        "in, depicted in, interacting with, reaching towards, walking across, "
        "casting light on, enclosing, encasing, leading, walking with, controlling, "
        "shaking hands with, recording, driving along, gripping, standing among, "
        "parked by, resting inside, smiling with, running past, grazing in, kissing, "
        "capturing sound from, attaching to, grazing near, blocking, connected to, "
        "reading, working near, holding hands with, encircling, driving through, "
        "pedaling, listening to, running across, depicted on, garnishing, preparing, "
        "having, resting near, talking into, appearing in, gesturing towards, "
        "manipulating, underlying, standing under, posing in front of, featuring, "
        "hitting, embedded in, tucked into, swinging, displaying content for, "
        "looking past, accommodating, laughing with, posing for, feeding, walking "
        "along, floating in water near, fastening, approaching, forming, petting, "
        "cushioning, dancing with, jumping over, handling, contained within, "
        "photographing, perching on, casting light upon, housing, sitting near, "
        "piercing, kicking, bordering, decorated with, resting beside, working on, "
        "stepping on, performing with, typing on, dipping into, traveling along, "
        "performing near, incorporating, lying in, grazing on, running towards, "
        "emitting sound for, striking, blooming from, reflected in, throwing, "
        "observing, looking into, sitting behind, topped with, integrated into, "
        "growing among, resting under, paddling, assisting, growing in front of, "
        "posing for photo with, growing beside, kneeling on, bending over, standing "
        "with, reaching toward, clinging to, cooling, filming, lining, sitting by, "
        "amplifying sound for, juggling with, galloping on, repairing, pouring, "
        "stirring, waiting for, parked next to, chasing, tied to, growing on, "
        "covered with, filled with, grilling, flipping, seasoning, balancing on, "
        "trotting on, guarding, crossing, queuing at"
    ).split(",")],
    "Minimal": ["holding", "wearing", "riding", "sitting on",
                "on", "in front of", "behind", "next to"],
}

# "All 263" = Semantic (248) + Spatial (15), deduped, order-preserving. Defined
# by combination rather than pasted so the two source lists stay the only
# place the strings live. The reorder below puts it first in the button row.
PREDICATE_PRESETS["All 263"] = list(dict.fromkeys(
    PREDICATE_PRESETS["Semantic"] + PREDICATE_PRESETS["Spatial"]
))
PREDICATE_PRESETS = {k: PREDICATE_PRESETS[k] for k in
                     ("All 263", "Default", "Spatial", "Semantic", "Minimal")}


def _preset_handler(name):
    def handler(classes_txt_value):
        text = ", ".join(PREDICATE_PRESETS[name])
        return text, apply_vocab(classes_txt_value, text)
    return handler


# ---------------------------------------------------------------- css / js

CSS = r"""
.gradio-container {max-width:100% !important;background:#fff !important;
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif !important}
#workspace {gap:20px !important;align-items:flex-start !important}
#main-panel {min-width:0;padding:0 !important}
#side-panel {min-width:350px;max-width:395px;padding:14px !important;
  background:#0e151d !important;border:1px solid #1e2935 !important;
  border-radius:12px !important;color:#dce4ed !important}
#source-tabs {background:#0b1017 !important;border:1px solid #151d27 !important;
  border-radius:11px 11px 0 0 !important;overflow:hidden !important}
#source-tabs .tab-nav {background:#0b1017 !important;
  border-bottom:1px solid #202b37 !important;padding:0 4px !important}
#source-tabs .tab-nav button {color:#8e9bab !important;font-size:12px !important;
  font-weight:650 !important;border:0 !important;padding:10px 18px !important}
#source-tabs .tab-nav button.selected {color:#fff !important;
  border-bottom:2px solid #67d396 !important}
#source-tabs img,#source-tabs video {background:#0b1017 !important;
  object-fit:contain !important;max-height:440px !important}
#scene-output {background:#0b1017 !important;border:1px solid #151d27 !important;
  border-radius:0 !important;overflow:hidden !important;min-height:420px !important}
#scene-output img {background:#0b1017 !important;object-fit:contain !important}
#prediction-panel {background:#0b1017 !important;border:1px solid #151d27 !important;
  border-top:0 !important;border-radius:0 0 11px 11px !important;overflow:hidden !important}
#prediction-tabs {background:#0b1017 !important}
#prediction-tabs .tab-nav {background:#0b1017 !important;
  border-bottom:1px solid #202b37 !important;padding:0 8px !important}
#prediction-tabs .tab-nav button {background:transparent !important;border:0 !important;
  color:#7f8c9c !important;font-size:12px !important;font-weight:650 !important;
  padding:10px 14px !important}
#prediction-tabs .tab-nav button.selected {color:#fff !important}
#relations-merged,#relations-spatial,#relations-semantic {
  background:#0b1017;color:#dce4ed;padding:5px 14px 12px;max-height:350px;overflow:auto}
.relations-header {display:flex;justify-content:space-between;align-items:center;
  font-size:12px;font-weight:750;color:#f3f6f9;margin-bottom:5px}
.relation-count {background:#1b2632;color:#9aa8b8;border-radius:999px;min-width:22px;
  height:22px;display:grid;place-items:center;font-size:10px}
.relation-section-title {font-size:9px;font-weight:800;text-transform:uppercase;
  letter-spacing:.09em;color:#7f8c9c;margin:4px 0}
.relation-section-title.spatial {color:#f2b05a !important}
.relation-section-title.semantic {color:#6ed79b !important}
.relation-card {display:flex;justify-content:space-between;align-items:center;gap:10px;
  padding:8px 0;border-top:1px solid #19232e}
.relation-text {font-size:12px;line-height:1.35;flex:1}
.entity {font-weight:650;color:#edf2f7}
.predicate {font-weight:750;margin:0 4px}
.predicate.semantic {color:#70d79b}
.predicate.spatial {color:#f2b05a}
.relation-score {width:105px;display:flex;gap:6px;align-items:center;
  color:#7d8a99;font-size:10px}
.score-track {flex:1;height:4px;background:#26313c;border-radius:999px;overflow:hidden}
.score-fill {height:100%;border-radius:999px}
.score-fill.semantic {background:#63d493}
.score-fill.spatial {background:#f0ac4d}
.empty-relations {color:#748295;font-size:12px;padding:10px 0}
.stats-strip {display:flex;background:#0b1017;color:#dce4ed;border:1px solid #17202b;
  border-top:0;overflow:hidden}
.stats-strip div {flex:1;padding:8px 10px;border-right:1px solid #1d2834}
.stats-strip div:last-child {border:0}
.stats-strip strong {display:block;font-size:13px}
.stats-strip span {color:#718093;font-size:9px;white-space:nowrap}
.side-title {color:#8dd6d8 !important;font-size:10px !important;text-transform:uppercase;
  letter-spacing:.1em;font-weight:800 !important;margin:2px 0 8px !important}
#samples {background:transparent !important;border:0 !important;margin:2px 0 9px !important}
#samples .grid-wrap {grid-template-columns:repeat(3,1fr) !important;gap:7px !important}
#samples .gallery-item {border-radius:7px !important;border:1px solid #2a3745 !important;
  overflow:hidden !important}
#samples img {height:70px !important;object-fit:cover !important}
#samples .gallery-item.selected {border:2px solid #55ca8a !important}
#label-mode .wrap {background:#111923 !important;border:1px solid #293644 !important;
  border-radius:8px !important;padding:3px !important}
#label-mode label {color:#8e9bab !important;font-size:11px !important;
  padding:7px 12px !important}
#label-mode label.selected {background:#17629c !important;color:#fff !important;
  border-radius:6px !important}
#side-panel textarea,#side-panel input {background:#141d27 !important;
  color:#e8edf2 !important;border:1px solid #2a3745 !important;border-radius:8px !important}
#side-panel label span {color:#c8d1db !important;font-size:11px !important}
#apply {background:#17629c !important;border:0 !important;border-radius:7px !important;
  font-size:12px !important;font-weight:750 !important}
#vocab-status {color:#8d9aaa !important;font-size:10px !important}
#preset-row {gap:5px !important;flex-wrap:wrap !important}
#preset-row button {flex:1 1 auto !important;min-width:0 !important;
  font-size:10px !important;font-weight:700 !important;padding:5px 8px !important;
  border-radius:6px !important;background:#1a2533 !important;color:#c8d1db !important;
  border:1px solid #2a3745 !important;box-shadow:none !important}
#preset-row button:hover {background:#22303f !important;color:#fff !important}
#inference-controls {border-top:1px solid #25313d !important;margin-top:12px !important;
  padding-top:12px !important}
@media(max-width:900px) {#workspace{flex-direction:column !important}
  #side-panel{max-width:none;width:100%}}
"""

CANVAS_JS = r"""
(function(){
  const DEBUG = true;
  function log(){ if(DEBUG) console.log('[ra-canvas]', ...arguments); }

  function findCheckbox(id){
    const host = document.getElementById(id);
    if(!host) return null;
    return host.querySelector('input[type=checkbox]')
        || host.querySelector('input');
  }

  function setup(){
    // Find the visible input image. We gave the still gr.Image elem_id="source-image".
    const hosts = document.querySelectorAll('#source-image');
    let host = null;
    for(const h of hosts){
      if(h.offsetParent !== null){ host = h; break; }
    }
    if(!host) return;
    if(host.dataset.raCanvas) return;
    const img = host.querySelector('img');
    if(!img) return;

    host.dataset.raCanvas = '1';
    if(getComputedStyle(host).position === 'static') host.style.position = 'relative';

    const canvas = document.createElement('canvas');
    canvas.style.cssText =
      'position:absolute;inset:0;pointer-events:none;z-index:50;';
    host.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const COLORS = ['#1769aa','#eb6834','#1baf7a','#8a63d2','#d6a82a','#4bc4d0'];
    let boxes = [];
    let drag = null;

    function fit(){
      const r = img.getBoundingClientRect();
      const hr = host.getBoundingClientRect();
      canvas.style.left   = (r.left - hr.left) + 'px';
      canvas.style.top    = (r.top - hr.top) + 'px';
      canvas.width  = r.width; canvas.height = r.height;
      canvas.style.width  = r.width + 'px';
      canvas.style.height = r.height + 'px';
      redraw();
    }

    function redraw(){
      ctx.clearRect(0,0,canvas.width,canvas.height);
      boxes.forEach((b,i)=>{
        const c = COLORS[i % COLORS.length];
        ctx.strokeStyle = c; ctx.lineWidth = 2.5;
        ctx.strokeRect(b.x0, b.y0, b.x1-b.x0, b.y1-b.y0);
        ctx.fillStyle = c;
        ctx.fillRect(b.x0, Math.max(0, b.y0-17), 22, 17);
        ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif';
        ctx.fillText(String(i+1), b.x0+6, Math.max(12, b.y0-4));
      });
      if(drag){
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.setLineDash([5,4]);
        ctx.strokeRect(drag.sx, drag.sy, drag.cx-drag.sx, drag.cy-drag.sy);
        ctx.setLineDash([]);
      }
    }

    function writeOut(){
      const lines = boxes.map(b => [
        (b.x0/canvas.width).toFixed(4),
        (b.y0/canvas.height).toFixed(4),
        (b.x1/canvas.width).toFixed(4),
        (b.y1/canvas.height).toFixed(4),
      ].join(',')).join('\n');
      const ta = document.querySelector('#manual-boxes textarea');
      if(!ta){ log('no #manual-boxes textarea'); return; }
      const setter = Object.getOwnPropertyDescriptor(
        window.HTMLTextAreaElement.prototype, 'value').set;
      setter.call(ta, lines);
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function drawActive(){
      const cb = findCheckbox('draw-mode');
      return !!(cb && cb.checked);
    }

    host.addEventListener('mousedown', (e)=>{
      if(!drawActive() || e.button !== 0) return;
      const r = canvas.getBoundingClientRect();
      drag = {sx:e.clientX-r.left, sy:e.clientY-r.top,
              cx:e.clientX-r.left, cy:e.clientY-r.top};
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e)=>{
      if(!drag) return;
      const r = canvas.getBoundingClientRect();
      drag.cx = e.clientX-r.left; drag.cy = e.clientY-r.top;
      redraw();
    });
    window.addEventListener('mouseup', ()=>{
      if(!drag) return;
      const x0 = Math.min(drag.sx,drag.cx), x1 = Math.max(drag.sx,drag.cx);
      const y0 = Math.min(drag.sy,drag.cy), y1 = Math.max(drag.sy,drag.cy);
      if(x1-x0 > 6 && y1-y0 > 6){
        boxes.push({x0,y0,x1,y1});
        writeOut();
      }
      drag = null; redraw();
    });
    host.addEventListener('dblclick', (e)=>{
      if(!drawActive()) return;
      boxes = []; writeOut(); redraw(); e.preventDefault();
    });

    img.addEventListener('load', fit);
    new ResizeObserver(fit).observe(img);
    window.addEventListener('resize', fit);
    setTimeout(fit, 100);
    log('canvas attached');
  }

  function tick(){ try{ setup(); }catch(e){ log('setup error', e); } }
  window.addEventListener('load', tick);
  setTimeout(tick, 400);
  new MutationObserver(tick).observe(document.body, {childList:true, subtree:true});
})();
"""


# ---------------------------------------------------------------- ui

def build_ui(device_note: str):
    with gr.Blocks(title="RelateAnything") as demo:
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=7, elem_id="main-panel"):
                with gr.Tabs(elem_id="source-tabs"):
                    with gr.Tab("Image", id="image"):
                        still = gr.Image(
                            sources=["upload", "clipboard"], type="numpy",
                            show_label=False, height=430, elem_id="source-image",
                        )
                    with gr.Tab("Video", id="video"):
                        gr.Video(sources=["upload"], show_label=False, height=430)
                    with gr.Tab("Webcam", id="webcam"):
                        cam = gr.Image(
                            sources=["webcam"], streaming=True, type="numpy",
                            show_label=False, height=430,
                        )

                out = gr.Image(label=None, height=420, show_label=False,
                               elem_id="scene-output")
                stats = gr.HTML()

                with gr.Column(elem_id="prediction-panel"):
                    with gr.Tabs(elem_id="prediction-tabs"):
                        with gr.Tab("both", id="both") as tab_both:
                            rel_both = gr.HTML(elem_id="relations-merged")
                        with gr.Tab("spatial", id="spatial") as tab_spa:
                            rel_spa = gr.HTML(elem_id="relations-spatial")
                        with gr.Tab("semantic", id="semantic") as tab_sem:
                            rel_sem = gr.HTML(elem_id="relations-semantic")

            with gr.Column(scale=4, elem_id="side-panel"):
                gr.Markdown("1  LABELS", elem_classes=["side-title"])
                label_mode = gr.Radio(
                    ["names", "colours"], value="colours",
                    label="label objects by", elem_id="label-mode",
                )

                gr.Markdown("2  BOXES", elem_classes=["side-title"])
                draw_mode = gr.Checkbox(
                    False, label="draw on image",
                    info="tick, then drag rectangles on the picture; double-click clears",
                    elem_id="draw-mode",
                )
                manual_boxes = gr.Textbox(
                    label="my boxes (x1,y1,x2,y2 per line)",
                    placeholder="0.30,0.10,0.70,0.60\n0.10,0.55,0.95,0.95",
                    lines=4, value="", elem_id="manual-boxes",
                )

                gr.Markdown("3  PICTURE", elem_classes=["side-title"])
                gr.Examples(examples=[[p] for p in SAMPLE_IMAGES], inputs=still,
                            label=None, examples_per_page=6, elem_id="samples")

                gr.Markdown("4  VOCABULARY", elem_classes=["side-title"])
                classes_txt = gr.Textbox(
                    label="object classes",
                    placeholder="person, horse, hand, laptop, cup...",
                    lines=3, value="",
                )
                preds_txt = gr.Textbox(
                    label="predicates", lines=3,
                    value=", ".join(_default_predicates()),
                )
                vocab_msg = gr.Markdown(elem_id="vocab-status")

                gr.Markdown("presets", elem_classes=["side-title"])
                with gr.Row(elem_id="preset-row"):
                    for _pname in PREDICATE_PRESETS:
                        gr.Button(_pname, size="sm").click(
                            _preset_handler(_pname),
                            [classes_txt],
                            [preds_txt, vocab_msg],
                        )
                apply_btn = gr.Button("Apply vocabularies", variant="primary",
                                      elem_id="apply")

                with gr.Column(elem_id="inference-controls"):
                    gr.Markdown("5  INFERENCE", elem_classes=["side-title"])
                    conf = gr.Slider(0.05, 0.9, 0.25, step=0.05,
                                     label="detector confidence")
                    top_k = gr.Slider(1, 30, 12, step=1, label="max triplets")
                    score_thr = gr.Slider(0.0, 0.95, 0.30, step=0.05,
                                          label="relation score threshold")

        # ---- wiring ----
        apply_btn.click(apply_vocab, [classes_txt, preds_txt], vocab_msg)

        inputs_list = [conf, top_k, score_thr, label_mode, manual_boxes]
        outputs_list = [out, rel_both, rel_spa, rel_sem]

        def _dispatch(frame, c, k, s, lm, mtxt):
            # Empty box list -> detector path. Non-empty -> manual path, the
            # boxes you list are the only regions scored.
            if (mtxt or "").strip():
                return run_manual(frame, mtxt, k, s, lm)
            return run_frame(frame, c, k, s, lm)

        still.change(_dispatch, [still] + inputs_list, outputs_list)
        cam.stream(_dispatch, [cam] + inputs_list, outputs_list,
                   stream_every=0.12, concurrency_limit=1, show_progress="hidden")

        def _rerun(frame, c, k, s, lm, use_m, mtxt, mode_name):
            if use_m:
                return run_manual(frame, mtxt, k, s, lm)
            # mode-specific re-render for detector path
            res_tmp = run_frame(frame, c, k, s, lm)
            return res_tmp

        tab_both.select(lambda *a: _dispatch(*a), [still] + inputs_list, outputs_list)
        tab_spa.select(lambda *a: _dispatch(*a), [still] + inputs_list, outputs_list)
        tab_sem.select(lambda *a: _dispatch(*a), [still] + inputs_list, outputs_list)

        for ctrl in [conf, top_k, score_thr, label_mode, manual_boxes, draw_mode]:
            ctrl.change(_dispatch, [still] + inputs_list, outputs_list)

    return demo


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PipelineConfig.ckpt)
    ap.add_argument("--det", default="checkpoints/detectors/yoloe-11m-seg-pf.pt")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_objects", type=int, default=16)
    ap.add_argument("--final_budget", type=int, default=64)
    ap.add_argument("--no_overlap", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    import torch
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if dev != a.device:
        print("[demo] CUDA unavailable — falling back to CPU")

    global PIPE
    prompt_free = "-pf" in a.det
    cfg = PipelineConfig(
        ckpt=a.ckpt, det_weights=a.det, device=dev,
        max_objects=a.max_objects, final_budget=a.final_budget,
        overlap=(dev == "cuda" and not a.no_overlap),
        default_classes=None if prompt_free else DEFAULT_CLASSES,
    )
    print(f"[demo] loading pipeline on {dev} …")
    PIPE = ParallelScenePipeline(cfg)
    note = f"Running on **{dev.upper()}**"
    print(f"[demo] ready — {note}")

    build_ui(note).queue(max_size=4).launch(
        server_name="0.0.0.0", server_port=a.port, share=a.share,
        show_error=True, theme=gr.themes.Base(), css=CSS, js=CANVAS_JS,
    )


if __name__ == "__main__":
    main()
