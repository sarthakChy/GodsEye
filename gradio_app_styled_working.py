"""RelateAnything live demo — YOLOE-11m (masks) + open-vocabulary relations.

    python deploy/gradio_app.py                    # auto GPU, opens in browser
    python deploy/gradio_app.py --device cpu       # no GPU needed
    python deploy/gradio_app.py --share            # public link (SME demo)
    python deploy/gradio_app.py --port 7860

Everything runs server-side; the browser only ships webcam frames, so the demo
works on a laptop with the model on a remote GPU. On CPU it still runs, just
slower — use the Image tab rather than the live webcam there.

BOTH models are re-parameterizable live, which is the point of the demo:
  * object classes  -> YOLOE text prompts (`set_classes` + `get_text_pe`)
  * predicates      -> relation head vocabulary, encoded by the checkpoint's
                       own text encoder; inference stays pure-vision afterwards
Type any words into either box, hit Apply, and the pipeline is re-targeted
without touching the weights.
"""

from __future__ import annotations

import argparse
import os
import sys
import html
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr                                        # noqa: E402
from deploy.pipeline import (ParallelScenePipeline,        # noqa: E402
                             PipelineConfig, _default_predicates)

DEFAULT_CLASSES = [
    "person", "face", "hand", "laptop", "keyboard", "mouse", "monitor", "cup",
    "bottle", "phone", "book", "chair", "desk", "backpack", "hat", "glasses",
    "plant", "door", "window", "picture", "lamp", "bag",
]

# Distinct hues per object instance, fixed by index so a box keeps its colour
# across frames (colour follows the entity, never its rank).
PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (138, 99, 210),
           (214, 168, 42), (52, 187, 235), (200, 62, 120), (120, 160, 60)]

COLOR_NAMES = ["blue", "orange", "green", "purple", "yellow", "cyan", "pink", "olive"]


def _predicate_kind(pred: str) -> str:
    """Which stream this predicate belongs to, per the current vocabulary."""
    if PIPE is None:
        return "semantic"
    try:
        i = PIPE.ra.predicates.index(pred)
    except (ValueError, AttributeError):
        return "semantic"
    try:
        return "spatial" if bool(PIPE.is_spatial[i]) else "semantic"
    except Exception:
        return "semantic"



PREDICATE_PRESETS = {
    "Default": [
        "wearing", "riding", "playing", "sitting on", "sitting at", "holding",
        "sitting in", "looking at", "using", "watching", "standing on",
        "carrying", "talking to", "smiling at", "standing beside",
        "walking past", "posing with", "leaning against", "part of",
        "resting on", "on", "covering", "inside", "on top of", "contained in",
        "hanging from", "surrounding", "attached to", "in front of", "beside",
        "to the left of", "to the right of", "behind", "above", "below",
    ],
    "Spatial": [
        "on", "inside", "on top of", "in front of", "beside",
        "to the left of", "to the right of", "behind", "above", "below",
        "next to", "near", "under", "over", "underneath",
    ],
    "Semantic": [
        "wearing", "riding", "playing", "sitting on", "sitting at", "holding",
        "sitting in", "looking at", "using", "watching", "standing on",
        "carrying", "talking to", "smiling at", "standing beside",
        "walking past", "posing with", "leaning against", "part of",
        "resting on", "covering", "contained in", "hanging from",
        "surrounding", "attached to", "supporting", "containing", "worn by",
        "illuminating", "decorating", "resting in", "shading",
        "accompanying", "touching", "standing behind", "resting against",
        "mounted on", "driving past", "forming part of", "casting shadow on",
        "standing by", "standing in front of", "parked near",
        "walking through", "growing in", "depicting", "tucked under",
        "reflecting", "singing into", "following", "covering head of",
        "standing in", "operating", "held by", "standing near",
        "comprising", "hanging on", "embracing", "facing", "occupying",
        "passing", "pointing at", "eating", "parked on", "towering over",
        "riding in", "parked behind", "steering", "lying on", "growing from",
        "eating from", "hugging", "framing", "pulling", "floating in",
        "parked beside", "topping", "displaying", "covering eyes of",
        "growing near", "speaking into", "working at", "filling", "driving",
        "obscuring", "walking towards", "drinking from", "swimming in",
        "leaning over", "walking on", "serving", "reflecting in",
        "leaning on", "flying over", "cutting", "looking towards",
        "climbing", "reaching for", "sitting beside", "driving on",
        "pushing", "looking through", "parked in", "mounting",
        "standing next to", "showing", "contains", "parked in front of",
        "writing on", "carried by", "playing with", "stored in",
        "depicted in", "interacting with", "reaching towards",
        "walking across", "casting light on", "enclosing", "encasing",
        "leading", "walking with", "controlling", "shaking hands with",
        "recording", "driving along", "gripping", "standing among",
        "parked by", "resting inside", "smiling with", "running past",
        "grazing in", "kissing", "capturing sound from", "attaching to",
        "grazing near", "blocking", "connected to", "reading",
        "working near", "holding hands with", "encircling",
        "driving through", "pedaling", "listening to", "running across",
        "depicted on", "garnishing", "preparing", "having", "resting near",
        "talking into", "appearing in", "gesturing towards", "manipulating",
        "underlying", "standing under", "posing in front of", "featuring",
        "hitting", "embedded in", "tucked into", "swinging",
        "displaying content for", "looking past", "accommodating",
        "laughing with", "posing for", "feeding", "walking along",
        "floating in water near", "fastening", "approaching", "forming",
        "petting", "cushioning", "dancing with", "jumping over",
        "handling", "contained within", "photographing", "perching on",
        "casting light upon", "housing", "sitting near", "piercing",
        "kicking", "bordering", "decorated with", "resting beside",
        "working on", "stepping on", "performing with", "typing on",
        "dipping into", "traveling along", "performing near",
        "incorporating", "lying in", "grazing on", "running towards",
        "emitting sound for", "striking", "blooming from", "reflected in",
        "throwing", "observing", "looking into", "sitting behind",
        "topped with", "integrated into", "growing among",
        "resting under", "paddling", "assisting", "growing in front of",
        "posing for photo with", "growing beside", "kneeling on",
        "bending over", "standing with", "reaching toward", "clinging to",
        "cooling", "filming", "lining", "sitting by",
        "amplifying sound for", "juggling with", "galloping on",
        "repairing", "pouring", "stirring", "waiting for",
        "parked next to", "chasing", "tied to", "growing on",
        "covered with", "filled with", "grilling", "flipping",
        "seasoning", "balancing on", "trotting on", "guarding",
        "crossing", "queuing at",
    ],
    "All 263": [
        p.strip() for p in """wearing, riding, playing, sitting on, sitting at, holding, sitting in, looking at, using, watching, standing on, carrying, talking to, smiling at, standing beside, walking past, posing with, leaning against, part of, resting on, on, covering, inside, on top of, contained in, hanging from, surrounding, attached to, in front of, beside, to the left of, to the right of, behind, above, below, next to, supporting, containing, worn by, illuminating, near, decorating, resting in, shading, accompanying, touching, standing behind, resting against, mounted on, driving past, forming part of, casting shadow on, standing by, standing in front of, parked near, walking through, growing in, depicting, tucked under, reflecting, singing into, following, covering head of, standing in, operating, held by, standing near, comprising, hanging on, embracing, facing, occupying, passing, pointing at, eating, parked on, towering over, riding in, parked behind, under, steering, lying on, growing from, eating from, hugging, framing, pulling, floating in, parked beside, topping, displaying, covering eyes of, over, growing near, speaking into, working at, filling, driving, obscuring, walking towards, drinking from, swimming in, leaning over, walking on, serving, reflecting in, leaning on, flying over, cutting, looking towards, climbing, reaching for, sitting beside, driving on, pushing, looking through, parked in, mounting, standing next to, showing, contains, parked in front of, writing on, carried by, playing with, stored in, depicted in, interacting with, reaching towards, walking across, casting light on, enclosing, encasing, leading, walking with, controlling, shaking hands with, recording, driving along, gripping, standing among, parked by, resting inside, smiling with, running past, grazing in, kissing, capturing sound from, attaching to, grazing near, blocking, connected to, reading, working near, holding hands with, encircling, driving through, pedaling, listening to, running across, depicted on, garnishing, preparing, having, resting near, talking into, appearing in, gesturing towards, manipulating, underlying, standing under, posing in front of, featuring, hitting, embedded in, tucked into, swinging, displaying content for, looking past, accommodating, laughing with, posing for, feeding, walking along, floating in water near, fastening, approaching, forming, petting, cushioning, dancing with, jumping over, handling, contained within, photographing, perching on, casting light upon, housing, sitting near, piercing, kicking, bordering, decorated with, resting beside, working on, stepping on, performing with, typing on, dipping into, traveling along, performing near, incorporating, lying in, grazing on, running towards, emitting sound for, striking, blooming from, reflected in, throwing, observing, looking into, sitting behind, topped with, integrated into, growing among, resting under, paddling, assisting, growing in front of, posing for photo with, growing beside, kneeling on, bending over, standing with, reaching toward, clinging to, cooling, filming, lining, sitting by, amplifying sound for, juggling with, galloping on, repairing, pouring, stirring, waiting for, parked next to, chasing, tied to, growing on, covered with, filled with, grilling, flipping, seasoning, balancing on, trotting on, guarding, crossing, queuing at""".split(",")
    ],
    "Minimal": [
        "holding", "wearing", "riding", "sitting on",
        "on", "in front of", "behind", "next to",
    ],
}


def _make_preset_handler(name):
    def handler(classes_txt_value):
        text = ", ".join(PREDICATE_PRESETS[name])
        return text, apply_vocab(classes_txt_value, text)
    return handler



def _object_name(res, i: int, label_mode: str = "names") -> str:
    if label_mode == "colours":
        return f"{COLOR_NAMES[i % len(COLOR_NAMES)]} object {i + 1}"
    return res.labels[i] if i < len(res.labels) else f"object {i + 1}"


def _repo_root() -> Path:
    here = Path(__file__).resolve().parent
    if (here / "assets").is_dir():
        return here
    if (here.parent / "assets").is_dir():
        return here.parent
    return here


REPO_ROOT = _repo_root()

# Exact sample images from the repository:
# assets/reel/images/{bicycle,catlaptop,frisbee,horse,skateboard,tennis}.jpg
SAMPLE_IMAGES = [
    REPO_ROOT / "assets" / "reel" / "images" / "bicycle.jpg",
    REPO_ROOT / "assets" / "reel" / "images" / "catlaptop.jpg",
    REPO_ROOT / "assets" / "reel" / "images" / "frisbee.jpg",
    REPO_ROOT / "assets" / "reel" / "images" / "horse.jpg",
    REPO_ROOT / "assets" / "reel" / "images" / "skateboard.jpg",
    REPO_ROOT / "assets" / "reel" / "images" / "tennis.jpg",
]

SAMPLE_IMAGES = [p for p in SAMPLE_IMAGES if p.is_file()]

PIPE: ParallelScenePipeline | None = None


def _color(i: int):
    return PALETTE[i % len(PALETTE)]


# Two-graph edge colours: layout (spatial) vs content (semantic). Fixed by
# meaning, not by rank — a predicate keeps its colour across frames.
SPATIAL_EDGE = (120, 200, 255)    # BGR: warm-cyan  → "where things are"
SEMANTIC_EDGE = (140, 255, 170)   # BGR: green      → "what things do"
MERGED_EDGE = (255, 255, 255)


def _edges_for(res, mode: str):
    """(list_of_triplets, colour) pairs for the requested graph mode."""
    if mode == "merged":
        return [(res.triplets, MERGED_EDGE)]
    if mode == "spatial":
        return [(res.triplets_spatial, SPATIAL_EDGE)]
    if mode == "semantic":
        return [(res.triplets_semantic, SEMANTIC_EDGE)]
    return [(res.triplets_spatial, SPATIAL_EDGE),
            (res.triplets_semantic, SEMANTIC_EDGE)]     # both


def render(res, show_masks: bool, show_labels: bool,
           mode: str = "merged", label_mode: str = "names") -> np.ndarray:
    """Draw masks, boxes and relation arrows onto the frame (BGR in, RGB out)."""
    img = res.frame.copy()
    H, W = img.shape[:2]

    if show_masks and res.masks is not None and len(res.masks):
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
            sc = float(res.scores[i]) if i < len(res.scores) else 0.0
            txt = f"{i}:{lab} {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                          _color(i), -1)
            cv2.putText(img, txt, (x1 + 2, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # relation arrows, thickness ~ score; colour ~ graph (layout vs content).
    # In "both" mode the two streams are drawn with a small perpendicular
    # offset so a pair carrying one edge of each type stays readable.
    streams = _edges_for(res, mode)
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
    return img[:,:,::-1]


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


CSS = r"""
    .gradio-container {
        max-width: 100% !important;
        background:#fff !important;
        font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif !important;
    }

    #ra-header {
        margin:-14px -14px 0 !important;
        padding:12px 24px !important;
        background:#fff !important;
        border-bottom:1px solid #e6e9ed !important;
    }
    .brand {display:flex;align-items:center;gap:10px;font-size:18px;font-weight:750;color:#18212b}
    .brand-mark {width:10px;height:10px;border-radius:50%;background:#1769aa;
                 box-shadow:9px 0 0 #35b879;margin-right:5px}
    .badge {font-size:10px;font-weight:800;letter-spacing:.08em;border:1px solid #c7e2f6;
            background:#eef8ff;color:#1769aa;border-radius:999px;padding:4px 8px}
    .nav {text-align:right;color:#667085;font-size:12px;padding-top:3px}

    #intro {padding:22px 0 15px !important}
    #intro p {max-width:850px;color:#596575 !important;font-size:14px !important;line-height:1.55 !important}

    #workspace {gap:20px !important;align-items:flex-start !important}
    #main-panel {min-width:0;padding:0 !important}
    #side-panel {
        min-width:350px;max-width:395px;padding:14px !important;
        background:#0e151d !important;border:1px solid #1e2935 !important;
        border-radius:12px !important;color:#dce4ed !important;
    }

    /* Image / Video / Webcam bar */
    #source-tabs {
        background:#0b1017 !important;border:1px solid #151d27 !important;
        border-radius:11px 11px 0 0 !important;overflow:hidden !important;
    }
    #source-tabs .tab-nav {
        background:#0b1017 !important;border-bottom:1px solid #202b37 !important;
        padding:0 4px !important;
    }
    #source-tabs .tab-nav button {
        color:#8e9bab !important;font-size:12px !important;font-weight:650 !important;
        border:0 !important;padding:10px 18px !important;
    }
    #source-tabs .tab-nav button.selected {
        color:#fff !important;border-bottom:2px solid #67d396 !important;
    }
    #source-tabs img,#source-tabs video {
        background:#0b1017 !important;object-fit:contain !important;max-height:440px !important;
    }

    #scene-output {
        background:#0b1017 !important;border:1px solid #151d27 !important;
        border-radius:0 !important;overflow:hidden !important;min-height:420px !important;
    }
    #scene-output img {background:#0b1017 !important;object-fit:contain !important}

    /* Prediction tabs directly under the rendered image */
    #prediction-panel {
        background:#0b1017 !important;border:1px solid #151d27 !important;
        border-top:0 !important;border-radius:0 0 11px 11px !important;
        overflow:hidden !important;
    }
    #prediction-tabs {background:#0b1017 !important}
    #prediction-tabs .tab-nav {
        background:#0b1017 !important;border-bottom:1px solid #202b37 !important;
        padding:0 8px !important;
    }
    #prediction-tabs .tab-nav button {
        background:transparent !important;border:0 !important;color:#7f8c9c !important;
        font-size:12px !important;font-weight:650 !important;padding:10px 14px !important;
    }
    #prediction-tabs .tab-nav button.selected {color:#fff !important}
    #prediction-tabs .tab-nav button:nth-child(2).selected {color:#f2b05a !important}
    #prediction-tabs .tab-nav button:nth-child(3).selected {color:#6ed79b !important}

    #relations {
        background:#0b1017;color:#dce4ed;padding:5px 14px 12px;
        max-height:350px;overflow:auto;
    }
    .relations-header {display:flex;justify-content:space-between;align-items:center;
                       font-size:12px;font-weight:750;color:#f3f6f9;margin-bottom:5px}
    .relation-count {background:#1b2632;color:#9aa8b8;border-radius:999px;min-width:22px;height:22px;
                     display:grid;place-items:center;font-size:10px}
    .relation-section-title {font-size:9px;font-weight:800;text-transform:uppercase;
                             letter-spacing:.09em;color:#7f8c9c;margin:4px 0}
    .relation-card {display:flex;justify-content:space-between;align-items:center;gap:10px;
                    padding:8px 0;border-top:1px solid #19232e}
    .relation-text {font-size:12px;line-height:1.35;flex:1}
    .entity {font-weight:650;color:#edf2f7}
    .predicate {font-weight:750;margin:0 4px}
    .predicate.semantic {color:#70d79b}
    .predicate.spatial {color:#f2b05a}
    .relation-score {width:105px;display:flex;gap:6px;align-items:center;color:#7d8a99;font-size:10px}
    .score-track {flex:1;height:4px;background:#26313c;border-radius:999px;overflow:hidden}
    .score-fill {height:100%;border-radius:999px}
    .score-fill.semantic {background:#63d493}
    .score-fill.spatial {background:#f0ac4d}

    .stats-strip {display:flex;background:#0b1017;color:#dce4ed;border:1px solid #17202b;
                   border-top:0;overflow:hidden}
    .stats-strip div {flex:1;padding:8px 10px;border-right:1px solid #1d2834}
    .stats-strip div:last-child {border:0}
    .stats-strip strong {display:block;font-size:13px}
    .stats-strip span {color:#718093;font-size:9px;white-space:nowrap}

    /* Reference-style right panel */
    .side-title {color:#8dd6d8 !important;font-size:10px !important;text-transform:uppercase;
                 letter-spacing:.1em;font-weight:800 !important;margin:2px 0 8px !important}
    .precomputed {border:1px solid #293644;border-radius:10px;padding:12px;margin-bottom:14px;background:#111923}
    .precomputed-kicker {color:#4bc4d0;font-size:10px;font-weight:850;letter-spacing:.1em;margin-bottom:8px}
    .precomputed p {color:#9aa8b7;font-size:12px;line-height:1.5;margin:0 0 11px}
    .run-live {display:block;text-align:center;padding:9px 10px;background:#17629c;color:#fff;
               border-radius:7px;font-size:12px;font-weight:750}

    #picture-source .wrap {
        background:#111923 !important;border:1px solid #293644 !important;
        border-radius:8px !important;padding:3px !important;
    }
    #picture-source label {
        color:#8e9bab !important;font-size:11px !important;padding:7px 11px !important;
    }
    #picture-source label.selected {
        background:#17629c !important;color:#fff !important;border-radius:6px !important;
    }

    #samples {background:transparent !important;border:0 !important;margin:2px 0 9px !important}
    #samples .grid-wrap {grid-template-columns:repeat(3,1fr) !important;gap:7px !important}
    #samples .gallery-item {border-radius:7px !important;border:1px solid #2a3745 !important;overflow:hidden !important}
    #samples img {height:70px !important;object-fit:cover !important}
    #samples .gallery-item.selected {border:2px solid #55ca8a !important}

    #label-mode .wrap {
        background:#111923 !important;border:1px solid #293644 !important;
        border-radius:8px !important;padding:3px !important;
    }
    #label-mode label {color:#8e9bab !important;font-size:11px !important;padding:7px 12px !important}
    #label-mode label.selected {background:#17629c !important;color:#fff !important;border-radius:6px !important}

    #side-panel textarea,#side-panel input {
        background:#141d27 !important;color:#e8edf2 !important;border:1px solid #2a3745 !important;
        border-radius:8px !important;
    }
    #side-panel label span {color:#c8d1db !important;font-size:11px !important}
    #apply {background:#17629c !important;border:0 !important;border-radius:7px !important;
            font-size:12px !important;font-weight:750 !important}
    #preset-row {gap:5px !important;flex-wrap:wrap !important}
    #preset-row button {
        flex:1 1 auto !important;min-width:0 !important;
        font-size:10px !important;font-weight:700 !important;
        padding:5px 8px !important;border-radius:6px !important;
        background:#1a2533 !important;color:#c8d1db !important;
        border:1px solid #2a3745 !important;box-shadow:none !important;
    }
    #preset-row button:hover {background:#22303f !important;color:#fff !important}
    #vocab-status {color:#8d9aaa !important;font-size:10px !important}

    #inference-controls {
        border-top:1px solid #25313d !important;margin-top:12px !important;padding-top:12px !important;
    }

    @media(max-width:900px) {
        #workspace{flex-direction:column !important}
        #side-panel{max-width:none;width:100%}
    }
    """

def build_ui(device_note: str):

    with gr.Blocks(title="RelateAnything") as demo:
        with gr.Row(elem_id="workspace"):
            with gr.Column(scale=7, elem_id="main-panel"):
                with gr.Tabs(elem_id="source-tabs") as source_tabs:
                    with gr.Tab("Image", id="image"):
                        still = gr.Image(
                            sources=["upload", "clipboard"], type="numpy",
                            show_label=False, height=430
                        )
                    with gr.Tab("Video", id="video"):
                        video = gr.Video(
                            sources=["upload"], show_label=False, height=430
                        )
                    with gr.Tab("Webcam", id="webcam"):
                        cam = gr.Image(
                            sources=["webcam"], streaming=True, type="numpy",
                            show_label=False, height=430
                        )

                out = gr.Image(
                    label=None, height=420, show_label=False, elem_id="scene-output"
                )
                stats = gr.HTML()

                with gr.Column(elem_id="prediction-panel"):
                    with gr.Tabs(elem_id="prediction-tabs") as prediction_tabs:
                        with gr.Tab("both", id="both") as merged_tab:
                            rel_merged = gr.HTML(elem_id="relations-both")
                        with gr.Tab("spatial", id="spatial") as spatial_tab:
                            rel_spatial = gr.HTML(elem_id="relations-spatial")
                        with gr.Tab("semantic", id="semantic") as semantic_tab:
                            rel_semantic = gr.HTML(elem_id="relations-semantic")

            with gr.Column(scale=4, elem_id="side-panel"):
                label_mode = gr.Radio(
                    ["names", "colours"], value="colours",
                    label="label objects by",
                    elem_id="label-mode",
                )

                gr.Examples(
                    examples=[[str(p)] for p in SAMPLE_IMAGES],
                    inputs=still,
                    label=None,
                    examples_per_page=6,
                    elem_id="samples",
                )

                classes_txt = gr.Textbox(
                    label="object classes",
                    placeholder="person, horse, hand, laptop, cup...",
                    lines=3, value=""
                )
                preds_txt = gr.Textbox(
                    label="predicates", lines=3,
                    value=", ".join(_default_predicates())
                )
                vocab_msg = gr.Markdown(elem_id="vocab-status")
                gr.Markdown("presets", elem_classes=["side-title"])
                with gr.Row(elem_id="preset-row"):
                    for _pname in PREDICATE_PRESETS:
                        gr.Button(_pname, size="sm").click(
                            _make_preset_handler(_pname),
                            [classes_txt],
                            [preds_txt, vocab_msg],
                        )

                apply_btn = gr.Button(
                    "Apply vocabularies", variant="primary", elem_id="apply"
                )

                with gr.Column(elem_id="inference-controls"):
                    conf = gr.Slider(
                        0.05, 0.9, 0.25, step=0.05, label="detector confidence"
                    )
                    top_k = gr.Slider(
                        1, 30, 12, step=1, label="max triplets"
                    )
                    score_thr = gr.Slider(
                        0.0, 0.95, 0.30, step=0.05,
                        label="relation score threshold"
                    )

        def run_frame(frame_rgb, conf_v, top_k_v, score_v, masks_v, mode_v, raw_v, label_v):
            if frame_rgb is None or PIPE is None:
                return None, "", "", ""
            PIPE.cfg.det_conf = float(conf_v)
            res = PIPE(
                frame_rgb[:, :, ::-1].copy(),
                top_k=int(top_k_v),
                score_thr=float(score_v),
                decompose=(mode_v != "merged"),
                spatial_drop_pair=bool(raw_v),
            )

            def make_relations(title, trips, edge_kind):
                body = ""
                for s, pred, o, sc in trips:
                    kind = _predicate_kind(pred) if edge_kind == "auto" else edge_kind
                    subject = _object_name(res, s, label_v)
                    obj = _object_name(res, o, label_v)
                    pct = max(0, min(100, int(float(sc) * 100)))
                    body += (
                        '<div class="relation-card">'
                        '<div class="relation-text">'
                        f'<span class="entity">{html.escape(subject)}</span>'
                        f'<span class="predicate {kind}">{html.escape(str(pred))}</span>'
                        f'<span class="entity">{html.escape(obj)}</span>'
                        '</div>'
                        '<div class="relation-score"><div class="score-track">'
                        f'<div class="score-fill {kind}" style="width:{pct}%"></div>'
                        f'</div><span>{float(sc):.2f}</span></div></div>'
                    )
                return (
                    '<div class="relations-header"><span>Predictions</span>'
                    f'<span class="relation-count">{len(trips)}</span></div>'
                    f'<div class="relation-section-title">{html.escape(title)}</div>'
                    + (body or '<div>No relations above threshold.</div>')
                )

            merged_html = make_relations("all relations", res.triplets, "auto")
            spatial_html = make_relations("layout · spatial", res.triplets_spatial, "spatial")
            semantic_html = make_relations("content · semantic", res.triplets_semantic, "semantic")

            t = res.timing
            stats_html = (
                '<div class="stats-strip">'
                f'<div><strong>{t.fps:.1f}</strong><span>FPS</span></div>'
                f'<div><strong>{t.det:.1f}</strong><span>det ms</span></div>'
                f'<div><strong>{t.backbone:.1f}</strong><span>backbone ms</span></div>'
                f'<div><strong>{t.relation:.1f}</strong><span>rel ms</span></div>'
                f'<div><strong>{t.total:.1f}</strong><span>total ms</span></div>'
                f'<div><strong>{len(res.boxes_xyxy)}</strong><span>objects</span></div>'
                '</div>'
            )

            return (
                render(res, masks_v, True, mode_v, label_v),
                merged_html, spatial_html, semantic_html
            )

        apply_btn.click(
            apply_vocab, [classes_txt, preds_txt], vocab_msg
        )

        # Image sample/upload inference.
        still.change(
            lambda frame, c, k, s, lm: run_frame(frame, c, k, s, True, "both", False, lm),
            [still, conf, top_k, score_thr, label_mode],
            [out, rel_merged, rel_spatial, rel_semantic],
        )

        # Webcam inference.
        cam.stream(
            lambda frame, c, k, s, lm: run_frame(frame, c, k, s, True, "both", False, lm),
            [cam, conf, top_k, score_thr, label_mode],
            [out, rel_merged, rel_spatial, rel_semantic],
            stream_every=0.12, concurrency_limit=1, show_progress="hidden",
        )

        # Keep the three prediction panels populated and let clicking a prediction
        # tab change the rendered graph mode.
        def rerun_for_mode(frame, c, k, s, m, sr, lm, mode_name):
            return run_frame(frame, c, k, s, m, mode_name, sr, lm)

        for tab, mode_name in [
            (None, "merged"),
        ]:
            pass

        # Prediction tab events are attached below after the tab objects are available.
        # Gradio exposes the tab event as .select().
        # The default merged tab is already handled by the image/webcam callbacks.

        merged_tab.select(
            lambda frame, c, k, s, lm: run_frame(frame, c, k, s, True, "both", False, lm),
            [still, conf, top_k, score_thr, label_mode],
            [out, rel_merged, rel_spatial, rel_semantic],
        )
        spatial_tab.select(
            lambda frame, c, k, s, lm: run_frame(frame, c, k, s, True, "spatial", False, lm),
            [still, conf, top_k, score_thr, label_mode],
            [out, rel_merged, rel_spatial, rel_semantic],
        )
        semantic_tab.select(
            lambda frame, c, k, s, lm: run_frame(frame, c, k, s, True, "semantic", False, lm),
            [still, conf, top_k, score_thr, label_mode],
            [out, rel_merged, rel_spatial, rel_semantic],
        )

        # Changing names/colours, masks, or thresholds refreshes the current sample.
        for control in [conf, top_k, score_thr, label_mode]:
            control.change(
                lambda frame, c, k, s, lm: run_frame(frame, c, k, s, True, "both", False, lm),
                [still, conf, top_k, score_thr, label_mode],
                [out, rel_merged, rel_spatial, rel_semantic],
            )

    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PipelineConfig.ckpt)
    ap.add_argument("--det", default="checkpoints/detectors/yoloe-11m-seg-pf.pt",
                    help="'-pf' = prompt-free YOLOE (works with no class list); "
                         "yoloe-11m-seg.pt = text-prompt (needs classes set)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_objects", type=int, default=16)
    ap.add_argument("--final_budget", type=int, default=64)
    ap.add_argument("--no_overlap", action="store_true",
                    help="disable detector‖backbone overlap (A/B the pipeline)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()

    import torch
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if dev != a.device:
        print("[demo] CUDA unavailable — falling back to CPU")

    global PIPE
    prompt_free = "-pf" in a.det
    cfg = PipelineConfig(ckpt=a.ckpt, det_weights=a.det, device=dev,
                         max_objects=a.max_objects,
                         final_budget=a.final_budget,
                         overlap=(dev == "cuda" and not a.no_overlap),
                         default_classes=None if prompt_free else DEFAULT_CLASSES)
    print(f"[demo] loading pipeline on {dev} …")
    PIPE = ParallelScenePipeline(cfg)
    note = (f"Running on **{dev.upper()}**"
            + (" with detector‖backbone overlap." if cfg.overlap else "."))
    print(f"[demo] ready — {note}")
    build_ui(note).queue(max_size=4).launch(
        server_name="0.0.0.0", server_port=a.port, share=a.share,
        show_error=True,
        theme=gr.themes.Base(), css=CSS,
        js=r"""
function raSetupCanvas() {
  // Find the input image container. Gradio 6 wraps images like:
  //   <div class="image-container"><img .../></div>
  const containers = document.querySelectorAll(
    '#source-tabs .image-container, #source-tabs .image-frame'
  );
  containers.forEach((container) => {
    if (container.dataset.raCanvas) return;
    const img = container.querySelector('img');
    if (!img) return;
    container.dataset.raCanvas = '1';
    if (getComputedStyle(container).position === 'static') {
      container.style.position = 'relative';
    }

    const canvas = document.createElement('canvas');
    canvas.style.cssText =
      'position:absolute;top:0;left:0;pointer-events:none;z-index:50;';
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');

    const COLORS = ['#1769aa','#eb6834','#1baf7a','#8a63d2','#d6a82a','#4bc4d0'];
    let boxes = [];
    let drag = null;

    function fit() {
      const r = img.getBoundingClientRect();
      canvas.width = r.width; canvas.height = r.height;
      canvas.style.width = r.width + 'px';
      canvas.style.height = r.height + 'px';
      redraw();
    }
    function redraw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const r = img.getBoundingClientRect();
      boxes.forEach((b, i) => {
        const c = COLORS[i % COLORS.length];
        ctx.strokeStyle = c; ctx.lineWidth = 2.5;
        ctx.strokeRect(b.x0, b.y0, b.x1 - b.x0, b.y1 - b.y0);
        ctx.fillStyle = c;
        ctx.fillRect(b.x0, Math.max(0, b.y0 - 17), 20, 17);
        ctx.fillStyle = '#fff'; ctx.font = 'bold 11px sans-serif';
        ctx.fillText(String(i + 1), b.x0 + 5, Math.max(11, b.y0 - 4));
      });
      if (drag) {
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.strokeRect(drag.sx, drag.sy, drag.cx - drag.sx, drag.cy - drag.sy);
        ctx.setLineDash([]);
      }
    }
    function writeToTextbox() {
      const r = img.getBoundingClientRect();
      const lines = boxes.map(b => [
        (b.x0 / r.width).toFixed(4),
        (b.y0 / r.height).toFixed(4),
        (b.x1 / r.width).toFixed(4),
        (b.y1 / r.height).toFixed(4),
      ].join(',')).join('\n');
      const ta = document.querySelector('#manual-boxes textarea');
      if (!ta) return;
      ta.value = lines;
      ta.dispatchEvent(new Event('input', { bubbles: true }));
    }

    function active() {
      const cb = document.querySelector('#draw-mode input[type=checkbox]');
      return cb && cb.checked;
    }

    container.addEventListener('mousedown', (e) => {
      if (!active() || e.button !== 0) return;
      const r = canvas.getBoundingClientRect();
      drag = { sx: e.clientX - r.left, sy: e.clientY - r.top,
               cx: e.clientX - r.left, cy: e.clientY - r.top };
      e.preventDefault();
    });
    window.addEventListener('mousemove', (e) => {
      if (!drag) return;
      const r = canvas.getBoundingClientRect();
      drag.cx = e.clientX - r.left; drag.cy = e.clientY - r.top;
      redraw();
    });
    window.addEventListener('mouseup', () => {
      if (!drag) return;
      const x0 = Math.min(drag.sx, drag.cx), x1 = Math.max(drag.sx, drag.cx);
      const y0 = Math.min(drag.sy, drag.cy), y1 = Math.max(drag.sy, drag.cy);
      if (x1 - x0 > 6 && y1 - y0 > 6) { boxes.push({ x0, y0, x1, y1 }); writeToTextbox(); }
      drag = null; redraw();
    });
    container.addEventListener('dblclick', (e) => {
      if (!active()) return;
      boxes = []; writeToTextbox(); redraw();
      e.preventDefault();
    });
    img.addEventListener('load', fit);
    new ResizeObserver(fit).observe(img);
    fit();
  });
}
raSetupCanvas();
new MutationObserver(raSetupCanvas).observe(document.body,
  { childList: true, subtree: true });
""")


if __name__ == "__main__":
    main()

