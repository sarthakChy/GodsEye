"""
GodsEye Dashboard v2 — Unified, Polished Interface
====================================================
A completely redesigned single-file Gradio dashboard that unifies:
  • Live Inference  (image / video / webcam + live animated progress + relation streams)
  • Temporal Scene Graph (frame scrubber, active NetworkX/Plotly scene graph at each timestamp, Gantt chart)
  • ExamGuard Audit Log (real-time malpractice detection & violation cards)

Run with:
    cd d:\\Major - 1\\GodsEye
    ..\\env\\Scripts\\python temporal\\viz\\dashboard_v2.py
        --ckpt model/model.pth
        --det checkpoints/detectors/yoloe-11s-seg-pf.pt
        --device cpu

The original dashboard_ui.py and gradio_app_styled.py are NOT modified.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import cv2
import numpy as np
import plotly.graph_objects as go
import networkx as nx

# ── make sure repo root is on path ──────────────────────────────────────────
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

import gradio as gr

# ── lazy pipeline import (only needed for live-inference tab) ────────────────
PIPE = None


def _load_pipe(ckpt: str, det: str, device: str):
    global PIPE
    if PIPE is not None:
        return
    from deploy.pipeline import ParallelScenePipeline, PipelineConfig
    cfg = PipelineConfig(ckpt=ckpt, det_weights=det, device=device)
    PIPE = ParallelScenePipeline(cfg)
    print(f"[GodsEye v2] pipeline ready on {device}")


# ── temporal / exam imports ──────────────────────────────────────────────────
from temporal.schema import TemporalRelation
from temporal.scene_graph import TemporalSceneGraph
from temporal.viz.graph_renderer import render_graph
from temporal.viz.overlay import render_overlay
from temporal.viz.timeline import render_timeline
from temporal.exam_rules import run_exam_rules, violations_at_time, MalpracticeEvent


# ═══════════════════════════════════════════════════════════════════════════
# Constants & Vocabulary
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_CLASSES = [
    "person", "face", "hand", "laptop", "keyboard", "mouse", "monitor",
    "cup", "bottle", "phone", "book", "chair", "desk", "backpack", "hat",
    "glasses", "plant", "door", "window", "picture", "lamp", "bag",
]

EXAM_CLASSES = [
    "person", "book", "notebook", "paper", "phone", "cell_phone",
    "desk", "table", "pen", "pencil", "bag", "backpack", "laptop",
]

PREDICATE_PRESETS = {
    "Default": [p.strip() for p in (
        "wearing, riding, sitting on, holding, sitting at, sitting in, "
        "looking at, using, watching, standing on, carrying, talking to, "
        "smiling at, standing beside, walking past, leaning against, "
        "resting on, on, inside, in front of, beside, to the left of, "
        "to the right of, behind, above, below"
    ).split(",")],
    "Spatial": [
        "on", "inside", "on top of", "in front of", "beside",
        "to the left of", "to the right of", "behind", "above", "below",
        "next to", "near", "under", "over",
    ],
    "ExamGuard": [
        "holding", "using", "carrying", "near", "next to",
        "talking to", "looking at", "reaching toward", "reaching towards",
        "passing", "handing", "leaning over",
    ],
    "Minimal": ["holding", "wearing", "riding", "sitting on", "on", "in front of", "behind"],
}

PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (138, 99, 210),
           (214, 168, 42), (52, 187, 235), (200, 62, 120), (120, 160, 60)]


# ═══════════════════════════════════════════════════════════════════════════
# Shared dashboard state
# ═══════════════════════════════════════════════════════════════════════════

class _State:
    manifest = None
    tsg = None
    events: List[TemporalRelation] = []
    run_dir: Optional[Path] = None
    video_loader = None
    violations: List[MalpracticeEvent] = []
    last_processed_run: Optional[str] = None

STATE = _State()


# ═══════════════════════════════════════════════════════════════════════════
# CSS — Dark Navy Theme & Glowing Animations
# ═══════════════════════════════════════════════════════════════════════════

CSS = r"""
:root {
  --bg0:    #07090f;
  --bg1:    #0d1117;
  --bg2:    #111827;
  --bg3:    #1a2234;
  --border: #1f2d3d;
  --accent: #3b82f6;
  --accent2: #22d3ee;
  --green:  #4ade80;
  --amber:  #fbbf24;
  --red:    #f87171;
  --purple: #c084fc;
  --text0:  #f1f5f9;
  --text1:  #94a3b8;
  --text2:  #64748b;
  --radius: 10px;
  --font:   Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

.gradio-container {
  max-width: 100% !important;
  background: var(--bg1) !important;
  font-family: var(--font) !important;
  color: var(--text0) !important;
}

#gd-header {
  background: linear-gradient(135deg, #0b1329 0%, #172554 100%);
  border-bottom: 1px solid var(--border);
  padding: 16px 24px !important;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
#gd-header .title-area { display: flex; align-items: center; gap: 12px; }
#gd-header .title { font-size: 22px; font-weight: 800; color: var(--text0); letter-spacing: -.02em; }
#gd-header .subtitle { font-size: 12px; color: #93c5fd; margin-top: 2px; }
#gd-header .badge {
  background: linear-gradient(135deg, #2563eb, #06b6d4);
  color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .06em;
  padding: 5px 12px; border-radius: 999px; box-shadow: 0 0 14px rgba(37,99,235,0.4);
}

.godseye-tabs > .tab-nav {
  background: var(--bg2) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 0 20px !important;
  gap: 0 !important;
}
.godseye-tabs > .tab-nav button {
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  color: var(--text1) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 14px 20px !important;
  margin: 0 !important;
  border-radius: 0 !important;
  transition: all .15s;
}
.godseye-tabs > .tab-nav button.selected {
  color: var(--accent2) !important;
  border-bottom: 2px solid var(--accent2) !important;
  background: transparent !important;
}
.godseye-tabs > .tab-nav button:hover:not(.selected) { color: var(--text0) !important; }

.gd-panel {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
}
.gd-panel-title {
  font-size: 11px;
  font-weight: 800;
  letter-spacing: .1em;
  text-transform: uppercase;
  color: var(--text1);
  margin-bottom: 10px;
  padding-bottom: 6px;
  border-bottom: 1px solid var(--border);
}

.gd-media {
  background: var(--bg0) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
}
.gd-media img, .gd-media video {
  object-fit: contain !important;
  background: var(--bg0) !important;
}

.btn-primary {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
  font-size: 13px !important;
  transition: background .15s;
}
.btn-primary:hover { background: #2563eb !important; }

.btn-secondary {
  background: var(--bg3) !important;
  color: var(--text0) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 12px !important;
}

.btn-examguard {
  background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-weight: 700 !important;
}

/* ── Animated Processing Banners ────────────────────────────────────────── */
@keyframes spin {
  to { transform: rotate(360deg); }
}
@keyframes pulse-bar {
  0%   { width: 5%; }
  50%  { width: 75%; }
  100% { width: 98%; }
}
@keyframes glow-pulse {
  0%, 100% { box-shadow: 0 0 10px rgba(59,130,246,0.3); }
  50%      { box-shadow: 0 0 22px rgba(34,211,238,0.6); }
}
@keyframes fade-slide {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}

.proc-banner {
  display: flex;
  align-items: center;
  gap: 16px;
  background: linear-gradient(135deg, #0b1a30 0%, #0e274a 100%);
  border: 1px solid #1d4ed8;
  border-radius: 10px;
  padding: 14px 18px;
  margin: 10px 0;
  animation: fade-slide .3s ease, glow-pulse 2s infinite ease-in-out;
  font-family: var(--font);
}
.proc-spinner {
  width: 32px; height: 32px; flex-shrink: 0;
  border: 3.5px solid #1e3a5f;
  border-top-color: #38bdf8;
  border-radius: 50%;
  animation: spin .7s linear infinite;
}
.proc-text { flex: 1; min-width: 0; }
.proc-title {
  font-size: 13px; font-weight: 750;
  color: #93c5fd; margin-bottom: 3px;
}
.proc-sub { font-size: 11px; color: #cbd5e1; }
.proc-track {
  height: 4px; background: #172554;
  border-radius: 999px; overflow: hidden;
  margin-top: 8px;
}
.proc-fill {
  height: 100%; background: linear-gradient(90deg, #3b82f6, #06b6d4);
  border-radius: 999px;
  animation: pulse-bar 2.5s ease-in-out infinite;
}

.done-banner {
  display: flex; align-items: center; gap: 12px;
  background: linear-gradient(135deg, #062b19 0%, #064e2b 100%);
  border: 1px solid #16a34a;
  border-radius: 10px; padding: 12px 18px;
  font-family: var(--font);
  margin: 10px 0;
  animation: fade-slide .3s ease;
}
.done-icon { font-size: 22px; }
.done-title { font-size: 13px; font-weight: 750; color: #4ade80; }
.done-sub { font-size: 11px; color: #cbd5e1; margin-top: 2px; }

/* ── Relations & Audit Log Styling ──────────────────────────────────────── */
.rel-header {
  display: flex; justify-content: space-between; align-items: center;
  font-size: 12px; font-weight: 750; color: var(--text0);
  margin-bottom: 6px; padding-top: 4px;
}
.rel-badge {
  background: var(--bg3); color: var(--text1);
  border-radius: 999px; min-width: 22px; height: 22px;
  display: grid; place-items: center; font-size: 10px;
}
.rel-card {
  display: flex; justify-content: space-between; align-items: center;
  gap: 10px; padding: 7px 0; border-top: 1px solid #19232e;
}
.rel-text { font-size: 12px; line-height: 1.35; flex: 1; }
.rel-entity { font-weight: 650; color: #e2e8f0; }
.rel-pred { font-weight: 750; margin: 0 4px; }
.rel-pred.semantic { color: #4ade80; }
.rel-pred.spatial  { color: #fbbf24; }
.rel-score {
  width: 100px; display: flex; gap: 6px; align-items: center;
  color: var(--text2); font-size: 10px;
}
.score-track { flex:1; height:3px; background: #1e2d3d; border-radius:999px; overflow:hidden; }
.score-fill { height:100%; border-radius:999px; }
.score-fill.semantic { background: #4ade80; }
.score-fill.spatial  { background: #fbbf24; }

.stats-bar {
  display: flex; background: var(--bg0);
  border: 1px solid var(--border); border-radius: 8px; overflow: hidden;
  margin: 8px 0;
}
.stats-bar > div { flex: 1; padding: 8px 12px; border-right: 1px solid var(--border); }
.stats-bar > div:last-child { border: none; }
.stats-bar strong { display: block; font-size: 14px; color: var(--text0); }
.stats-bar span { color: var(--text2); font-size: 9px; white-space: nowrap; }

.audit-card {
  border-radius: 8px; padding: 10px 14px; margin-bottom: 8px;
  font-size: 12px; font-family: var(--font); color: var(--text0);
  border-left: 4px solid;
}
.audit-critical { background:#c084fc18; border-color:#c084fc; }
.audit-high     { background:#f8717118; border-color:#f87171; }
.audit-medium   { background:#fbbf2418; border-color:#fbbf24; }
.audit-low      { background:#4ade8018; border-color:#4ade80; }
.audit-scroll { max-height: 420px; overflow-y: auto; padding-right: 4px; }
.audit-scroll::-webkit-scrollbar { width: 4px; }
.audit-scroll::-webkit-scrollbar-track { background: var(--bg2); }
.audit-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius:999px; }

.hud-pill {
  display: inline-flex; align-items: center; gap: 8px;
  background: var(--bg3); border: 1px solid var(--border);
  border-radius: 8px; padding: 8px 14px; margin-bottom: 8px;
  font-family: var(--font); font-size: 12px; color: var(--text1);
}
.hud-pill strong { color: var(--accent2); font-size: 13px; }
"""


# ── Canvas JS for manual box drawing ─────────────────────────────────────────
CANVAS_JS = r"""
(function(){
  function findCheckbox(id){
    const h = document.getElementById(id);
    if(!h) return null;
    return h.querySelector('input[type=checkbox]') || h.querySelector('input');
  }
  function setup(){
    const hosts = document.querySelectorAll('#v2-source-image');
    let host=null;
    for(const h of hosts){ if(h.offsetParent!==null){host=h;break;} }
    if(!host||host.dataset.raCanvas) return;
    const img=host.querySelector('img');
    if(!img) return;
    host.dataset.raCanvas='1';
    if(getComputedStyle(host).position==='static') host.style.position='relative';
    const canvas=document.createElement('canvas');
    canvas.style.cssText='position:absolute;inset:0;pointer-events:none;z-index:50;';
    host.appendChild(canvas);
    const ctx=canvas.getContext('2d');
    const COLORS=['#3b82f6','#f97316','#22c55e','#a855f7','#eab308','#06b6d4'];
    let boxes=[],drag=null;
    function fit(){
      const r=img.getBoundingClientRect(),hr=host.getBoundingClientRect();
      canvas.style.left=(r.left-hr.left)+'px'; canvas.style.top=(r.top-hr.top)+'px';
      canvas.width=r.width; canvas.height=r.height;
      canvas.style.width=r.width+'px'; canvas.style.height=r.height+'px'; redraw();
    }
    function redraw(){
      ctx.clearRect(0,0,canvas.width,canvas.height);
      boxes.forEach((b,i)=>{
        const c=COLORS[i%COLORS.length];
        ctx.strokeStyle=c; ctx.lineWidth=2;
        ctx.strokeRect(b.x0,b.y0,b.x1-b.x0,b.y1-b.y0);
        ctx.fillStyle=c; ctx.fillRect(b.x0,Math.max(0,b.y0-18),24,18);
        ctx.fillStyle='#fff'; ctx.font='bold 11px sans-serif';
        ctx.fillText(String(i+1),b.x0+6,Math.max(13,b.y0-4));
      });
      if(drag){
        ctx.strokeStyle='#fff'; ctx.lineWidth=1.5; ctx.setLineDash([5,4]);
        ctx.strokeRect(drag.sx,drag.sy,drag.cx-drag.sx,drag.cy-drag.sy); ctx.setLineDash([]);
      }
    }
    function writeOut(){
      const lines=boxes.map(b=>[
        (b.x0/canvas.width).toFixed(4),(b.y0/canvas.height).toFixed(4),
        (b.x1/canvas.width).toFixed(4),(b.y1/canvas.height).toFixed(4)
      ].join(',')).join('\n');
      const ta=document.querySelector('#v2-manual-boxes textarea');
      if(!ta) return;
      const setter=Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype,'value').set;
      setter.call(ta,lines); ta.dispatchEvent(new Event('input',{bubbles:true}));
    }
    function drawActive(){ const cb=findCheckbox('v2-draw-mode'); return !!(cb&&cb.checked); }
    host.addEventListener('mousedown',(e)=>{
      if(!drawActive()||e.button!==0) return;
      const r=canvas.getBoundingClientRect();
      drag={sx:e.clientX-r.left,sy:e.clientY-r.top,cx:e.clientX-r.left,cy:e.clientY-r.top};
      e.preventDefault();
    });
    window.addEventListener('mousemove',(e)=>{
      if(!drag) return;
      const r=canvas.getBoundingClientRect();
      drag.cx=e.clientX-r.left; drag.cy=e.clientY-r.top; redraw();
    });
    window.addEventListener('mouseup',()=>{
      if(!drag) return;
      const x0=Math.min(drag.sx,drag.cx),x1=Math.max(drag.sx,drag.cx);
      const y0=Math.min(drag.sy,drag.cy),y1=Math.max(drag.sy,drag.cy);
      if(x1-x0>6&&y1-y0>6){ boxes.push({x0,y0,x1,y1}); writeOut(); }
      drag=null; redraw();
    });
    host.addEventListener('dblclick',(e)=>{
      if(!drawActive()) return; boxes=[]; writeOut(); redraw(); e.preventDefault();
    });
    img.addEventListener('load',fit);
    new ResizeObserver(fit).observe(img);
    window.addEventListener('resize',fit);
    setTimeout(fit,100);
  }
  function tick(){ try{setup();}catch(e){} }
  window.addEventListener('load',tick);
  setTimeout(tick,400);
  new MutationObserver(tick).observe(document.body,{childList:true,subtree:true});
})();
"""


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — Visual Presentation & HTML Rendering
# ═══════════════════════════════════════════════════════════════════════════

def _color(i: int):
    return PALETTE[i % len(PALETTE)]


def _pred_kind(pred: str) -> str:
    if PIPE is None:
        return "semantic"
    try:
        i = PIPE.ra.predicates.index(pred)
        return "spatial" if bool(PIPE.is_spatial[i]) else "semantic"
    except Exception:
        return "semantic"


def _proc_banner(label: str, sublabel: str = "") -> str:
    return (
        f'<div class="proc-banner">'
        f'<div class="proc-spinner"></div>'
        f'<div class="proc-text">'
        f'<div class="proc-title">{html.escape(label)}</div>'
        f'<div class="proc-sub">{html.escape(sublabel)}</div>'
        f'<div class="proc-track"><div class="proc-fill"></div></div>'
        f'</div></div>'
    )


def _done_banner(label: str, sub: str = "") -> str:
    return (
        f'<div class="done-banner">'
        f'<div class="done-icon">✅</div>'
        f'<div><div class="done-title">{html.escape(label)}</div>'
        f'<div class="done-sub">{html.escape(sub)}</div></div>'
        f'</div>'
    )


def _ph(msg: str) -> str:
    return (f'<div style="color:#64748b;font-size:12px;'
            f'font-family:Inter,sans-serif;padding:12px 4px">{msg}</div>')


def _relations_html(trips, labels_fn=None, section_title="all relations", kind="auto"):
    total = len(trips)
    out = [
        f'<div class="rel-header"><span>{section_title}</span>'
        f'<span class="rel-badge">{total}</span></div>'
    ]
    if not trips:
        out.append('<div style="color:#64748b;font-size:12px;padding:8px 0">No relations above threshold.</div>')
    for s, pred, o, sc in trips:
        k = _pred_kind(pred) if kind == "auto" else kind
        pct = max(0, min(100, int(float(sc) * 100)))
        sn = labels_fn(s) if labels_fn else f"obj {s}"
        on = labels_fn(o) if labels_fn else f"obj {o}"
        out.append(
            f'<div class="rel-card"><div class="rel-text">'
            f'<span class="rel-entity">{html.escape(str(sn))}</span>'
            f'<span class="rel-pred {k}"> {html.escape(str(pred))} </span>'
            f'<span class="rel-entity">{html.escape(str(on))}</span>'
            f'</div><div class="rel-score"><div class="score-track">'
            f'<div class="score-fill {k}" style="width:{pct}%"></div>'
            f'</div><span>{float(sc):.2f}</span></div></div>'
        )
    return "".join(out)


def _active_rels_chips_html(rels: List[Dict[str, Any]], ts: float) -> str:
    """Renders active relation chips for the scene graph panel at timestamp ts."""
    if not rels:
        return '<div style="color:#64748b;font-size:12px;padding:8px 2px">No active relations in this frame.</div>'
    chips = []
    for r in rels:
        sub = r.get("subject", "")
        pred = r.get("predicate", "")
        obj = r.get("object", "")
        k = _pred_kind(pred)
        col = "#fbbf24" if k == "spatial" else "#4ade80"
        chips.append(
            f'<div style="display:inline-flex;align-items:center;gap:6px;'
            f'background:#1a2234;border:1px solid #2d3b55;border-radius:6px;'
            f'padding:4px 9px;margin:3px 5px 3px 0;font-size:11px;font-family:Inter,sans-serif">'
            f'<span style="color:#e2e8f0;font-weight:600">{html.escape(sub)}</span>'
            f'<span style="color:{col};font-weight:750">──[{html.escape(pred)}]──></span>'
            f'<span style="color:#e2e8f0;font-weight:600">{html.escape(obj)}</span>'
            f'</div>'
        )
    hdr = f'<div style="font-size:10px;font-weight:800;color:#94a3b8;letter-spacing:.08em;margin-bottom:4px">ACTIVE RELATIONS IN SNAPSHOT ({len(rels)})</div>'
    return f'<div style="max-height:90px;overflow-y:auto;padding:2px 0">{hdr}' + "".join(chips) + '</div>'


def _build_dark_scene_graph(subgraph: nx.MultiDiGraph, ts: float) -> go.Figure:
    """Builds a Plotly scene graph styled with dark background matching the theme."""
    if len(subgraph.nodes) == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor='#0b1017',
            plot_bgcolor='#0b1017',
            height=360,
            margin=dict(l=10, r=10, t=35, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=[dict(
                text=f"No active relations at t={ts:.2f}s",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False,
                font=dict(color="#64748b", size=13, family="Inter, sans-serif")
            )]
        )
        return fig

    pos = nx.spring_layout(subgraph, seed=42)
    edge_x, edge_y, mid_x, mid_y, edge_text = [], [], [], [], []

    for u, v, data in subgraph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([float(x0), float(x1), None])
        edge_y.extend([float(y0), float(y1), None])
        mid_x.append(float((x0 + x1) / 2))
        mid_y.append(float((y0 + y1) / 2))
        edge_text.append(data.get("predicate", ""))

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.8, color='#3b82f6'),
        hoverinfo='none', mode='lines'
    )
    edge_label_trace = go.Scatter(
        x=mid_x, y=mid_y,
        mode='text', text=edge_text,
        textposition='top center', hoverinfo='none',
        textfont=dict(color='#4ade80', size=11, family='Inter, sans-serif')
    )

    node_x, node_y, node_text, node_colors = [], [], [], []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(float(x))
        node_y.append(float(y))
        node_text.append(str(node))
        # Color nodes by entity type
        if "person" in str(node).lower():
            node_colors.append('#38bdf8')
        elif any(k in str(node).lower() for k in ("phone", "mobile")):
            node_colors.append('#f87171')
        elif any(k in str(node).lower() for k in ("book", "paper")):
            node_colors.append('#c084fc')
        else:
            node_colors.append('#fbbf24')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text', text=node_text,
        textposition='bottom center',
        textfont=dict(color='#f1f5f9', size=11, family='Inter, sans-serif'),
        marker=dict(
            showscale=False,
            color=node_colors,
            size=22,
            line=dict(width=2, color='#ffffff')
        )
    )

    fig = go.Figure(
        data=[edge_trace, edge_label_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            paper_bgcolor='#0b1017',
            plot_bgcolor='#0b1017',
            height=360,
            margin=dict(l=10, r=10, t=35, b=10),
            title=dict(
                text=f"Active Scene Graph @ t={ts:.2f}s · {len(subgraph.nodes)} Entities · {len(subgraph.edges)} Relations",
                font=dict(size=12, color='#38bdf8', family='Inter, sans-serif'),
                x=0.02, y=0.98
            ),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )
    return fig


def _build_dark_timeline(events: List[TemporalRelation], duration: float) -> go.Figure:
    fig = render_timeline(events, duration)
    if fig and hasattr(fig, "update_layout"):
        fig.update_layout(
            paper_bgcolor='#0b1017',
            plot_bgcolor='#0b1017',
            font=dict(color='#94a3b8', family='Inter, sans-serif'),
            height=340,
            margin=dict(l=10, r=10, t=35, b=20),
            xaxis=dict(gridcolor='#1e293b', zerolinecolor='#334155', title="Time (seconds)"),
            yaxis=dict(gridcolor='#1e293b', title="Subject - Object"),
            title=dict(
                text=f"Temporal Event Gantt Timeline · Duration: {duration:.1f}s · {len(events)} Intervals",
                font=dict(size=13, color='#f1f5f9', family='Inter, sans-serif'),
                x=0.02
            )
        )
    return fig


# ── ExamGuard Audit Cards ───────────────────────────────────────────────────

_SEV_CLS = {1: "audit-low", 2: "audit-medium", 3: "audit-high", 4: "audit-critical"}
_SEV_ICON = {1: "🟢", 2: "🟡", 3: "🔴", 4: "🟣"}


def _violation_card(v: MalpracticeEvent, dim=False) -> str:
    cls = _SEV_CLS.get(v.severity, "audit-medium")
    icon = _SEV_ICON.get(v.severity, "⚠")
    op = "0.45" if dim else "1"
    subj = ", ".join(v.subjects)
    objs = ", ".join(v.objects) if v.objects else "—"
    conf_pct = int(v.confidence * 100)
    return (
        f'<div class="audit-card {cls}" style="opacity:{op}">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
        f'<span style="font-weight:800;font-size:11px">{icon} {v.rule_id} · {v.severity_label}</span>'
        f'<span style="color:#94a3b8;font-size:10px">{v.time_range} · {v.duration:.1f}s</span>'
        f'</div>'
        f'<div style="font-weight:600;margin-bottom:4px;font-size:12px">{html.escape(v.description)}</div>'
        f'<div style="display:flex;gap:12px;color:#64748b;font-size:11px;align-items:center">'
        f'<span>👤 {html.escape(subj)}</span>'
        f'<span>📦 {html.escape(objs)}</span>'
        f'<span style="margin-left:auto;font-weight:700;color:#94a3b8">{conf_pct}% conf</span>'
        f'</div>'
        f'</div>'
    )


def _audit_live_html(ts: float) -> str:
    if not STATE.violations:
        if STATE.manifest is None:
            return _ph("Load a run to start the audit.")
        return _ph("✅ No violations detected in this run.")
    active = violations_at_time(STATE.violations, ts, lookahead=0.3)
    upcoming = [v for v in STATE.violations
                if v.start_time > ts and v.start_time <= ts + 5.0]
    n, tot = len(active), len(STATE.violations)
    hdr = (
        f'<div style="display:flex;justify-content:space-between;'
        f'align-items:center;margin-bottom:8px">'
        f'<span style="font-weight:800;font-size:13px;color:#f1f5f9">'
        f'🔍 Active Violations @ {ts:.2f}s</span>'
        f'<span style="background:#1a2234;color:#94a3b8;border-radius:999px;'
        f'padding:3px 10px;font-size:11px">{n} active / {tot} total</span>'
        f'</div>'
    )
    body = ""
    if active:
        body += '<div style="font-size:9px;text-transform:uppercase;letter-spacing:.08em;color:#f87171;margin-bottom:5px;font-weight:700">⚠ ACTIVE NOW</div>'
        for v in active:
            body += _violation_card(v)
    if upcoming:
        body += '<div style="font-size:9px;text-transform:uppercase;letter-spacing:.08em;color:#fbbf24;margin:8px 0 5px;font-weight:700">⏱ UPCOMING (within 5s)</div>'
        for v in upcoming:
            body += _violation_card(v, dim=True)
    if not active and not upcoming:
        body += '<div style="color:#64748b;font-size:12px;padding:12px 0">✅ No active violations at this timestamp.</div>'
    return f'<div class="audit-scroll" style="font-family:Inter,sans-serif">{hdr}{body}</div>'


def _audit_summary_html(violations: List[MalpracticeEvent]) -> str:
    if not violations:
        return _ph("✅ No violations detected in this run.")
    counts = {}
    for v in violations:
        counts[v.severity] = counts.get(v.severity, 0) + 1
    _bg = {4: "#c084fc", 3: "#f87171", 2: "#fbbf24", 1: "#4ade80"}
    _tx = {4: "#1e1b4b", 3: "#450a0a", 2: "#451a03", 1: "#052e16"}
    pills = ""
    for sev in (4, 3, 2, 1):
        if counts.get(sev):
            lbl = ["", "LOW", "MED", "HIGH", "CRIT"][sev]
            pills += (f'<span style="background:{_bg[sev]};color:{_tx[sev]};'
                      f'border-radius:999px;padding:3px 9px;font-size:11px;'
                      f'font-weight:700;margin-right:4px">{counts[sev]} {lbl}</span>')
    hdr = (
        f'<div style="font-family:Inter,sans-serif;margin-bottom:8px">'
        f'<div style="font-weight:800;font-size:13px;color:#f1f5f9;margin-bottom:6px">'
        f'🛡 Full Run Audit — {len(violations)} Violation(s)</div>{pills}</div>'
    )
    cards = "".join(_violation_card(v) for v in violations)
    return f'<div class="audit-scroll" style="font-family:Inter,sans-serif">{hdr}{cards}</div>'


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — Frame Rendering & Scrubbing
# ═══════════════════════════════════════════════════════════════════════════

def _clean(trips, score_thr, top_k):
    keep = [(int(s), str(p), int(o), float(sc))
            for s, p, o, sc in trips if float(sc) >= float(score_thr)]
    return keep[:int(top_k)]


def _render_frame(res, show_masks=True, show_labels=True, label_mode="names"):
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
            from deploy.gradio_app_styled import COLOR_NAMES
            lab = f"{COLOR_NAMES[i % len(COLOR_NAMES)]} obj {i+1}" if label_mode == "colours" else (res.labels[i] if i < len(res.labels) else f"obj {i+1}")
            sc = float(res.scores[i]) if i < len(res.scores) else 1.0
            txt = f"{i}:{lab}" + (f" {sc:.2f}" if sc < 0.99 else "")
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), _color(i), -1)
            cv2.putText(img, txt, (x1 + 2, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    for s, pred, o, sc in getattr(res, "triplets", []):
        if s >= len(centers) or o >= len(centers):
            continue
        kind = _pred_kind(pred)
        col = (251, 191, 36) if kind == "spatial" else (74, 222, 128)
        p1, p2 = centers[s], centers[o]
        cv2.arrowedLine(img, p1, p2, col, max(1, int(1 + 2 * sc)), cv2.LINE_AA, tipLength=0.04)
        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        (tw, th), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (mid[0] - 2, mid[1] - th - 4), (mid[0] + tw + 4, mid[1] + 3), (20, 20, 20), -1)
        cv2.putText(img, pred, (mid[0] + 1, mid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    t = getattr(res, "timing", None)
    if t:
        hud = f"{t.fps:5.1f} FPS | det {t.det:5.1f}ms | rel {t.relation:5.1f}ms"
        cv2.rectangle(img, (0, 0), (W, 22), (10, 10, 10), -1)
        cv2.putText(img, hud, (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 220, 150), 1, cv2.LINE_AA)
    return img[:, :, ::-1]


def _parse_boxes(text, W, H):
    lines = [l.strip() for l in (text or "").splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return None, "no boxes"
    rows = []
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) != 4:
            return None, f"line {i+1}: need 4 numbers"
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            return None, f"line {i+1}: not numbers"
    arr = np.asarray(rows, np.float32)
    if float(arr.max()) <= 1.5:
        arr = arr * np.array([W, H, W, H], np.float32)
    arr[:, [0, 2]] = arr[:, [0, 2]].clip(0, W - 1)
    arr[:, [1, 3]] = arr[:, [1, 3]].clip(0, H - 1)
    return arr, None


def _stats_html(res):
    t = getattr(res, "timing", None)
    if not t: return ""
    det = "—" if t.det < 0.01 else f"{t.det:.0f}"
    bb  = "—" if t.backbone < 0.01 else f"{t.backbone:.0f}"
    return (
        '<div class="stats-bar">'
        f'<div><strong>{t.fps:.1f}</strong><span>FPS</span></div>'
        f'<div><strong>{det}</strong><span>det ms</span></div>'
        f'<div><strong>{bb}</strong><span>backbone ms</span></div>'
        f'<div><strong>{t.relation:.0f}</strong><span>rel ms</span></div>'
        f'<div><strong>{t.total:.0f}</strong><span>total ms</span></div>'
        f'<div><strong>{len(res.boxes_xyxy)}</strong><span>objects</span></div>'
        '</div>'
    )


# ── Live Inference Dispatchers ───────────────────────────────────────────────

def run_image(frame_rgb, conf_v, top_k_v, score_v, label_v):
    if frame_rgb is None or PIPE is None:
        err = _ph("No image or pipeline not ready")
        return None, "", err, err, err
    PIPE.cfg.det_conf = float(conf_v)
    res = PIPE(frame_rgb[:, :, ::-1].copy(), top_k=int(top_k_v),
               score_thr=float(score_v), decompose=True, spatial_drop_pair=False)
    merged = _clean(res.triplets, score_v, top_k_v)
    spa    = _clean(getattr(res, "triplets_spatial", []) or [], score_v, top_k_v)
    sem    = _clean(getattr(res, "triplets_semantic", []) or [], score_v, top_k_v)
    lf = (lambda i: res.labels[i] if i < len(res.labels) else f"obj {i}") if label_v == "names" else None
    return (
        _render_frame(res, True, True, label_v),
        _stats_html(res),
        _relations_html(merged, lf, "all relations"),
        _relations_html(spa, lf, "spatial", "spatial"),
        _relations_html(sem, lf, "semantic", "semantic"),
    )


def run_manual(frame_rgb, boxes_txt, top_k_v, score_v, label_v):
    if frame_rgb is None or PIPE is None:
        err = _ph("No image or pipeline not ready")
        return None, "", err, err, err
    H, W = frame_rgb.shape[:2]
    boxes, err = _parse_boxes(boxes_txt, W, H)
    if err or len(boxes) < 2:
        em = _ph(err or "Need at least 2 boxes")
        return None, "", em, em, em
    bgr = frame_rgb[:, :, ::-1].copy()
    t0 = time.perf_counter()
    graphs = PIPE.ra.predict(bgr, boxes, topk=int(top_k_v), max_boxes=len(boxes), decompose=True)
    dt = (time.perf_counter() - t0) * 1000.0
    def trips(g): return [(t.subject_idx, t.predicate, t.object_idx, float(t.score)) for t in g]
    spa_all = _clean(trips(graphs.get("spatial", [])), score_v, top_k_v)
    sem_all = _clean(trips(graphs.get("semantic", [])), score_v, top_k_v)
    merged, seen = [], set()
    for t in sorted(spa_all + sem_all, key=lambda x: -x[3]):
        k = (t[0], t[1], t[2])
        if k not in seen:
            seen.add(k); merged.append(t)
    merged = merged[:int(top_k_v)]

    class _Res: pass
    res = _Res()
    res.frame = bgr; res.boxes_xyxy = boxes
    res.labels = [f"box {i+1}" for i in range(len(boxes))]
    res.scores = np.ones(len(boxes), np.float32); res.masks = None
    res.triplets = merged; res.triplets_spatial = spa_all; res.triplets_semantic = sem_all

    class _T: pass
    tt = _T()
    tt.fps = 1000.0 / max(dt, 1e-3); tt.det = 0.0; tt.backbone = 0.0; tt.relation = dt; tt.total = dt
    res.timing = tt
    return (
        _render_frame(res, False, True, label_v),
        _stats_html(res),
        _relations_html(merged, None, "all relations"),
        _relations_html(spa_all, None, "spatial", "spatial"),
        _relations_html(sem_all, None, "semantic", "semantic"),
    )


def dispatch_image(frame, conf, topk, score, label, boxes_txt):
    if (boxes_txt or "").strip():
        return run_manual(frame, boxes_txt, topk, score, label)
    return run_image(frame, conf, topk, score, label)


def apply_vocab(classes_txt, preds_txt):
    if PIPE is None:
        return "⚠ pipeline not loaded"
    msgs = []
    try:
        cls = [c.strip() for c in classes_txt.replace("\n", ",").split(",") if c.strip()]
        if cls:
            PIPE.set_object_classes(cls)
            msgs.append(f"✓ detector → {len(cls)} classes")
    except Exception as e:
        msgs.append(f"✗ detector: {e}")
    try:
        prs = [p.strip() for p in preds_txt.replace("\n", ",").split(",") if p.strip()]
        if prs:
            PIPE.set_predicates(prs)
            msgs.append(f"✓ relations → {len(prs)} predicates")
    except Exception as e:
        msgs.append(f"✗ relations: {e}")
    return " · ".join(msgs)


# ── Generator-based Video Processor with Live Progress Animation ─────────────

def run_video_proc(video_path, conf_v, top_k_v, score_v, label_v, progress=gr.Progress()):
    """
    Generator yielding live progress HTML frame-by-frame so the UI stays animated and responsive.
    """
    if not video_path or PIPE is None:
        err = _ph("No video uploaded or pipeline is not ready." if not video_path else "Pipeline not loaded.")
        yield (None, err, err, err, err)
        return

    import tempfile, imageio
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        err = _ph("Could not open the uploaded video file.")
        yield (None, err, err, err, err)
        return

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    target_fps = 4.0
    stride     = max(1, int(round(src_fps / target_fps)))
    max_frames = 120
    n_to_proc  = min(max_frames, max(1, n_total // stride))
    PIPE.cfg.det_conf = float(conf_v)

    # IoU tracker
    _tracks, _counts = [], {}

    def _iou(a, b):
        x1 = max(float(a[0]), float(b[0])); y1 = max(float(a[1]), float(b[1]))
        x2 = min(float(a[2]), float(b[2])); y2 = min(float(a[3]), float(b[3]))
        iw = max(0., x2 - x1); ih = max(0., y2 - y1); inter = iw * ih
        ua = ((float(a[2])-float(a[0]))*(float(a[3])-float(a[1]))
              + (float(b[2])-float(b[0]))*(float(b[3])-float(b[1])) - inter)
        return inter / ua if ua > 0 else 0.

    def assign_ids(labels, boxes, fi):
        nonlocal _tracks
        _tracks = [t for t in _tracks if fi - t["lf"] <= 15]
        used, out = {}, {}
        for i, (lab, box) in enumerate(zip(labels, boxes)):
            bv, bj = -1, -1
            for j, t in enumerate(_tracks):
                if j in used or t["lab"] != lab: continue
                v = _iou(box, t["box"])
                if v > bv: bv, bj = v, j
            if bj >= 0 and bv >= 0.3:
                out[i] = _tracks[bj]["id"]; used[bj] = 1
                _tracks[bj]["box"] = box; _tracks[bj]["lf"] = fi
            else:
                n = _counts.get(lab, 0) + 1; _counts[lab] = n
                nid = f"{lab}_{n:02d}"
                _tracks.append({"id": nid, "lab": lab, "box": box, "lf": fi})
                out[i] = nid
        return out

    tmp = tempfile.mkdtemp(prefix="gev2_")
    out_path = os.path.join(tmp, "annotated.mp4")
    writer = imageio.get_writer(out_path, fps=target_fps, codec="libx264", quality=7, macro_block_size=None)

    all_rels, frame_records, processed, idx = [], [], 0, 0
    t_start = time.time()

    # Immediate initial banner
    yield (
        None,
        _proc_banner("⚙ Initializing Video Processing…",
                     f"Analyzing {n_to_proc} frames at {target_fps:.1f} FPS with open-vocabulary detector & RelateAnything head"),
        _ph("Processing in progress…"), _ph("Processing in progress…"), _ph("Processing in progress…")
    )

    while processed < max_frames:
        ret, bgr = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts  = idx / float(src_fps)
            try:
                res = PIPE(rgb[:, :, ::-1].copy(), top_k=int(top_k_v),
                           score_thr=float(score_v), decompose=True, spatial_drop_pair=False)
                writer.append_data(_render_frame(res, True, True, label_v))
                trip_list = [(int(s), str(p), int(o), float(sc)) for s, p, o, sc in res.triplets]
                all_rels.extend(trip_list)
                d2s = assign_ids(res.labels, res.boxes_xyxy, idx)
                frame_records.append({
                    "idx": idx, "ts": ts, "boxes": res.boxes_xyxy.tolist(),
                    "labels": list(res.labels), "det2semantic": d2s, "triplets": trip_list
                })
            except Exception as e:
                print(f"[v2 video] frame {idx} error: {e}")
                writer.append_data(rgb)

            processed += 1
            pct = min(processed / max(n_to_proc, 1), 0.99)
            elapsed = time.time() - t_start
            fps_proc = processed / max(elapsed, 0.1)

            # Live animation updates every 2 frames
            if processed % 2 == 0 or processed == 1 or processed == n_to_proc:
                progress(pct, desc=f"Frame {processed}/{n_to_proc}")
                status_msg = (
                    f"Frame {processed}/{n_to_proc} ({int(pct*100)}%) · "
                    f"t={ts:.1f}s · {len(all_rels)} relations · {fps_proc:.1f} FPS"
                )
                yield (
                    None,
                    _proc_banner(f"⚙ Processing Video: {int(pct*100)}% complete", status_msg),
                    _ph("Analyzing relation streams…"), _ph("Analyzing spatial streams…"), _ph("Analyzing semantic streams…")
                )
        idx += 1

    cap.release()
    writer.close()

    # Save run for temporal graph
    run_saved = None
    try:
        from deploy.live_to_godseye import save_run
        run_saved = save_run(video_path=video_path, sample_fps=target_fps,
                             frames_records=frame_records, score_threshold=float(score_v))
        STATE.last_processed_run = run_saved
        print(f"[v2] GodsEye run saved: {run_saved}")
    except Exception as e:
        print(f"[v2] run save failed: {e}")

    # Build unique relations
    by_key = {}
    for s, p, o, sc in all_rels:
        k = (s, p, o)
        if k not in by_key or sc > by_key[k][3]: by_key[k] = (s, p, o, sc)
    uniq = sorted(by_key.values(), key=lambda x: -x[3])[:int(top_k_v)]
    spa  = [t for t in uniq if _pred_kind(t[1]) == "spatial"]
    sem  = [t for t in uniq if _pred_kind(t[1]) == "semantic"]

    total_time = time.time() - t_start
    done_title = "✅ Video Processing Complete!"
    done_sub = (
        f"Processed {processed} frames in {total_time:.1f}s ({processed/max(total_time,0.1):.1f} FPS) · "
        f"Found {len(uniq)} unique relations"
        + (f" · Auto-saved to <strong>outputs/{run_saved}</strong>" if run_saved else "")
    )

    yield (
        out_path,
        _done_banner(done_title, done_sub),
        _relations_html(uniq, None, "all relations"),
        _relations_html(spa,  None, "spatial",  "spatial"),
        _relations_html(sem,  None, "semantic", "semantic")
    )


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Scene Graph & Timestamp Player Logic
# ═══════════════════════════════════════════════════════════════════════════

def list_runs() -> List[str]:
    base = Path("outputs")
    if not base.exists():
        return []
    runs = [d for d in base.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in runs]


def load_run(run_name: Optional[str]):
    """Loads a processed run and initializes all temporal analysis and audit panels."""
    if not run_name:
        empty_fig = go.Figure()
        return (
            gr.update(minimum=0, maximum=100, value=0, interactive=False),
            empty_fig, None, empty_fig,
            "No run selected.",
            _ph("No active relations."),
            _ph("Select and load a run to begin audit."),
            _ph("No run loaded.")
        )

    STATE.run_dir = Path("outputs") / run_name
    with open(STATE.run_dir / "manifest.json") as f:
        STATE.manifest = json.load(f)
    with open(STATE.run_dir / "scene_graph.json") as f:
        STATE.tsg = TemporalSceneGraph.from_dict(json.load(f))
    with open(STATE.run_dir / "events.json") as f:
        STATE.events = [TemporalRelation(**d) for d in json.load(f)]

    # Compute ExamGuard violations
    STATE.violations = run_exam_rules(STATE.events)
    vp = STATE.run_dir / "violations.json"
    vp.write_text(json.dumps([v.to_dict() for v in STATE.violations], indent=2))
    print(f"[ExamGuard] {len(STATE.violations)} violation(s) for '{run_name}'")

    total = STATE.manifest.get("total_frames", len(STATE.manifest["frames"]))
    fps   = STATE.manifest.get("sample_fps", 2.0)
    dur   = total / fps if fps > 0 else 0

    from temporal.video import VideoLoader
    STATE.video_loader = VideoLoader(STATE.manifest["video_path"], sample_fps=fps)

    tl_fig = _build_dark_timeline(STATE.events, dur)
    slider = gr.update(minimum=0, maximum=total - 1, value=0, step=1, interactive=True)

    try:
        init_img, init_sg, init_rels, init_ts = _frame_at(0)
    except Exception as e:
        print(f"[load_run] _frame_at(0) error: {e}")
        init_img, init_sg, init_rels, init_ts = None, go.Figure(), [], 0.0

    hud_status = f"✅ Loaded '{run_name}' · {total} frames · {dur:.1f}s video · {len(STATE.events)} temporal intervals"
    active_chips = _active_rels_chips_html(init_rels, init_ts)
    audit_live = _audit_live_html(0.0)
    audit_sum  = _audit_summary_html(STATE.violations)

    return slider, tl_fig, init_img, init_sg, hud_status, active_chips, audit_live, audit_sum


def _frame_at(frame_idx: int) -> Tuple[Optional[np.ndarray], go.Figure, List[Dict[str, Any]], float]:
    """Retrieves frame image, computes active scene graph, and returns active relations."""
    fi = int(frame_idx)
    fi = max(0, min(fi, len(STATE.manifest["frames"]) - 1))
    info = STATE.manifest["frames"][fi]
    ts = info["ts"]

    img = STATE.video_loader.get_frame(ts)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    sub = STATE.tsg.query_time(ts)
    rels = [
        {"subject": u, "object": v, "predicate": d.get("predicate", ""), "score": d.get("score", 0.0)}
        for u, v, d in sub.edges(data=True)
    ]

    boxes = np.array(info.get("boxes", []))
    d2s = {int(k): v for k, v in info.get("det2semantic", {}).items()}
    if img is not None and len(boxes) > 0:
        img = render_overlay(img, boxes, d2s, rels)

    sg_fig = _build_dark_scene_graph(sub, ts)
    return img, sg_fig, rels, ts


def on_frame_change(frame_idx: int):
    """Scrubber callback: updates frame, Plotly scene graph, relation chips, and active audit log."""
    if STATE.manifest is None:
        empty = go.Figure()
        return None, empty, "No run loaded.", _ph("No run loaded."), _ph("No run loaded.")
    try:
        img, fig, rels, ts = _frame_at(frame_idx)
        fi = int(frame_idx)
        total_f = len(STATE.manifest["frames"])
        status = f"⏱ Frame {fi+1} / {total_f} · Timestamp: {ts:.2f}s · {len(rels)} Active Relations"
        chips_html = _active_rels_chips_html(rels, ts)
        audit_html = _audit_live_html(ts)
        return img, fig, status, chips_html, audit_html
    except Exception as e:
        import traceback; traceback.print_exc()
        empty = go.Figure()
        return None, empty, f"Error: {e}", _ph(f"Error: {e}"), _ph(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# UI Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_ui():
    _REPO_LOCAL = Path(__file__).resolve().parents[2]
    sample_imgs = []
    for n in ("bicycle.jpg", "catlaptop.jpg", "frisbee.jpg", "horse.jpg", "skateboard.jpg", "tennis.jpg"):
        p = _REPO_LOCAL / "assets" / "reel" / "images" / n
        if p.is_file(): sample_imgs.append(str(p))

    with gr.Blocks(title="GodsEye v2 — Unified Dashboard") as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="gd-header">
          <div class="title-area">
            <div>
              <div class="title">👁 GodsEye v2</div>
              <div class="subtitle">Neuro-Symbolic Temporal Event Graphs & Exam Integrity Auditor</div>
            </div>
          </div>
          <span class="badge">ExamGuard-Causal Active</span>
        </div>
        """)

        # ── Main Tabs ─────────────────────────────────────────────────────
        with gr.Tabs(elem_classes="godseye-tabs") as main_tabs:

            # ════════════════════════════════════════════════════════════
            # TAB 1 — LIVE VIDEO & INFERENCE
            # ════════════════════════════════════════════════════════════
            with gr.Tab("🎬  Live Video & Inference", id="tab_live"):
                with gr.Row(equal_height=False):

                    # Main column
                    with gr.Column(scale=7):
                        with gr.Tabs():
                            with gr.Tab("🎬  Video"):
                                video_in = gr.Video(sources=["upload"], show_label=False,
                                                    height=380, elem_classes="gd-media")
                                with gr.Row():
                                    video_go = gr.Button("▶ Process Video", variant="primary",
                                                         elem_classes="btn-primary", scale=2)
                                    view_in_temporal_btn = gr.Button("📊 View in Temporal Scene Graph",
                                                                    variant="secondary",
                                                                    elem_classes="btn-secondary", scale=2)

                                # Live animation status banner
                                vid_status = gr.HTML(
                                    value='<div style="color:#64748b;font-size:12px;'
                                          'font-family:Inter,sans-serif;padding:8px 0">'
                                          '⬆ Upload video above and click <strong>Process Video</strong> to run the neuro-symbolic pipeline.</div>'
                                )
                                video_out = gr.Video(show_label=False, height=380,
                                                     interactive=False, elem_classes="gd-media")

                            with gr.Tab("🖼  Image"):
                                still = gr.Image(
                                    sources=["upload", "clipboard"],
                                    type="numpy", show_label=False, height=380,
                                    elem_id="v2-source-image",
                                    elem_classes="gd-media",
                                )
                                out_img = gr.Image(label=None, height=380, show_label=False,
                                                   elem_classes="gd-media")

                            with gr.Tab("📷  Webcam"):
                                cam = gr.Image(
                                    sources=["webcam"], streaming=True, type="numpy",
                                    show_label=False, height=380, elem_classes="gd-media",
                                )

                        stats_html = gr.HTML()

                        with gr.Tabs():
                            with gr.Tab("All Relations"):
                                rel_all = gr.HTML(elem_id="rel-panel-all")
                            with gr.Tab("Spatial 🟡"):
                                rel_spa = gr.HTML(elem_id="rel-panel-spatial")
                            with gr.Tab("Semantic 🟢"):
                                rel_sem = gr.HTML(elem_id="rel-panel-semantic")

                    # Side Controls
                    with gr.Column(scale=3, elem_id="side-panel-v2"):
                        gr.HTML('<div class="gd-panel-title">① LABELS</div>')
                        label_mode = gr.Radio(["names", "colours"], value="colours",
                                              label="Label objects by")

                        gr.HTML('<div class="gd-panel-title" style="margin-top:14px">② BOXES (OPTIONAL MANUAL)</div>')
                        draw_mode = gr.Checkbox(False, label="Draw on image",
                                                info="Drag rectangles on image. Double-click to clear.",
                                                elem_id="v2-draw-mode")
                        manual_boxes = gr.Textbox(
                            label="Manual boxes (x1,y1,x2,y2 per line)",
                            placeholder="0.30,0.10,0.70,0.60\n0.10,0.55,0.95,0.95",
                            lines=3, value="", elem_id="v2-manual-boxes",
                        )

                        if sample_imgs:
                            gr.HTML('<div class="gd-panel-title" style="margin-top:14px">③ SAMPLE IMAGES</div>')
                            gr.Examples(examples=[[p] for p in sample_imgs], inputs=still,
                                        label=None, examples_per_page=6, elem_id="sample-gallery")

                        gr.HTML('<div class="gd-panel-title" style="margin-top:14px">④ VOCABULARY</div>')
                        classes_txt = gr.Textbox(
                            label="Object classes",
                            placeholder="person, book, phone, desk...",
                            lines=2, value="",
                        )
                        preds_txt = gr.Textbox(
                            label="Predicates", lines=3,
                            value=", ".join(PREDICATE_PRESETS["Default"]),
                        )
                        vocab_msg = gr.Markdown(value="")

                        gr.HTML('<div class="gd-panel-title" style="margin-top:4px">PRESETS</div>')
                        with gr.Row(elem_classes="preset-row"):
                            for pname in PREDICATE_PRESETS:
                                def _handler(n=pname):
                                    def h(cls_txt):
                                        txt = ", ".join(PREDICATE_PRESETS[n])
                                        return txt, apply_vocab(cls_txt, txt)
                                    return h
                                gr.Button(pname, size="sm").click(
                                    _handler(), [classes_txt], [preds_txt, vocab_msg])
                        apply_btn = gr.Button("Apply Vocabularies", variant="primary",
                                              elem_classes="btn-primary")
                        apply_btn.click(apply_vocab, [classes_txt, preds_txt], vocab_msg)

                        gr.HTML('<div class="gd-panel-title" style="margin-top:14px">⑤ INFERENCE SETTINGS</div>')
                        conf    = gr.Slider(0.05, 0.9,  0.25, step=0.05, label="Detector confidence")
                        top_k   = gr.Slider(1,    30,   12,   step=1,    label="Max triplets")
                        score_t = gr.Slider(0.0,  0.95, 0.30, step=0.05, label="Relation score threshold")

            # ════════════════════════════════════════════════════════════
            # TAB 2 — TEMPORAL SCENE GRAPH & TIMELINE (LIKE DASHBOARD 1)
            # ════════════════════════════════════════════════════════════
            with gr.Tab("📊  Temporal Scene Graph & Timeline", id="tab_temporal"):

                # Run selector bar
                with gr.Row(elem_classes="run-row"):
                    runs_dd = gr.Dropdown(
                        choices=list_runs(), value=(list_runs() or [None])[0],
                        label="Select Processed Video Run", scale=4,
                    )
                    load_btn    = gr.Button("⏬ Load Run",     variant="primary", scale=1, elem_classes="btn-primary")
                    refresh_btn = gr.Button("🔄 Refresh List", scale=1, elem_classes="btn-secondary")

                # Frame status HUD banner
                t_status = gr.HTML(
                    value='<div class="hud-pill"><strong>Ready</strong> · Select a run and scrub the slider below.</div>'
                )

                # Timeline scrubber slider
                frame_slider = gr.Slider(
                    0, 100, step=1,
                    label="⏱ Scrub Video Timeline (Frame by Frame — Relation Graph Updates Instantly)",
                    interactive=True
                )

                # Side-by-side: Video Frame Overlay + Interactive NetworkX Scene Graph at this timestamp
                with gr.Row():
                    with gr.Column(scale=5, elem_id="temporal-frame-col"):
                        gr.HTML('<div class="gd-panel-title">📹 Video Frame with Detected Relations</div>')
                        video_frame = gr.Image(label="Annotated Frame Overlay", type="numpy",
                                                elem_classes="gd-media", height=360)

                    with gr.Column(scale=5, elem_id="temporal-graph-col"):
                        gr.HTML('<div class="gd-panel-title">🌐 Active Relation Graph at Current Timestamp</div>')
                        scene_graph = gr.Plot(label="Active Scene Graph (Plotly)")

                # Active Relations text chips for this timestamp
                active_rels_html = gr.HTML(
                    value='<div style="color:#64748b;font-size:12px;padding:4px 0">Active relations will appear here as you scrub.</div>'
                )

                # Gantt timeline chart across the entire video
                with gr.Row():
                    with gr.Column():
                        gr.HTML('<div class="gd-panel-title" style="margin-top:12px">📈 Temporal Event Gantt Chart (All Intervals)</div>')
                        tl_plot = gr.Plot(label="Temporal Event Gantt Chart")

                # Real-time Exam Integrity Audit Log directly on this tab!
                gr.HTML('<div class="gd-panel-title" style="margin-top:18px">🛡️ Real-Time Exam Integrity Audit Log</div>')
                with gr.Row():
                    with gr.Column(scale=5):
                        audit_live = gr.HTML(
                            value=_ph("Load a run to begin real-time audit."),
                        )
                    with gr.Column(scale=5):
                        audit_sum = gr.HTML(
                            value=_ph("Load a run to view full run summary."),
                        )

            # ════════════════════════════════════════════════════════════
            # TAB 3 — EXAMGUARD AUDIT DEEP-DIVE
            # ════════════════════════════════════════════════════════════
            with gr.Tab("🛡  ExamGuard Auditor", id="tab_audit"):
                gr.HTML("""
                <div style="font-family:Inter,sans-serif;padding:12px 0 6px">
                  <div style="font-size:18px;font-weight:800;color:#f1f5f9;margin-bottom:4px">
                    🛡 ExamGuard-Causal — Integrity Rule Engine
                  </div>
                  <div style="font-size:12px;color:#94a3b8">
                    Adjust neuro-symbolic thresholds and re-evaluate malpractice rules on the currently loaded video.
                  </div>
                </div>
                """)

                with gr.Row():
                    with gr.Column(scale=3):
                        gr.HTML('<div class="gd-panel-title">Rule Thresholds</div>')
                        thr_transfer = gr.Slider(2.0, 20.0, 8.0, step=0.5, label="RULE-1 Transfer window (s)")
                        thr_encrch   = gr.Slider(1.0, 10.0, 3.0, step=0.5, label="RULE-3 Encroachment dwell (s)")
                        thr_gaze     = gr.Slider(1.0, 10.0, 4.0, step=0.5, label="RULE-4b Gaze duration (s)")
                        thr_conf     = gr.Slider(0.05, 0.6, 0.25, step=0.05, label="Min relation confidence")
                        rerun_btn    = gr.Button("🔁 Re-evaluate Rules", variant="primary", elem_classes="btn-examguard")

                    with gr.Column(scale=7):
                        gr.HTML('<div class="gd-panel-title">📋 Full Audit Report</div>')
                        audit_deep_sum = gr.HTML(value=_ph("Load a run in the Temporal tab first."))

        # ═════════════════════════════════════════════════════════════════════
        # Event Wiring
        # ═════════════════════════════════════════════════════════════════════

        # Live Image & Camera Wiring
        _inf_inputs  = [conf, top_k, score_t, label_mode, manual_boxes]
        _inf_outputs = [out_img, stats_html, rel_all, rel_spa, rel_sem]

        still.change(dispatch_image, [still] + _inf_inputs, _inf_outputs)
        cam.stream(dispatch_image, [cam] + _inf_inputs, _inf_outputs,
                   stream_every=0.12, concurrency_limit=1, show_progress="hidden")
        for ctrl in [conf, top_k, score_t, label_mode, manual_boxes, draw_mode]:
            ctrl.change(dispatch_image, [still] + _inf_inputs, _inf_outputs)

        # Live Video Generator Wiring (responsive animated progress!)
        _vid_outputs = [video_out, vid_status, rel_all, rel_spa, rel_sem]
        video_go.click(
            run_video_proc,
            inputs=[video_in, conf, top_k, score_t, label_mode],
            outputs=_vid_outputs,
            show_progress="hidden",
        )

        # "View in Temporal Scene Graph" Button: automatically switches to the new run
        def _jump_to_temporal():
            runs = list_runs()
            target_run = STATE.last_processed_run or (runs[0] if runs else None)
            slider, tl, img, sg, status, chips, al, asum = load_run(target_run)
            return (
                gr.update(choices=runs, value=target_run),
                slider, tl, img, sg, status, chips, al, asum, asum
            )

        view_in_temporal_btn.click(
            _jump_to_temporal,
            inputs=None,
            outputs=[runs_dd, frame_slider, tl_plot, video_frame, scene_graph,
                     t_status, active_rels_html, audit_live, audit_sum, audit_deep_sum]
        )

        # Temporal Analysis Load & Refresh Wiring
        def _handle_load(run_name):
            slider, tl, img, sg, status, chips, al, asum = load_run(run_name)
            return slider, tl, img, sg, status, chips, al, asum, asum

        load_btn.click(
            _handle_load,
            inputs=[runs_dd],
            outputs=[frame_slider, tl_plot, video_frame, scene_graph,
                     t_status, active_rels_html, audit_live, audit_sum, audit_deep_sum]
        )

        refresh_btn.click(
            lambda: gr.update(choices=list_runs()),
            inputs=None,
            outputs=runs_dd
        )

        # Frame Scrubber Slider Wiring (Real-time update of frame, scene graph, chips, audit)
        frame_slider.change(
            on_frame_change,
            inputs=[frame_slider],
            outputs=[video_frame, scene_graph, t_status, active_rels_html, audit_live]
        )

        # Re-run Rules in Audit Tab
        def _rerun_rules(tw, em, gm, mc):
            if not STATE.events:
                msg = _ph("No events loaded. Load a video run first.")
                return msg, msg
            STATE.violations = run_exam_rules(
                STATE.events,
                transfer_window=tw,
                encroachment_min_sec=em,
                gaze_min_sec=gm,
                min_confidence=mc,
            )
            if STATE.run_dir:
                (STATE.run_dir / "violations.json").write_text(
                    json.dumps([v.to_dict() for v in STATE.violations], indent=2)
                )
            sum_html = _audit_summary_html(STATE.violations)
            live_html = _audit_live_html(0.0)
            return sum_html, live_html

        rerun_btn.click(
            _rerun_rules,
            inputs=[thr_transfer, thr_encrch, thr_gaze, thr_conf],
            outputs=[audit_deep_sum, audit_live]
        )

        # Auto-load default run on initial startup!
        def _init_page():
            runs = list_runs()
            if not runs:
                return gr.update(), None, None, None, "No runs found in outputs/.", _ph("No relations."), _ph("No runs."), _ph("No runs."), _ph("No runs.")
            default_run = runs[0]
            try:
                slider, tl, img, sg, status, chips, al, asum = load_run(default_run)
                return slider, tl, img, sg, status, chips, al, asum, asum
            except Exception as e:
                print(f"[_init_page] could not load '{default_run}': {e}")
                msg = "No run loaded — upload a video and click Process."
                return (gr.update(), None, None, None, msg,
                        _ph("No relations."), _ph("No runs."),
                        _ph("No runs."), _ph("No runs."))

        demo.load(
            _init_page,
            inputs=None,
            outputs=[frame_slider, tl_plot, video_frame, scene_graph,
                     t_status, active_rels_html, audit_live, audit_sum, audit_deep_sum]
        )

    return demo


# ═══════════════════════════════════════════════════════════════════════════
# Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="GodsEye Dashboard v2")
    ap.add_argument("--ckpt",   default="model/model.pth")
    ap.add_argument("--det",    default="checkpoints/detectors/yoloe-11s-seg-pf.pt")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--port",   type=int, default=7861)
    ap.add_argument("--share",  action="store_true")
    ap.add_argument("--no-pipeline", action="store_true",
                    help="Skip loading the inference pipeline (temporal tabs only)")
    a = ap.parse_args()

    if not a.no_pipeline:
        try:
            _load_pipe(a.ckpt, a.det, a.device)
        except Exception as e:
            print(f"[GodsEye v2] ⚠ pipeline load failed: {e}")
            print("[GodsEye v2]   Live Inference tab will be disabled. Use --no-pipeline to suppress this.")

    build_ui().queue(max_size=4).launch(
        server_name="127.0.0.1",
        server_port=a.port,
        share=a.share,
        show_error=True,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
        ),
        css=CSS,
        js=CANVAS_JS,
    )


if __name__ == "__main__":
    main()
