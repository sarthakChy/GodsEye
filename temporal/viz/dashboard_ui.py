"""
GodsEye Temporal Dashboard — with ExamGuard Audit Log
======================================================
Adds a new "Exam Integrity Audit Log" row beneath the existing Gantt timeline.
The audit log updates in real-time when the frame slider is moved, showing
only violations that are *active* at the current playhead timestamp.
"""

import gradio as gr
import json
import cv2
import os
from pathlib import Path
import numpy as np

from temporal.viz.overlay import render_overlay
from temporal.viz.graph_renderer import render_graph
from temporal.scene_graph import TemporalSceneGraph
from temporal.viz.timeline import render_timeline
from temporal.schema import TemporalRelation
from temporal.exam_rules import run_exam_rules, violations_at_time, MalpracticeEvent
from typing import List


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

class DashboardState:
    def __init__(self):
        self.manifest = None
        self.tsg = None
        self.events = None
        self.run_dir = None
        self.video_loader = None
        self.violations: List[MalpracticeEvent] = []

STATE = DashboardState()


# ---------------------------------------------------------------------------
# Run management
# ---------------------------------------------------------------------------

def list_runs():
    """Return run names, most recently processed first."""
    base = Path("outputs")
    if not base.exists():
        return []
    runs = [d for d in base.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    runs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return [d.name for d in runs]


def load_run(run_name):
    if not run_name:
        raise ValueError("No run selected")
    STATE.run_dir = Path("outputs") / run_name

    with open(STATE.run_dir / "manifest.json", "r") as f:
        STATE.manifest = json.load(f)
    with open(STATE.run_dir / "scene_graph.json", "r") as f:
        STATE.tsg = TemporalSceneGraph.from_dict(json.load(f))
    with open(STATE.run_dir / "events.json", "r") as f:
        STATE.events = [TemporalRelation(**d) for d in json.load(f)]

    # --- Run ExamGuard rules over the full event list ---
    STATE.violations = run_exam_rules(STATE.events)
    print(f"[ExamGuard] {len(STATE.violations)} violation(s) detected for run '{run_name}'")

    # Optionally persist to disk so the CLI can inspect later
    viol_path = STATE.run_dir / "violations.json"
    viol_path.write_text(json.dumps([v.to_dict() for v in STATE.violations], indent=2))

    total_frames = STATE.manifest.get("total_frames", len(STATE.manifest["frames"]))
    sample_fps = STATE.manifest.get("sample_fps", 2.0)

    from temporal.video import VideoLoader
    STATE.video_loader = VideoLoader(STATE.manifest["video_path"], sample_fps=sample_fps)

    video_duration = total_frames / sample_fps if sample_fps > 0 else 0
    timeline_fig = render_timeline(STATE.events, video_duration)

    slider_update = gr.update(minimum=0, maximum=total_frames - 1, value=0, step=1, interactive=True)
    try:
        initial_img, initial_sg = update_frame(0)
    except Exception:
        import traceback; traceback.print_exc()
        initial_img, initial_sg = None, None

    audit_html = _render_audit_summary(STATE.violations)
    return slider_update, timeline_fig, initial_img, initial_sg, audit_html


# ---------------------------------------------------------------------------
# Frame update
# ---------------------------------------------------------------------------

def update_frame(frame_idx):
    if not STATE.manifest or not STATE.video_loader:
        return None, None
    try:
        frame_idx = int(frame_idx)
        frame_idx = max(0, min(frame_idx, len(STATE.manifest["frames"]) - 1))
        frame_info = STATE.manifest["frames"][frame_idx]
        ts = frame_info["ts"]

        img = STATE.video_loader.get_frame(ts)
        if img is None:
            return None, None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        subgraph = STATE.tsg.query_time(ts)
        active_rels = [
            {"subject": u, "object": v, "predicate": d.get("predicate", "")}
            for u, v, d in subgraph.edges(data=True)
        ]
        boxes = np.array(frame_info.get("boxes", []))
        det2semantic = {int(k): v for k, v in frame_info.get("det2semantic", {}).items()}
        if len(boxes) > 0:
            img = render_overlay(img, boxes, det2semantic, active_rels)

        sg_fig = render_graph(subgraph)
        return img, sg_fig
    except Exception:
        import traceback
        traceback.print_exc()
        return None, None


# ---------------------------------------------------------------------------
# Audit Log HTML rendering
# ---------------------------------------------------------------------------

# Severity colours matching a dark-panel aesthetic
_SEV_COLOR = {
    1: ("#ffd97d", "#3d2f00"),   # LOW    – amber
    2: ("#ffad5a", "#3d1e00"),   # MEDIUM – orange
    3: ("#ff6b6b", "#3d0000"),   # HIGH   – red
    4: ("#c678dd", "#2d0033"),   # CRITICAL – purple
}


def _render_violation_card(v: MalpracticeEvent, active: bool = True) -> str:
    bg, text = _SEV_COLOR.get(v.severity, ("#444", "#ddd"))
    opacity = "1.0" if active else "0.4"
    conf_pct = int(v.confidence * 100)
    subjects = ", ".join(v.subjects)
    objects  = ", ".join(v.objects) if v.objects else "—"
    card = f"""
<div style="
    background:{bg}22;
    border-left:4px solid {bg};
    border-radius:6px;
    padding:9px 12px;
    margin-bottom:7px;
    opacity:{opacity};
    font-family:Inter,ui-sans-serif,system-ui,sans-serif;
    font-size:12px;
    color:#dce4ed;
">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
    <span style="
        background:{bg};color:{text};
        border-radius:4px;padding:2px 7px;
        font-weight:800;font-size:10px;letter-spacing:.06em
    ">{v.rule_id} · {v.severity_label}</span>
    <span style="color:#9aacbc;font-size:10px">{v.time_range} · {v.duration:.1f}s</span>
  </div>
  <div style="font-weight:600;color:#edf2f7;margin-bottom:4px">{v.description}</div>
  <div style="display:flex;gap:16px;color:#8d9aaa;font-size:11px">
    <span>👤 {subjects}</span>
    <span>📦 {objects}</span>
    <div style="flex:1;text-align:right">
      <span style="color:{bg};font-weight:700">{conf_pct}%</span>
      <span> conf</span>
    </div>
  </div>
</div>"""
    return card


def _render_audit_live(timestamp: float) -> str:
    """HTML for active-at-timestamp panel (called on slider change)."""
    if not STATE.violations:
        if STATE.manifest is None:
            return _audit_placeholder("Load a run to start the audit.")
        return _audit_placeholder("✅ No violations detected in this run.")

    active = violations_at_time(STATE.violations, timestamp, lookahead=0.5)
    upcoming = [v for v in STATE.violations
                if v.start_time > timestamp and v.start_time <= timestamp + 5.0]

    total = len(STATE.violations)
    n_active = len(active)

    header = (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:10px;font-family:Inter,sans-serif">'
        f'<span style="color:#f3f6f9;font-weight:750;font-size:13px">🔍 Violations @ {timestamp:.2f}s</span>'
        f'<span style="background:#1b2632;color:#9aa8b8;border-radius:999px;'
        f'padding:2px 10px;font-size:11px">{n_active} active / {total} total</span>'
        f'</div>'
    )

    body = ""
    if active:
        body += '<div style="color:#8d9aaa;font-size:10px;text-transform:uppercase;'
        body += 'letter-spacing:.08em;margin-bottom:5px">⚠ ACTIVE NOW</div>'
        for v in active:
            body += _render_violation_card(v, active=True)

    if upcoming:
        body += '<div style="color:#8d9aaa;font-size:10px;text-transform:uppercase;'
        body += 'letter-spacing:.08em;margin:8px 0 5px">⏱ UPCOMING (within 5s)</div>'
        for v in upcoming:
            body += _render_violation_card(v, active=False)

    if not active and not upcoming:
        body += ('<div style="color:#748295;font-size:12px;padding:12px 0">'
                 '✅ No active violations at this timestamp.</div>')

    return f'<div style="padding:4px 2px">{header}{body}</div>'


def _render_audit_summary(violations: List[MalpracticeEvent]) -> str:
    """HTML summary table shown immediately after a run is loaded."""
    if not violations:
        return _audit_placeholder("✅ No violations detected in this run.")

    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for v in violations:
        counts[v.severity] = counts.get(v.severity, 0) + 1

    pills = ""
    for sev in (4, 3, 2, 1):
        if counts[sev]:
            bg, text = _SEV_COLOR[sev]
            label = ["", "LOW", "MEDIUM", "HIGH", "CRITICAL"][sev]
            pills += (f'<span style="background:{bg};color:{text};border-radius:999px;'
                      f'padding:3px 10px;font-size:11px;font-weight:700;margin-right:5px">'
                      f'{counts[sev]} {label}</span>')

    header = (
        f'<div style="font-family:Inter,sans-serif;margin-bottom:10px">'
        f'<div style="color:#f3f6f9;font-weight:750;font-size:13px;margin-bottom:6px">'
        f'🔍 Exam Audit — {len(violations)} Violation(s)</div>'
        f'<div>{pills}</div></div>'
    )

    cards = "".join(_render_violation_card(v) for v in violations)
    return f'<div style="padding:4px 2px">{header}{cards}</div>'


def _audit_placeholder(msg: str) -> str:
    return (
        f'<div style="color:#748295;font-size:12px;'
        f'font-family:Inter,sans-serif;padding:14px 4px">{msg}</div>'
    )


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

AUDIT_CSS = """
#exam-audit-panel {
    background: #0b1017 !important;
    border: 1px solid #1e2935 !important;
    border-radius: 10px !important;
    padding: 12px !important;
    min-height: 160px;
    max-height: 480px;
    overflow-y: auto;
}
"""


def build_godseye_ui():
    gr.Markdown("## GodsEye Temporal Video Analysis")

    runs = list_runs()
    default = runs[0] if runs else None

    with gr.Row():
        run_dropdown = gr.Dropdown(choices=runs, value=default, label="Select Processed Video Run")
        load_btn = gr.Button("Load Run", variant="primary")
        refresh_btn = gr.Button("Refresh List")

    with gr.Row():
        with gr.Column(scale=1):
            frame_slider = gr.Slider(0, 100, step=1, label="Timeline (Frame)")
            video_frame = gr.Image(label="Video Frame", type="numpy")
            debug_box = gr.Textbox(label="Debug", interactive=False)
        with gr.Column(scale=1):
            scene_graph = gr.Plot(label="Active Scene Graph")

    with gr.Row():
        timeline = gr.Plot(label="Temporal Event Timeline")

    # ---- ExamGuard Audit Log panel ----------------------------------------
    gr.Markdown("### 🛡️ Exam Integrity Audit Log")
    with gr.Row():
        with gr.Column(scale=2):
            audit_live = gr.HTML(
                value=_audit_placeholder("Load a run to start the audit."),
                label="Active Violations at Current Timestamp",
                elem_id="exam-audit-panel",
            )
        with gr.Column(scale=1):
            audit_summary = gr.HTML(
                value=_audit_placeholder("Load a run to see the full audit summary."),
                label="Full Run Summary",
                elem_id="exam-audit-panel",
            )

    # ---- Callbacks --------------------------------------------------------

    def _debug_update(frame_idx):
        import traceback
        msg = f"slider={frame_idx!r} (type={type(frame_idx).__name__})"
        print("[slider]", msg, flush=True)
        if STATE.manifest is None:
            return None, None, "no run loaded — click 'Load Run' first", _audit_placeholder("Load a run first.")
        try:
            img, fig = update_frame(frame_idx)
            ts = STATE.manifest["frames"][int(frame_idx)]["ts"] if STATE.manifest else 0.0
            status = f"{msg}\nimg={None if img is None else img.shape}\nfig={type(fig).__name__ if fig is not None else None}"
            audit_html = _render_audit_live(ts)
            return img, fig, status, audit_html
        except Exception as e:
            traceback.print_exc()
            return None, None, f"{msg}\nERROR: {e}", _audit_placeholder(f"Error: {e}")

    def _debug_load(run_name):
        print("[load]", repr(run_name), flush=True)
        try:
            slider_upd, tl_fig, img, sg, audit_html = load_run(run_name)
            print("[load] ok", flush=True)
            # summary is the full-run audit; live starts at t=0
            live_html = _render_audit_live(0.0)
            return slider_upd, tl_fig, img, sg, live_html, audit_html
        except Exception as e:
            import traceback; traceback.print_exc()
            return gr.update(), None, None, None, _audit_placeholder(f"Load failed: {e}"), ""

    def _debug_refresh():
        return gr.update(choices=list_runs())

    load_btn.click(
        _debug_load,
        inputs=[run_dropdown],
        outputs=[frame_slider, timeline, video_frame, scene_graph, audit_live, audit_summary],
    )
    refresh_btn.click(_debug_refresh, None, run_dropdown)
    frame_slider.change(
        _debug_update,
        inputs=[frame_slider],
        outputs=[video_frame, scene_graph, debug_box, audit_live],
    )

    return run_dropdown


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    with gr.Blocks(title="GodsEye", css=AUDIT_CSS) as demo:
        gr.Markdown("# GodsEye — Temporal Scene Graph & ExamGuard")
        default_dd = build_godseye_ui()
        demo.load(
            lambda r: load_run(r) if r else (gr.update(), None, None, None,
                                              _audit_placeholder("No run."),
                                              _audit_placeholder("No run.")),
            inputs=[default_dd],
            outputs=None,
        )
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
