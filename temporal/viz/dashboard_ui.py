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

class DashboardState:
    def __init__(self):
        self.manifest = None
        self.tsg = None
        self.events = None
        self.run_dir = None
        self.video_loader = None

STATE = DashboardState()


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
    return slider_update, timeline_fig, initial_img, initial_sg


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

    def _debug_update(frame_idx):
        import traceback
        msg = f"slider={frame_idx!r} (type={type(frame_idx).__name__})"
        print("[slider]", msg, flush=True)
        if STATE.manifest is None:
            return None, None, "no run loaded — click 'Load Run' first"
        try:
            img, fig = update_frame(frame_idx)
            status = f"{msg}\nimg={None if img is None else img.shape}\nfig={type(fig).__name__ if fig is not None else None}"
            return img, fig, status
        except Exception as e:
            traceback.print_exc()
            return None, None, f"{msg}\nERROR: {e}"

    def _debug_load(run_name):
        print("[load]", repr(run_name), flush=True)
        try:
            out = load_run(run_name)
            print("[load] ok", flush=True)
            return out
        except Exception as e:
            import traceback; traceback.print_exc()
            return gr.update(), None, None, None

    def _debug_refresh():
        return gr.update(choices=list_runs())

    load_btn.click(_debug_load, inputs=[run_dropdown],
                   outputs=[frame_slider, timeline, video_frame, scene_graph])
    refresh_btn.click(_debug_refresh, None, run_dropdown)
    frame_slider.change(_debug_update, inputs=[frame_slider],
                        outputs=[video_frame, scene_graph, debug_box])

    return run_dropdown


def main():
    with gr.Blocks(title="GodsEye") as demo:
        gr.Markdown("# GodsEye — Temporal Scene Graph")
        default_dd = build_godseye_ui()
        demo.load(
            lambda r: load_run(r) if r else (gr.update(), None, None, None),
            inputs=[default_dd],
            outputs=None,
        )
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()