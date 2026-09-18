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

# Global state for the dashboard
class DashboardState:
    def __init__(self):
        self.manifest = None
        self.tsg = None
        self.events = None
        self.run_dir = None
        self.video_loader = None
        
STATE = DashboardState()

def list_runs():
    base = Path("outputs")
    if not base.exists():
        return []
    return [d.name for d in base.iterdir() if d.is_dir() and (d / "manifest.json").exists()]

def load_run(run_name):
    STATE.run_dir = Path("outputs") / run_name
    
    with open(STATE.run_dir / "manifest.json", "r") as f:
        STATE.manifest = json.load(f)
        
    with open(STATE.run_dir / "scene_graph.json", "r") as f:
        tsg_dict = json.load(f)
        STATE.tsg = TemporalSceneGraph.from_dict(tsg_dict)
        
    with open(STATE.run_dir / "events.json", "r") as f:
        events_dict = json.load(f)
        STATE.events = [TemporalRelation(**d) for d in events_dict]
        
    total_frames = STATE.manifest.get("total_frames", len(STATE.manifest["frames"]))
    sample_fps = STATE.manifest.get("sample_fps", 2.0)
    
    # Initialize VideoLoader for scrubbing
    from temporal.video import VideoLoader
    STATE.video_loader = VideoLoader(STATE.manifest["video_path"], sample_fps=sample_fps)
    
    video_duration = total_frames / sample_fps if sample_fps > 0 else 0
    timeline_fig = render_timeline(STATE.events, video_duration)
    
    slider_update = gr.Slider(minimum=0, maximum=total_frames-1, value=0, step=1, interactive=True)
    
    # Force initial frame render so UI doesn't look broken
    initial_img, initial_sg = update_frame(0)
    
    return slider_update, timeline_fig, initial_img, initial_sg
    
def update_frame(frame_idx):
    if not STATE.manifest or not STATE.video_loader:
        return None, None
        
    frame_info = STATE.manifest["frames"][frame_idx]
    ts = frame_info["ts"]
    
    # Get frame using imageio_ffmpeg for deterministic resizing
    img = STATE.video_loader.get_frame(ts)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Query graph
    subgraph = STATE.tsg.query_time(ts)
    
    # Extract active relations for the overlay
    active_rels = []
    for u, v, d in subgraph.edges(data=True):
        active_rels.append({
            "subject": u,
            "object": v,
            "predicate": d.get("predicate", "")
        })
        
    # Render overlay using the stored boxes from manifest
    boxes = np.array(frame_info.get("boxes", []))
    det2semantic = frame_info.get("det2semantic", {})
    # JSON dict keys are strings, convert to int
    det2semantic = {int(k): v for k, v in det2semantic.items()}
    
    if len(boxes) > 0 and img is not None:
        img = render_overlay(img, boxes, det2semantic, active_rels)
    
    sg_fig = render_graph(subgraph)
    
    return img, sg_fig

def build_godseye_ui():
    gr.Markdown("---")
    gr.Markdown("## GodsEye Temporal Video Analysis")
    
    with gr.Row():
        run_dropdown = gr.Dropdown(choices=list_runs(), label="Select Processed Video Run")
        refresh_btn = gr.Button("Refresh List")
        
    with gr.Row():
        with gr.Column(scale=1):
            frame_slider = gr.Slider(0, 100, step=1, label="Timeline (Frame)")
            video_frame = gr.Image(label="Video Frame", type="numpy")
            
        with gr.Column(scale=1):
            scene_graph = gr.Plot(label="Active Scene Graph")
            
    with gr.Row():
        timeline = gr.Plot(label="Temporal Event Timeline")
        
    # Wiring
    refresh_btn.click(lambda: gr.Dropdown(choices=list_runs()), None, run_dropdown)
    run_dropdown.change(load_run, inputs=[run_dropdown], outputs=[frame_slider, timeline, video_frame, scene_graph])
    frame_slider.change(update_frame, inputs=[frame_slider], outputs=[video_frame, scene_graph])
