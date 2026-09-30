import argparse
import sys
import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal.config import GodsEyeConfig, TrackingConfig, RelationConfig, TemporalConfig, VideoConfig
from temporal.video import VideoLoader
from temporal.tracker import VideoTracker
from temporal.extractor import RelationExtractor
from temporal.belief import TemporalAggregator
from temporal.scene_graph import TemporalSceneGraph
from temporal.state_tracker import StateTracker

def main():
    parser = argparse.ArgumentParser(description="Process video into temporal event graph.")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument("--detector", type=str, default="checkpoints/detectors/yoloe-11m-seg-pf.pt", help="Path to YOLO weights")
    parser.add_argument("--remind-jsonl", type=str, default=None, help="Path to REMIND detections.jsonl. If set, skip detector + tracker and read detections from this file. The persistent object_id becomes the semantic ID suffix.")
    args = parser.parse_args()

    # Setup directories
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Configuration
    config = GodsEyeConfig(
        video=VideoConfig(sample_fps=2.0),
        tracking=TrackingConfig(detector_model=args.detector),
        relation=RelationConfig(device=args.device),
        temporal=TemporalConfig(use_belief=True)
    )

    print("Initializing pipeline components...")
    # Detector + tracker are loaded only when REMIND is not supplying them.
    remind_cache = None
    detector = None
    if args.remind_jsonl:
        from temporal.remind_adapter import RemindCache
        remind_cache = RemindCache(args.remind_jsonl)
        print(f"  REMIND cache: {args.remind_jsonl} "
              f"({len(remind_cache)} frames)  [detector + tracker skipped]")
    else:
        # Match deploy/pipeline.py's loader dispatch: YOLOE class for yoloe
        # checkpoints (handles prompt-free mode), YOLO for everything else.
        det_path = config.tracking.detector_model
        if "yoloe" in os.path.basename(det_path).lower():
            from ultralytics import YOLOE as _Det
        else:
            from ultralytics import YOLO as _Det
        detector = _Det(det_path)
        prompt_free = "-pf" in os.path.basename(det_path)
        print(f"  detector: {det_path}  (prompt_free={prompt_free})")
    
    loader = VideoLoader(args.video, sample_fps=config.video.sample_fps)
    tracker = None if remind_cache is not None else VideoTracker(config.tracking)
    extractor = RelationExtractor(config.relation)
    aggregator = TemporalAggregator(config.temporal)
    
    print(f"Processing video {args.video} at {config.video.sample_fps} FPS...")
    
    relations_log = open(out_dir / "relations.jsonl", "w")
    manifest = {
        "video_path": os.path.abspath(args.video),
        "sample_fps": config.video.sample_fps,
        "metadata": loader.metadata,
        "frames": []
    }
    
    # Track first_seen/last_seen per semantic ID for the scene graph.
    # In REMIND mode there is no registry to ask, so we build this here.
    seen_objects: dict[str, dict] = {}

    for frame_idx, timestamp, frame in tqdm(loader.stream_frames()):
        
        # 1. Detect + 2. Track & Re-ID
        H, W = frame.shape[:2]
        if remind_cache is not None:
            hit = remind_cache.det2semantic_for(timestamp)
            if hit is None:
                continue
            det2semantic, boxes, labels, confs = hit
            active_tracks = []
        else:
            res = detector(frame, verbose=False, device=config.relation.device)[0]
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            names = detector.names
            labels = [names[cls_id] for cls_id in cls_ids]
            det2semantic, active_tracks = tracker.process_frame(
                frame_idx, timestamp, boxes, labels, confs, W, H
            )

        # Record per-object first/last seen for the scene graph.
        for sem_id in det2semantic.values():
            cls = sem_id.rsplit("_", 1)[0]
            rec = seen_objects.get(sem_id)
            if rec is None:
                seen_objects[sem_id] = {
                    "class_name": cls,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                }
            else:
                rec["last_seen"] = timestamp
        active_semantic_ids = set(t.label for t in active_tracks) # Wait, tracker's label is original class.
        # We need semantic_ids for active tracks
        active_semantic_ids = set(det2semantic.values())
        
        # Save manifest entry with tracking data for the UI
        manifest["frames"].append({
            "idx": frame_idx, 
            "ts": timestamp, 
            "boxes": boxes.tolist(), 
            "det2semantic": {int(k): v for k, v in det2semantic.items()}
        })
        
        # 3. Extract Relations (includes raw logits)
        frame_rels, logits_np, pair_np, kof = extractor.extract(
            frame, boxes, det2semantic, timestamp, frame_idx
        )
        
        # Log frame relations
        for rel in frame_rels:
            relations_log.write(json.dumps(rel.__dict__) + "\n")
            
        class MockTriplet:
            def __init__(self, s, o, p, sc):
                self.subject_idx = s
                self.object_idx = o
                self.predicate = p
                self.score = sc
                
        # Format triplets for EdgeBook. EdgeBook expects subject_idx, object_idx, etc.
        mock_triplets = []
        for rel in frame_rels:
            mock_triplets.append(MockTriplet(rel.subject_idx, rel.object_idx, rel.predicate, rel.score))
            
        # 4. Temporal Aggregation
        vidx = {v: i for i, v in enumerate(extractor.vocab)}
        contract = extractor.ra.contract
        aggregator.observe_frame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            triplets=mock_triplets,
            det2track=det2semantic,
            active_tracks_ids=active_semantic_ids,
            raw=(logits_np, pair_np, kof, contract, vidx)
        )

    relations_log.close()
    
    # After all frames, extract temporal relations and build scene graph
    print("Building Temporal Scene Graph...")
    temporal_rels = aggregator.get_temporal_relations()
    
    tsg = TemporalSceneGraph()
    # Add objects
    if remind_cache is not None:
        # REMIND path: objects are whatever appeared in det2semantic.
        for obj_id, rec in seen_objects.items():
            tsg.add_object(obj_id, rec["class_name"],
                           rec["first_seen"], rec["last_seen"])
    else:
        # Built-in tracker path: pull full metadata from the registry.
        for obj_id, obj_data in tracker.get_active_objects().items():
            tsg.add_object(obj_id, obj_data.class_name,
                           obj_data.first_seen, obj_data.last_seen)
        
    # Add relations
    for rel in temporal_rels:
        tsg.add_relation(rel)
        
    # Save scene graph
    with open(out_dir / "scene_graph.json", "w") as f:
        json.dump(tsg.to_dict(), f, indent=2)
        
    # Also save raw temporal relations as json
    with open(out_dir / "events.json", "w") as f:
        json.dump([r.__dict__ for r in temporal_rels], f, indent=2)
    
    # Save manifest
    manifest["total_frames"] = len(manifest["frames"])
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Finished processing. Outputs saved to {out_dir}")

if __name__ == "__main__":
    main()
