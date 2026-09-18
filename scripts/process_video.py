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
    parser.add_argument("--detector", type=str, default="yolov8m.pt", help="Path to YOLO weights")
    args = parser.parse_args()

    # Setup directories
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    
    # Configuration
    config = GodsEyeConfig(
        video=VideoConfig(sample_fps=2.0),
        tracking=TrackingConfig(detector_model=args.detector),
        relation=RelationConfig(device=args.device),
        temporal=TemporalConfig(use_belief=True)
    )

    print("Initializing pipeline components...")
    # Using Ultralytics YOLO directly for detection since Tracker just needs (boxes, labels, confs)
    from ultralytics import YOLO
    detector = YOLO(config.tracking.detector_model)
    
    loader = VideoLoader(args.video, sample_fps=config.video.sample_fps)
    tracker = VideoTracker(config.tracking)
    extractor = RelationExtractor(config.relation)
    aggregator = TemporalAggregator(config.temporal)
    
    print(f"Processing video {args.video} at {config.video.sample_fps} FPS...")
    
    relations_log = open(out_dir / "relations.jsonl", "w")
    manifest = {
        "video_path": args.video,
        "metadata": loader.metadata,
        "frames": []
    }
    
    for frame_idx, timestamp, frame in tqdm(loader.stream_frames()):
        # Save frame
        frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(frame_path), frame)
        manifest["frames"].append({"idx": frame_idx, "ts": timestamp, "file": frame_path.name})
        
        # 1. Detect
        H, W = frame.shape[:2]
        res = detector(frame, verbose=False, device=config.relation.device)[0]
        
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        cls_ids = res.boxes.cls.cpu().numpy().astype(int)
        names = detector.names
        labels = [names[cls_id] for cls_id in cls_ids]
        
        # 2. Track & Re-ID
        det2semantic, active_tracks = tracker.process_frame(
            frame_idx, timestamp, boxes, labels, confs, W, H
        )
        active_semantic_ids = set(t.label for t in active_tracks) # Wait, tracker's label is original class.
        # We need semantic_ids for active tracks
        active_semantic_ids = set(det2semantic.values())
        
        # 3. Extract Relations (includes raw logits)
        frame_rels, logits_np, pair_np, kof = extractor.extract(
            frame, boxes, det2semantic, timestamp, frame_idx
        )
        
        # Log frame relations
        for rel in frame_rels:
            relations_log.write(json.dumps(rel.__dict__) + "\n")
            
        class MockTriplet:
            def __init__(self, s, o, p, sc):
                self.sub = s
                self.obj = o
                self.pred = p
                self.score = sc
                
        # Format triplets for EdgeBook. EdgeBook expects subject_idx, object_idx, etc.
        # But we adapted it to use semantic IDs. We just need an object with sub, obj, pred, score
        mock_triplets = []
        for rel in frame_rels:
            mock_triplets.append(MockTriplet(rel.subject_id, rel.object_id, rel.predicate, rel.score))
            
        # 4. Temporal Aggregation
        aggregator.observe_frame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            triplets=mock_triplets,
            det2track=det2semantic,
            active_tracks_ids=active_semantic_ids,
            raw=(logits_np, pair_np, kof)
        )

    relations_log.close()
    
    # After all frames, extract temporal relations and build scene graph
    print("Building Temporal Scene Graph...")
    temporal_rels = aggregator.get_temporal_relations()
    
    tsg = TemporalSceneGraph()
    # Add objects
    for obj_id, obj_data in tracker.get_active_objects().items():
        tsg.add_object(obj_id, obj_data.class_name, obj_data.first_seen, obj_data.last_seen)
        
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
