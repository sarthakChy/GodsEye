import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from temporal.config import TrackingConfig
from deploy.render_video import Track

@dataclass
class TrackedObject:
    object_id: str
    class_name: str
    first_seen: float
    last_seen: float
    trajectory: List[Tuple[float, np.ndarray]] = field(default_factory=list)

def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate IoU between two bounding boxes (xyxy)."""
    x0 = max(box1[0], box2[0])
    y0 = max(box1[1], box2[1])
    x1 = min(box1[2], box2[2])
    y1 = min(box1[3], box2[3])
    
    inter_area = max(0, x1 - x0) * max(0, y1 - y0)
    if inter_area == 0:
        return 0.0
        
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(area1 + area2 - inter_area)

class Registry:
    """Maps tracker IDs → semantic IDs with re-ID across breaks."""
    
    def __init__(self, config: TrackingConfig):
        self.merge_window = config.merge_window
        self.merge_iou_threshold = config.merge_iou_threshold
        self.static_merge_window = config.static_merge_window
        self.dynamic_merge_window = config.dynamic_merge_window
        self.static_motion_threshold = config.static_motion_threshold
        self.dynamic_match_radius = config.dynamic_match_radius
        
        self._id_map: Dict[int, str] = {}           # tracker_id -> semantic_id
        self._class_counters: Dict[str, int] = {}   # class -> next index
        self.active_objects: Dict[str, TrackedObject] = {}
        
        # Lost tracks for re-id: (timestamp_lost, semantic_id, last_bbox)
        self._lost: List[Tuple[float, str, np.ndarray]] = []
        
    def _generate_semantic_id(self, class_name: str) -> str:
        count = self._class_counters.get(class_name, 0) + 1
        self._class_counters[class_name] = count
        return f"{class_name}_{count:02d}"

    def _classify_motion(self, obj: TrackedObject) -> str:
        if len(obj.trajectory) < 3:
            return "static"
        recent = obj.trajectory[-3:]
        centers = np.array([
            [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0]
            for _, b in recent
        ])
        disp = np.mean(np.linalg.norm(np.diff(centers, axis=0), axis=1))
        return "dynamic" if disp > self.static_motion_threshold else "static"

    def _match_score(self, t: Track, lost_bbox: np.ndarray, motion: str) -> float:
        if motion == "dynamic":
            t_c = np.array([(t.box[0] + t.box[2]) / 2.0, (t.box[1] + t.box[3]) / 2.0])
            l_c = np.array([(lost_bbox[0] + lost_bbox[2]) / 2.0,
                            (lost_bbox[1] + lost_bbox[3]) / 2.0])
            dist = float(np.linalg.norm(t_c - l_c))
            return max(0.0, 1.0 - dist / self.dynamic_match_radius)
        return _iou(t.box, lost_bbox)

    def _effective_window(self, obj: TrackedObject) -> int:
        return (self.dynamic_merge_window
                if self._classify_motion(obj) == "dynamic"
                else self.static_merge_window)

    def update(self, current_time: float, current_frame_idx: int, tracks: List[Track]) -> Dict[int, str]:
        """Process active tracks from the tracker and perform Re-ID if necessary.
        
        Args:
            current_time: current timestamp in seconds
            current_frame_idx: the frame index (can be used for merge window calculations if preferred, 
                               but we use timestamps or frame counts. Let's use frame counts for merge_window.)
            tracks: list of active Track objects from deploy.render_video.Tracker
            
        Returns:
            Dictionary mapping tracker_id -> semantic_id for this frame.
        """
        current_active_tids = set(t.tid for t in tracks)
        
        # 1. Check for newly lost tracks (were in id_map but not in current_active_tids)
        newly_lost_tids = set(self._id_map.keys()) - current_active_tids
        for tid in newly_lost_tids:
            semantic_id = self._id_map.pop(tid)
            obj = self.active_objects[semantic_id]
            if obj.trajectory:
                last_bbox = obj.trajectory[-1][1]
                self._lost.append((current_frame_idx, semantic_id, last_bbox))
        
        # Clean up old lost tracks using per-object effective windows.
        kept_lost = []
        for f_idx, s_id, bbox in self._lost:
            obj = self.active_objects.get(s_id)
            if obj is None:
                continue
            window = self._effective_window(obj)
            if current_frame_idx - f_idx <= window:
                kept_lost.append((f_idx, s_id, bbox))
        self._lost = kept_lost

        # 2. Process current tracks
        track_to_semantic: Dict[int, str] = {}

        for t in tracks:
            if t.hits < 2:
                continue

            if t.tid in self._id_map:
                semantic_id = self._id_map[t.tid]
            else:
                matched_semantic_id = None
                best_score = 0.0
                best_lost_idx = -1

                for i, (lost_f_idx, lost_s_id, lost_bbox) in enumerate(self._lost):
                    lost_obj = self.active_objects[lost_s_id]
                    if lost_obj.class_name != t.label:
                        continue
                    motion = self._classify_motion(lost_obj)
                    score = self._match_score(t, lost_bbox, motion)
                    threshold = self.merge_iou_threshold if motion == "static" else 0.5
                    if score > threshold and score > best_score:
                        best_score = score
                        matched_semantic_id = lost_s_id
                        best_lost_idx = i

                if matched_semantic_id:
                    semantic_id = matched_semantic_id
                    self._lost.pop(best_lost_idx)
                else:
                    semantic_id = self._generate_semantic_id(t.label)
                    self.active_objects[semantic_id] = TrackedObject(
                        object_id=semantic_id,
                        class_name=t.label,
                        first_seen=current_time,
                        last_seen=current_time,
                    )
                self._id_map[t.tid] = semantic_id
            
            # Update object trajectory and last seen
            obj = self.active_objects[semantic_id]
            obj.last_seen = current_time
            obj.trajectory.append((current_time, t.box))
            
            track_to_semantic[t.tid] = semantic_id
            
        return track_to_semantic
