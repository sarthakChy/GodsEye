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
        
        self._id_map: Dict[int, str] = {}           # tracker_id -> semantic_id
        self._class_counters: Dict[str, int] = {}   # class -> next index
        self.active_objects: Dict[str, TrackedObject] = {}
        
        # Lost tracks for re-id: (timestamp_lost, semantic_id, last_bbox)
        self._lost: List[Tuple[float, str, np.ndarray]] = []
        
    def _generate_semantic_id(self, class_name: str) -> str:
        count = self._class_counters.get(class_name, 0) + 1
        self._class_counters[class_name] = count
        return f"{class_name}_{count:02d}"

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
        
        # Clean up old lost tracks
        self._lost = [
            (f_idx, s_id, bbox) for f_idx, s_id, bbox in self._lost
            if current_frame_idx - f_idx <= self.merge_window
        ]
        
        # 2. Process current tracks
        track_to_semantic: Dict[int, str] = {}
        
        for t in tracks:
            # Skip tracks that haven't hit the minimum observations to be considered valid
            if t.hits < 2: 
                continue
                
            if t.tid in self._id_map:
                semantic_id = self._id_map[t.tid]
            else:
                # This is a new track. Check if we can Re-ID it.
                matched_semantic_id = None
                best_iou = 0.0
                best_lost_idx = -1
                
                for i, (lost_f_idx, lost_s_id, lost_bbox) in enumerate(self._lost):
                    # Check class consistency
                    if self.active_objects[lost_s_id].class_name != t.label:
                        continue
                        
                    iou = _iou(t.box, lost_bbox)
                    if iou > self.merge_iou_threshold and iou > best_iou:
                        best_iou = iou
                        matched_semantic_id = lost_s_id
                        best_lost_idx = i
                
                if matched_semantic_id:
                    # Re-ID successful
                    semantic_id = matched_semantic_id
                    self._lost.pop(best_lost_idx)
                else:
                    # Genuinely new track
                    semantic_id = self._generate_semantic_id(t.label)
                    self.active_objects[semantic_id] = TrackedObject(
                        object_id=semantic_id,
                        class_name=t.label,
                        first_seen=current_time,
                        last_seen=current_time
                    )
                
                self._id_map[t.tid] = semantic_id
            
            # Update object trajectory and last seen
            obj = self.active_objects[semantic_id]
            obj.last_seen = current_time
            obj.trajectory.append((current_time, t.box))
            
            track_to_semantic[t.tid] = semantic_id
            
        return track_to_semantic
