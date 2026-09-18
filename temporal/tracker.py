from typing import List, Dict, Tuple, Any
import numpy as np

from deploy.render_video import Tracker, Track
from temporal.config import TrackingConfig
from temporal.registry import Registry, TrackedObject

class VideoTracker:
    """Adapts render_video.py's Tracker for batch video processing.
    
    Integrates with the Registry for semantic ID assignment and Re-ID across
    track breaks.
    """
    def __init__(self, config: TrackingConfig):
        self.tracker = Tracker(
            iou_thr=config.iou_threshold,
            low_iou_thr=0.5, # Default from render_video
            merge_iou=0.75,
            max_age=config.track_buffer,
            min_hits=2,
            high_conf=config.confidence_threshold,
        )
        self.registry = Registry(config)
        self.config = config

    def process_frame(self, frame_idx: int, timestamp: float, 
                      boxes: np.ndarray, labels: List[str], confs: np.ndarray, 
                      frame_width: int, frame_height: int) -> Tuple[Dict[int, str], List[Track]]:
        """Process detections for a single frame.
        
        Args:
            frame_idx: current frame index
            timestamp: current frame timestamp in seconds
            boxes: bounding boxes of shape (N, 4) in xyxy format
            labels: List of length N with class names
            confs: Array of length N with confidence scores
            frame_width: frame width
            frame_height: frame height
            
        Returns:
            Tuple containing:
            - Dict mapping original detection index to semantic ID (e.g. {0: 'person_01'})
            - List of active Track objects for the current frame
        """
        # We pass fade=0.0 because alpha fading is for rendering, and we are doing offline tracking
        if len(boxes) > 0:
            det2track = self.tracker.observe(boxes, labels, confs)
        else:
            det2track = {}
            
        self.tracker.step(frame_width, frame_height, fade=1.0)
        
        # Get active tracks and run through registry for Re-ID and semantic ID assignment
        active_tracks = [t for t in self.tracker.tracks if t.hits >= 2 and t.misses == 0]
        track2semantic = self.registry.update(timestamp, frame_idx, active_tracks)
        
        # Create det2semantic mapping (detection index -> semantic ID)
        det2semantic = {}
        for det_idx, tid in det2track.items():
            if tid in track2semantic:
                det2semantic[det_idx] = track2semantic[tid]
                
        return det2semantic, active_tracks

    def get_active_objects(self) -> Dict[str, TrackedObject]:
        """Returns the dictionary of all tracked objects across the video."""
        return self.registry.active_objects
