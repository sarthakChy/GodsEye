import pytest
import numpy as np
from temporal.registry import Registry
from temporal.config import TrackingConfig

class DummyTrack:
    def __init__(self, tid, cls_name, bbox):
        self.tid = tid
        self.label = cls_name
        self.box = bbox
        self.hits = 5
        
def test_registry_reid():
    """Test that a track breaking for a few frames maps back to the same semantic ID."""
    config = TrackingConfig(merge_window=5, merge_iou_threshold=0.5)
    registry = Registry(config)
    
    # Frame 1: track 1 appears
    t1 = DummyTrack(1, "person", np.array([0, 0, 10, 10]))
    id_map = registry.update(0.0, 0, [t1])
    
    assert 1 in id_map
    semantic_id = id_map[1]
    assert semantic_id == "person_01"
    
    # Frame 2: track 1 disappears
    id_map = registry.update(0.1, 1, [])
    assert len(id_map) == 0
    
    # Frame 4: track 2 appears (same class, high IoU to track 1's last position)
    t2 = DummyTrack(2, "person", np.array([1, 1, 11, 11]))
    id_map = registry.update(0.3, 3, [t2])
    
    assert 2 in id_map
    # Should Re-ID to track 1's semantic ID
    assert id_map[2] == "person_01"
    
    # Track 2 disappears at frame 4
    registry.update(0.4, 4, [])
    
    # Frame 10: track 3 appears (same class, but too late -> outside merge_window 5 frames)
    # The last frame it was seen was frame 3 (since we updated at frame 3). 
    # At frame 10, delta is 10 - 4 = 6 > merge_window 5.
    t3 = DummyTrack(3, "person", np.array([1, 1, 11, 11]))
    id_map = registry.update(1.0, 10, [t3])
    
    assert 3 in id_map
    # Should NOT Re-ID, should get new ID
    assert id_map[3] == "person_02"
    
def test_registry_no_reid_different_class():
    config = TrackingConfig(merge_window=5, merge_iou_threshold=0.5)
    registry = Registry(config)
    
    t1 = DummyTrack(1, "person", np.array([0, 0, 10, 10]))
    registry.update(0.0, 0, [t1])
    
    registry.update(0.1, 1, [])
    
    # Appears in exact same spot, but different class
    t2 = DummyTrack(2, "car", np.array([0, 0, 10, 10]))
    id_map = registry.update(0.2, 2, [t2])
    
    assert id_map[2] == "car_01"
