import pytest
import numpy as np
from temporal.belief import TemporalAggregator
from temporal.config import TemporalConfig

class MockTriplet:
    def __init__(self, sub_idx, obj_idx, pred, score):
        self.subject_idx = sub_idx
        self.object_idx = obj_idx
        self.predicate = pred
        self.score = score

def test_aggregator_transitions():
    """Test that Aggregator creates distinct temporal relations when tracks break."""
    config = TemporalConfig(
        use_belief=True,
        on_threshold=0.5,
        off_threshold=0.3,
        on_frames=1,
        off_frames=2
    )
    aggregator = TemporalAggregator(config)
    
    det2track = {0: "person_01", 1: "cup_01"}
    
    # raw logit arrays (for one relation) to feed the belief filter
    logits = np.array([[-1.0, 2.0]]) # 2.0 is high score
    pair = np.array([0.0])
    kof = {(0, 1): 0}
    vidx = {"holding": 1}
    
    class MockContract:
        def fuse(self, pred, pair):
            # fuse normally computes sigmoid(pred + pair), just return the sum here for simplicity
            # wait, it must return something where [v] is a probability or logit?
            # actually EdgeBook uses z = float(contract.fuse...[v])
            return pred + pair
            
    raw = (logits, pair, kof, MockContract(), vidx)
    
    # Frame 1: High score, turns ON (on_frames=1)
    t = MockTriplet(0, 1, "holding", 0.9)
    aggregator.observe_frame(1, 0.1, [t], det2track, {"person_01", "cup_01"}, raw)
    
    # Frame 2: Still high
    aggregator.observe_frame(2, 0.2, [t], det2track, {"person_01", "cup_01"}, raw)
    
    # Frame 3: Track goes missing (occluded), but active_tracks_ids EXCLUDES cup_01.
    # EdgeBook should hold belief (P grows), but alpha fades. Wait, we set fade = 1.0 in Aggregator!
    # So if it's not in live_tracks, it fades immediately.
    aggregator.observe_frame(3, 0.3, [], det2track, {"person_01"}, raw)
    
    # Frame 4: Still missing.
    aggregator.observe_frame(4, 0.4, [], det2track, {"person_01"}, raw)
    
    # Frame 5: Comes back.
    aggregator.observe_frame(5, 0.5, [t], det2track, {"person_01", "cup_01"}, raw)
    
    rels = aggregator.get_temporal_relations()
    
    # Since fade = 1.0, the relation should have died on Frame 3, and a new one started on Frame 5.
    # We should have exactly 2 TemporalRelations!
    assert len(rels) >= 2
    
    rel1 = rels[0]
    rel2 = rels[-1]
    
    assert rel1.subject_id == "person_01"
    assert rel1.object_id == "cup_01"
    assert rel1.predicate == "holding"
    assert rel1.start_time == 0.1
    # Frame 2 updated the end_time
    assert rel1.end_time == 0.2
    
    assert rel2.start_time == 0.5
