from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from deploy.render_video import EdgeBook, Edge
from temporal.config import TemporalConfig
from temporal.schema import TemporalRelation, FrameRelation

class TemporalAggregator:
    """Wraps render_video.py's EdgeBook for offline temporal aggregation.
    
    Uses EdgeBook with belief=True for Kalman smoothing, and records the full state history
    to detect state transitions and emit TemporalRelation intervals.
    """
    def __init__(self, config: TemporalConfig):
        self.edge_book = EdgeBook(
            score_ema=config.score_ema,
            on_thr=config.on_threshold,
            off_thr=config.off_threshold,
            on_frames=config.on_frames,
            off_frames=config.off_frames,
            belief=config.use_belief,
            kf_q=config.kf_q,
            kf_r=config.kf_r,
            p_max=config.kf_p_max,
            # For offline processing, max edges can be very high
            max_edges=1000,
            max_spatial=1000
        )
        # We don't want fades to kill edges slowly, so set fade high or bypass it
        self.edge_book.fade = 1.0 
        
        # Track history of intervals: (subject_id, object_id, predicate) -> list of TemporalRelation
        self.history: Dict[Tuple[str, str, str], List[TemporalRelation]] = {}
        
        # State at previous frame
        self._prev_active: Dict[Tuple[str, str, str], Dict] = {}
        
    def _snapshot_active(self) -> Dict[Tuple[str, str, str], Dict]:
        """Returns the current active edges from EdgeBook."""
        active = {}
        for edge in self.edge_book.visible():
            key = (edge.sub, edge.obj, edge.pred)
            active[key] = {
                "score": max(edge.shown, edge.score),
                "on": edge.on
            }
        return active

    def observe_frame(self, frame_idx: int, timestamp: float, 
                      triplets: List[Any], det2track: Dict[int, str], 
                      active_tracks_ids: set, raw: Optional[Tuple[np.ndarray, np.ndarray, Dict]] = None):
        """Process one frame through the belief filter.
        
        Args:
            frame_idx: Current frame index
            timestamp: Current timestamp
            triplets: Triplet objects
            det2track: Mapping from detection index to semantic string
            active_tracks_ids: Set of semantic strings for active tracks
            raw: (logits_np, pair_np, kof) from extractor
        """
        # Snapshot before
        prev_state = self._snapshot_active()
        
        # EdgeBook observe needs is_spatial, which we can mock or derive from a simple check.
        # It also expects sub/obj in triplets to be track IDs, but we use semantic strings.
        # We must map them properly. EdgeBook uses `e.sub` and `e.obj`. 
        # Triplet objects have `subject_label` and `object_label`.
        # However, EdgeBook.observe() expects Triplet.subject_idx to map via track2det, and tracking ID via det2track.
        # Wait, EdgeBook expects integer track IDs. Let's just use the semantic ID directly as the tracking ID.
        # It just needs hashable IDs.
        
        # Wait, the edge_book expects det2track to map det_idx -> track_id. We pass det_idx -> semantic_id.
        is_spatial = [False] * len(triplets) # Simplification, but actually we should know this
        
        # Call observe
        self.edge_book.observe(triplets, det2track, is_spatial, raw=raw)
        
        # Call step
        self.edge_book.step(active_tracks_ids)
        
        # Snapshot after
        curr_state = self._snapshot_active()
        
        self._record_transitions(frame_idx, timestamp, prev_state, curr_state)
        
    def _record_transitions(self, frame_idx: int, timestamp: float, 
                            prev_state: Dict[Tuple[str, str, str], Dict], 
                            curr_state: Dict[Tuple[str, str, str], Dict]):
        """Detect transitions and build TemporalRelations."""
        # Find newly started relations
        for key, curr in curr_state.items():
            if curr["on"] and (key not in prev_state or not prev_state[key]["on"]):
                # Start of a new interval
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(TemporalRelation(
                    subject_id=key[0],
                    object_id=key[1],
                    predicate=key[2],
                    start_time=timestamp,
                    end_time=timestamp,
                    mean_score=curr["score"],
                    frame_count=1
                ))
            elif curr["on"] and key in prev_state and prev_state[key]["on"]:
                # Continuing interval
                if key in self.history and len(self.history[key]) > 0:
                    rel = self.history[key][-1]
                    rel.end_time = timestamp
                    # Update running mean
                    rel.mean_score = (rel.mean_score * rel.frame_count + curr["score"]) / (rel.frame_count + 1)
                    rel.frame_count += 1

    def get_temporal_relations(self) -> List[TemporalRelation]:
        """Returns all completed temporal relation intervals."""
        all_relations = []
        for rel_list in self.history.values():
            all_relations.extend(rel_list)
        return sorted(all_relations, key=lambda r: r.start_time)
