from typing import List, Dict, Tuple, Any, Optional
from collections import deque
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

        # Sliding-window state refinement (GraSP-VLA §III-B)
        self.theta = config.theta
        self.theta_k = config.theta_k
        self.min_interval_sec = config.min_interval_sec
        self.omega_sigma = config.omega_sigma

        # Per-key: history of raw on/off booleans, refined state, last ts
        self._obs_hist: Dict[Tuple[str, str, str], deque] = {}
        self._refined: Dict[Tuple[str, str, str], bool] = {}
        self._last_ts: Dict[Tuple[str, str, str], float] = {}

        # Per-key: currently open interval (if refined_on is True)
        self._open: Dict[Tuple[str, str, str], TemporalRelation] = {}
        
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
        is_spatial = {} # EdgeBook looks up is_spatial.get(t.predicate, False)
        
        # Call observe
        self.edge_book.observe(triplets, det2track, is_spatial, raw=raw)
        
        # Call step
        self.edge_book.step(active_tracks_ids)
        
        # Snapshot after
        curr_state = self._snapshot_active()
        
        self._record_transitions(frame_idx, timestamp, prev_state, curr_state)
        
    def _refine_state(self, hist: deque, key: Tuple[str, str, str]) -> bool:
        """Sliding-window state refinement.

        Design:
          * OPEN is eager: a single raw ON opens the interval, so short-lived
            relations are not lost. The window is not required to be full.
          * CLOSE requires theta_k consecutive raw OFF observations, so a
            single missed frame (occlusion) does not fork the interval.
          * In between, we hold the previous refined state.

        This mirrors GraSP-VLA's intent (reject single-frame noise) without
        penalising low-fps pipelines where a relation may only appear once.
        """
        # Eager open: trust the latest raw observation when the window is
        # not yet full OR when it just became True.
        if hist and hist[-1]:
            return True
        # Window not full and latest is False -> trust latest
        if len(hist) < self.theta_k:
            return hist[-1] if hist else False
        # Window full and latest is False -> require the whole window off
        tail = list(hist)[-self.theta_k:]
        if not any(tail):
            return False
        return self._refined.get(key, False)

    def _record_transitions(self, frame_idx, timestamp, prev_state, curr_state):
        """Detect refined transitions and build TemporalRelation intervals.

        Runs on top of EdgeBook. Every key ever seen passes through here each
        frame (even if absent from curr_state), so we can hold the state open
        while occlusion is in progress.
        """
        all_keys = set(prev_state) | set(curr_state) | set(self._refined) | set(self._open)

        for key in all_keys:
            curr = curr_state.get(key)
            raw_on = bool(curr and curr.get("on"))
            raw_score = curr["score"] if curr else 0.0

            hist = self._obs_hist.setdefault(key, deque(maxlen=self.theta))
            hist.append(raw_on)

            refined_on = self._refine_state(hist, key)
            was_on = self._refined.get(key, False)
            self._refined[key] = refined_on

            # --- OFF -> ON: start a new interval ---
            if refined_on and not was_on:
                self._open[key] = TemporalRelation(
                    subject_id=key[0],
                    predicate=key[2],
                    object_id=key[1],
                    start_time=timestamp,
                    end_time=timestamp,
                    mean_score=raw_score,
                    frame_count=1,
                    confidence=raw_score,
                )
                self._last_ts[key] = timestamp

            # --- ON -> ON: extend, accumulate confidence ---
            elif refined_on and was_on:
                if key in self._open:
                    rel = self._open[key]
                    rel.end_time = timestamp
                    if curr:
                        n = rel.frame_count
                        rel.mean_score = (rel.mean_score * n + raw_score) / (n + 1)
                        rel.frame_count = n + 1
                    # GraSP-VLA Eq. 5: omega_r += sigma * (tau_c - tau_r)
                    dt = timestamp - self._last_ts.get(key, timestamp)
                    rel.confidence += self.omega_sigma * max(0.0, dt)
                    self._last_ts[key] = timestamp

            # --- ON -> OFF: close the interval ---
            elif not refined_on and was_on:
                if key in self._open:
                    rel = self._open.pop(key)
                    duration = rel.end_time - rel.start_time
                    if duration >= self.min_interval_sec:
                        self.history.setdefault(key, []).append(rel)
                    self._last_ts.pop(key, None)

    def get_temporal_relations(self) -> List[TemporalRelation]:
        """Returns all temporal relation intervals.

        Includes intervals that are still open at end-of-processing, so a
        relation that is active on the last observed frame is reported.
        """
        all_relations = []
        for rel_list in self.history.values():
            all_relations.extend(rel_list)
        for rel in self._open.values():
            duration = rel.end_time - rel.start_time
            if duration >= self.min_interval_sec:
                all_relations.append(rel)
        return sorted(all_relations, key=lambda r: r.start_time)
