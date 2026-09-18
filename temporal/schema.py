from dataclasses import dataclass
from typing import List, Optional

@dataclass
class FrameRelation:
    subject_id: str          # "person_01"
    predicate: str           # "holding"
    object_id: str           # "laptop_01"
    logit: float             # Raw log-odds (for belief filter)
    score: float             # Calibrated probability (for display)
    timestamp: float         # seconds
    frame_idx: int

@dataclass
class TemporalRelation:
    subject_id: str
    predicate: str
    object_id: str
    start_time: float
    end_time: float
    mean_score: float
    frame_count: int

@dataclass
class Event:
    event_id: int
    event_type: str          # "PICK_UP"
    subject_id: str
    object_id: Optional[str]
    location_id: Optional[str]
    start_time: float
    end_time: float
    confidence: float
    source_relations: List[TemporalRelation]
