from dataclasses import dataclass, field

@dataclass
class VideoConfig:
    sample_fps: float = 2.0         # Extract 2 frames/sec
    max_frames: int = 600           # Cap for long videos
    resize_max_dim: int = 1280      # Resize large frames

@dataclass
class TrackingConfig:
    detector_model: str = "yolov8m.pt"
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.5
    track_buffer: int = 30          # render_video.py uses max_age=12
    merge_window: int = 15          # (fallback) Frames to check for re-ID
    merge_iou_threshold: float = 0.4

    # Static/dynamic re-ID (Embodied VideoAgent §C)
    static_merge_window: int = 30       # Longer: static objects don't move
    dynamic_merge_window: int = 5       # Shorter: dynamic objects move fast
    static_motion_threshold: float = 3.0  # Avg px/frame; above -> dynamic
    dynamic_match_radius: float = 60.0    # Px; center-distance fallback

@dataclass
class RelationConfig:
    model_id: str = "maelic/relsgg-vits16"  # ViT-S/16 (46M) for 4GB VRAM
    topk_relations: int = 20
    device: str = "cpu"             # Relation model on CPU, detector on GPU

@dataclass
class TemporalConfig:
    # Inherited from render_video.py's EdgeBook defaults:
    score_ema: float = 0.35
    on_threshold: float = 0.50
    off_threshold: float = 0.38
    on_frames: int = 2
    off_frames: int = 6
    # Belief filter (render_video.py's RelationBelief):
    use_belief: bool = True         # Use Kalman belief filter (not EMA fallback)
    kf_q: float = 0.05             # Process noise
    kf_r: float = 1.0              # Measurement noise
    kf_p_max: float = 8.0          # Kill edge if unobserved too long
    min_duration_sec: float = 0.3

    # Sliding-window state refinement (GraSP-VLA §III-B)
    theta: int = 3                # Window size for state voting
    theta_k: int = 2              # Consecutive frames needed to accept a transition
    min_interval_sec: float = 0.2 # Discard intervals shorter than this

    # Confidence accumulation (GraSP-VLA Eq. 5)
    omega_sigma: float = 0.5      # Per-second confidence gain while active

@dataclass
class GodsEyeConfig:
    video: VideoConfig = field(default_factory=VideoConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    relation: RelationConfig = field(default_factory=RelationConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
