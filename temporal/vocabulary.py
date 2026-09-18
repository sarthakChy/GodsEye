import numpy as np
from deploy.pipeline import _default_predicates
from relsgg.vocabulary import DEFAULT_PREDICATES

# The union of the pipeline defaults (22) and trained vocabulary (35)
VALIDATED_VOCABULARY = sorted(set(_default_predicates()) | set(DEFAULT_PREDICATES))

# Remove the ones that are not in the default sets and don't work reliably,
# such as "outside", "approaching", "moving away from"
_EXCLUDE = {"outside", "approaching", "moving away from", "picking up", "putting down"}
VALIDATED_VOCABULARY = [p for p in VALIDATED_VOCABULARY if p not in _EXCLUDE]

HEURISTIC_VOCABULARY = ["approaching", "moving away from"]

def compute_motion(centroid_a_old: np.ndarray, centroid_a_new: np.ndarray, 
                   centroid_b_old: np.ndarray, centroid_b_new: np.ndarray) -> str | None:
    """Distance-based heuristic for motion predicates.
    
    Checks if the distance between two objects is significantly changing.
    """
    dist_old = np.linalg.norm(centroid_a_old - centroid_b_old)
    dist_new = np.linalg.norm(centroid_a_new - centroid_b_new)
    
    # Avoid division by zero or extremely small distances
    if dist_old < 1e-3:
        return None
        
    if dist_new < dist_old * 0.7:
        return "approaching"
    if dist_new > dist_old * 1.3:
        return "moving away from"
    return None
