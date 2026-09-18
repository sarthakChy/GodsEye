import cv2
import numpy as np
from typing import Dict, Any, List

def _color(i: int):
    PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (138, 99, 210),
               (214, 168, 42), (52, 187, 235), (200, 62, 120), (120, 160, 60)]
    return PALETTE[i % len(PALETTE)]

def render_overlay(frame: np.ndarray, boxes: np.ndarray, 
                   det2semantic: Dict[int, str], active_relations: List[Dict[str, Any]]) -> np.ndarray:
    """Render bounding boxes, labels, and active relation edges on a frame."""
    img = frame.copy()
    
    centers = {}
    
    # Draw boxes
    for i, bb in enumerate(boxes):
        if i not in det2semantic:
            continue
            
        label = det2semantic[i]
        # We can extract a stable color index by hashing the string
        color_idx = hash(label) % 8
        c = _color(color_idx)
        
        x1, y1, x2, y2 = [int(v) for v in bb]
        centers[label] = ((x1 + x2) // 2, (y1 + y2) // 2)
        
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        
        txt = label
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), c, -1)
        cv2.putText(img, txt, (x1 + 2, max(10, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                    
    # Draw relations (lines between centers)
    for rel in active_relations:
        sub = rel["subject"]
        obj = rel["object"]
        pred = rel["predicate"]
        
        if sub in centers and obj in centers:
            pt1 = centers[sub]
            pt2 = centers[obj]
            
            # Draw line
            cv2.line(img, pt1, pt2, (255, 255, 255), 2)
            
            # Draw predicate text in the middle
            mx = (pt1[0] + pt2[0]) // 2
            my = (pt1[1] + pt2[1]) // 2
            
            (tw, th), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (mx - tw//2, my - th - 4), (mx + tw//2, my + 4), (0,0,0), -1)
            cv2.putText(img, pred, (mx - tw//2, my), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            
    return img
