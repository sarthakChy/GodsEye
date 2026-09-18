import torch
import numpy as np
from typing import List, Dict, Sequence

from relsgg import RelateAnything
from temporal.config import RelationConfig
from temporal.schema import FrameRelation
from temporal.vocabulary import VALIDATED_VOCABULARY

class RelationExtractor:
    """Wraps RelateAnything for per-frame relation extraction.
    
    Crucially, exposes BOTH decoded triplets and raw logit matrices, 
    so the belief filter can observe negative evidence.
    """
    
    def __init__(self, config: RelationConfig):
        # We always enforce calibration=True so scores are meaningful.
        self.ra = RelateAnything.from_pretrained(
            repo_id=config.model_id,
            predicates=VALIDATED_VOCABULARY,
            device=config.device,
            calibration=True
        )
        self.topk = config.topk_relations
        self.vocab = VALIDATED_VOCABULARY

    def extract(self, image: np.ndarray, boxes: np.ndarray, 
                det2semantic: Dict[int, str], timestamp: float, frame_idx: int) -> List[FrameRelation]:
        """Extract relations for a single frame.
        
        Args:
            image: BGR numpy array
            boxes: (N, 4) xyxy boxes
            det2semantic: mapping from detection index (in boxes) to semantic ID
            timestamp: current frame timestamp in seconds
            frame_idx: current frame index
            
        Returns:
            List of FrameRelation objects with raw logits and calibrated scores.
        """
        # RelateAnything predict returns (triplets, raw_matrices) when we need it,
        # but the default API predict() returns just triplets.
        # We'll use ra.predict() and then manually access the logits,
        # or we can modify how we interact with it based on ra.model forward pass.
        
        if len(boxes) < 2:
            return [], None, None, None
            
        N = len(boxes)
        # Prepare inputs like ra.predict does
        img_t, W, H = self.ra._to_chw(image, self.ra.img_size)
        img_t = img_t.to(self.ra.device)
        b = boxes.copy()
        b[:, [0, 2]] /= max(W, 1)
        b[:, [1, 3]] /= max(H, 1)
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        boxes_t = torch.from_numpy(np.stack([cx, cy, b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], -1).astype(np.float32)).unsqueeze(0).to(self.ra.device)
        
        with torch.no_grad():
            out = self.ra.model(img_t, boxes_t, box_counts=torch.tensor([N], device=self.ra.device), targets=None)
            
        logits = out["logits"][0].float()      # [K, V]
        pair = out["pair_logits"][0].float()   # [K]
        sub_idx = out["sub_idx"][0].cpu().numpy()
        obj_idx = out["obj_idx"][0].cpu().numpy()
        valid = out["valid_mask"][0].cpu().numpy().astype(bool)
        keep = valid & (sub_idx < N) & (obj_idx < N) & (sub_idx != obj_idx)
        
        # Calculate scores
        scores = self.ra.contract.scores(logits, pair) # [K, V]
        best_s, best_p = scores.max(dim=-1)            # [K], [K]
        best_s = best_s.cpu().numpy()
        best_p = best_p.cpu().numpy()
        
        # Create FrameRelation objects for top predictions
        relations = []
        cand = [(float(best_s[k]), int(sub_idx[k]), int(obj_idx[k]), int(best_p[k]), k)
                for k in range(len(sub_idx)) if keep[k]]
        cand.sort(key=lambda x: -x[0])
        cand = cand[:self.topk]
        
        # kof maps (sub_idx, obj_idx) to k (index in logits/pair)
        kof = {(int(sub_idx[k]), int(obj_idx[k])): k for k in range(len(sub_idx)) if keep[k]}
        
        # Extract matrices as numpy arrays for belief filter
        logits_np = logits.cpu().numpy()
        pair_np = pair.cpu().numpy()
        
        for s, si, oi, p, k in cand:
            if si in det2semantic and oi in det2semantic:
                sub_id = det2semantic[si]
                obj_id = det2semantic[oi]
                pred = self.vocab[p]
                logit_val = float((logits_np[k] + pair_np[k])[p])
                relations.append(FrameRelation(
                    subject_id=sub_id,
                    predicate=pred,
                    object_id=obj_id,
                    logit=logit_val,
                    score=s,
                    timestamp=timestamp,
                    frame_idx=frame_idx
                ))
                
        return relations, logits_np, pair_np, kof
