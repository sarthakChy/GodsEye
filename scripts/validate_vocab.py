import argparse
import sys
import os
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from temporal.config import RelationConfig
from temporal.extractor import RelationExtractor

def main():
    parser = argparse.ArgumentParser(description="Test validated vocabulary on a demo image.")
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--model", type=str, default="maelic/relsgg-vits16", help="HF model hub id")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    args = parser.parse_args()

    print(f"Loading model {args.model} on {args.device}...")
    config = RelationConfig(model_id=args.model, device=args.device, topk_relations=20)
    extractor = RelationExtractor(config)
    
    print(f"Vocabulary loaded with {len(extractor.vocab)} predicates.")
    print("Predicates:", ", ".join(extractor.vocab))
    
    if not os.path.exists(args.image):
        print(f"Error: image {args.image} not found.")
        return
        
    img = cv2.imread(args.image)
    if img is None:
        print(f"Error: could not read {args.image}.")
        return
        
    # Create some dummy boxes for testing
    H, W = img.shape[:2]
    cx, cy = W // 2, H // 2
    # Two simple boxes: left and right halves
    boxes = np.array([
        [0, 0, cx, H],
        [cx, 0, W, H],
        [W//4, H//4, 3*W//4, 3*H//4] # Center box
    ], dtype=np.float32)
    
    det2semantic = {0: "object_01", 1: "object_02", 2: "object_03"}
    
    print(f"Extracting relations for {len(boxes)} boxes...")
    relations, logits, pair, kof = extractor.extract(img, boxes, det2semantic, 0.0, 0)
    
    print(f"\nExtracted {len(relations)} top relations:")
    for r in relations:
        print(f"  {r.subject_id} {r.predicate} {r.object_id} (score: {r.score:.4f}, logit: {r.logit:.4f})")

if __name__ == "__main__":
    main()
