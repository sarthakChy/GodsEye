from PIL import Image
from ultralytics import YOLO
from relsgg import RelateAnything

det = YOLO("yolov8s.pt")                              # ~22 MB, downloads once
ra  = RelateAnything.from_pretrained("maelic/relsgg-vits16", device="cuda")

img = Image.open("assets/reel/images/catlaptop.jpg").convert("RGB")
r = det(img, verbose=False)[0]

boxes  = r.boxes.xyxy.cpu().numpy()
labels = [det.names[int(c)] for c in r.boxes.cls]
scores = r.boxes.conf.cpu().numpy()
print(f"{len(boxes)} boxes:", labels)

for t in ra.predict(img, boxes, box_labels=labels,
                    box_scores=scores, topk=15, max_boxes=16):
    print(t)
