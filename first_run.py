from PIL import Image
from relsgg import RelateAnything

model = RelateAnything.from_pretrained("maelic/relsgg-vits16", device="cuda")

img = Image.open("assets/reel/images/catlaptop.jpg").convert("RGB")
W, H = img.size
print(f"image {W}x{H}")

# [x1, y1, x2, y2] in pixels — eyeball against the image and nudge.
boxes = [
    [0.30*W, 0.05*H, 0.85*W, 0.55*H],   # cat
    [0.10*W, 0.40*H, 0.95*W, 0.95*H],   # laptop
]

for t in model.predict(img, boxes, topk=10, max_boxes=16):
    print(t)
