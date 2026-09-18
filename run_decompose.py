from PIL import Image
from relsgg import RelateAnything

ra = RelateAnything.from_pretrained("maelic/relsgg-vits16", device="cuda")

img = Image.open("assets/reel/images/catlaptop.jpg").convert("RGB")
W, H = img.size
boxes = [[0.30*W, 0.05*H, 0.85*W, 0.55*H],
         [0.10*W, 0.40*H, 0.95*W, 0.95*H]]

g = ra.predict(img, boxes, topk=10, max_boxes=16, decompose=True)
print("\nSPATIAL (layout):")
for t in g["spatial"]:
    print("  ", t)
print("\nSEMANTIC (interaction):")
for t in g["semantic"]:
    print("  ", t)
