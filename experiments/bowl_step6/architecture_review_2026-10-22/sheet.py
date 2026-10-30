"""Side-by-side of the exact images P0 received: A base s0 vs B live v2, at the first five requests."""
import sys, h5py, numpy as np
from PIL import Image, ImageDraw
a = h5py.File(sys.argv[1], 'r'); b = h5py.File(sys.argv[2], 'r'); out = sys.argv[3]
rows = [0, 8, 16, 24, 32, 40]
tiles = []
for r in rows:
    aw, ae = a['video.wrist_image_left'][r], a['video.exterior_image_1_left'][r]
    bw, be = b[f'{r}/video.wrist_image_left'][0, 0], b[f'{r}/video.exterior_image_1_left'][0, 0]
    tiles.append(np.concatenate([ae, aw, be, bw], axis=1))
img = Image.fromarray(np.concatenate(tiles, axis=0)); dr = ImageDraw.Draw(img)
for i, r in enumerate(rows):
    for j, lab in enumerate(['A ext', 'A wrist', 'B ext', 'B wrist']):
        dr.text((j * 320 + 4, i * 180 + 4), f'{lab} req@row {r}', fill=(255, 255, 0))
img.save(out); print(img.size)
