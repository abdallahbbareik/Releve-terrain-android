"""Génère les icônes Android (classiques, rondes, adaptatives) à partir de icon.png."""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

SRC, RES = 'icon.png', 'android/app/src/main/res'
DENS = {'mdpi': (48, 108), 'hdpi': (72, 162), 'xhdpi': (96, 216), 'xxhdpi': (144, 324), 'xxxhdpi': (192, 432)}

a = np.asarray(Image.open(SRC).convert('RGB'))

# 1. Détecter le fond blanc extérieur (les coins de l'image), sans toucher aux blancs intérieurs
white = (a > 236).all(axis=2)
white = ndi.binary_opening(white, structure=np.ones((15, 15)))
lab, _ = ndi.label(white)
border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
outside = ndi.binary_dilation(np.isin(lab, list(border)), iterations=5)

# 2. Recadrer sur la tuile et remplir les coins avec la couleur voisine
ys, xs = np.where(~outside)
y0, y1, x0, x1 = ys.min() + 2, ys.max() - 2, xs.min() + 2, xs.max() - 2
idx = ndi.distance_transform_edt(outside, return_distances=False, return_indices=True)
filled = a[tuple(idx)]
side = min(y1 - y0, x1 - x0)
cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
box = (slice(cy - side // 2, cy + side // 2), slice(cx - side // 2, cx + side // 2))
art = Image.fromarray(filled[box])
alpha = ndi.gaussian_filter((~outside[box]).astype(float), 1.5)
tile = Image.fromarray(np.dstack([filled[box], (alpha * 255).astype(np.uint8)]), 'RGBA')

def padded(img, canvas, frac):
    """Image réduite au centre, bords prolongés jusqu'à remplir le canevas."""
    n = int(canvas * frac)
    small = np.asarray(img.convert('RGB').resize((n, n), Image.LANCZOS))
    p = canvas - n
    return Image.fromarray(np.pad(small, ((p // 2, p - p // 2), (p // 2, p - p // 2), (0, 0)), mode='edge'))

def circle(img):
    s = img.size[0]
    m = Image.new('L', (s * 4, s * 4), 0)
    ImageDraw.Draw(m).ellipse((0, 0, s * 4 - 1, s * 4 - 1), fill=255)
    out = img.convert('RGBA'); out.putalpha(m.resize((s, s), Image.LANCZOS)); return out

for d, (leg, fg) in DENS.items():
    base = f'{RES}/mipmap-{d}/'
    tile.resize((leg, leg), Image.LANCZOS).save(base + 'ic_launcher.png')
    circle(padded(art, leg, 0.82)).save(base + 'ic_launcher_round.png')
    padded(art, fg, 0.66).save(base + 'ic_launcher_foreground.png')
print('Icônes générées')
