from __future__ import annotations
import numpy as np
from PIL import Image
from .utils import hex_to_rgb

def build_palette_rgb(hex_list):
    return np.array([hex_to_rgb(h) for h in hex_list], dtype=np.float32)

def nearest_palette_color(rgb, palette):
    # rgb: (H,W,3) float32
    # palette: (K,3) float32
    # returns indices and quantized rgb
    flat = rgb.reshape(-1,3)
    # Euclidean distance in RGB; for better results could use OKLab
    dists = ((flat[:,None,:]-palette[None,:,:])**2).sum(axis=2)
    idx = np.argmin(dists, axis=1).astype(np.int32)
    q = palette[idx]
    return idx.reshape(rgb.shape[:2]), q.reshape(rgb.shape)

def floyd_steinberg_quantize(im: Image.Image, palette_hex, dither=True):
    arr = np.asarray(im).astype(np.float32)
    pal = build_palette_rgb(palette_hex)
    h,w,_ = arr.shape
    out = arr.copy()
    if not dither:
        _, q = nearest_palette_color(out, pal)
        return Image.fromarray(np.clip(q,0,255).astype(np.uint8))

    for y in range(h):
        for x in range(w):
            old = out[y,x]
            _, q = nearest_palette_color(old[None,None,:], pal)
            new = q[0,0]
            out[y,x] = new
            err = old - new
            if x+1 < w: out[y, x+1] += err * 7/16
            if y+1 < h and x>0: out[y+1, x-1] += err * 3/16
            if y+1 < h: out[y+1, x] += err * 5/16
            if y+1 < h and x+1 < w: out[y+1, x+1] += err * 1/16
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))

def pixelate(im: Image.Image, block=2):
    if block <= 1:
        return im
    w,h = im.size
    small = im.resize((max(1,w//block), max(1,h//block)), Image.Resampling.NEAREST)
    return small.resize((w,h), Image.Resampling.NEAREST)


LOAD_GENERAL_ANCHORS = True

def adaptive_quantize(im: Image.Image, colors=256, dither=True):
    # Pillow's median-cut adaptive palette quantization
    dmode = Image.FLOYDSTEINBERG if dither else Image.NONE
    pal_im = im.quantize(colors=colors, method=Image.MEDIANCUT, dither=dmode)
    return pal_im.convert("RGB")

def _load_general_anchors():
    import json, os
    here = os.path.dirname(os.path.abspath(__file__))
    assets = os.path.join(os.path.dirname(here), 'assets', 'palette_general_v4.json')
    if os.path.exists(assets):
        try:
            with open(assets,'r') as f:
                data = json.load(f)
            return [tuple(map(int, c)) for c in data.get('anchors_rgb', [])]
        except Exception:
            return None
    return None

def pipeline(im: Image.Image, palette_hex, target_long=512, block=2, dither=True, saturation=1.0, contrast=1.0, gamma=1.0, adjust_fn=None, adaptive_colors=None):
    from .utils import load_image_keep_ratio, apply_adjustments
    im = im if isinstance(im, Image.Image) else Image.open(im).convert("RGB")
    # Resize (no crop) by longest side
    w,h = im.size
    scale = target_long / max(w,h)
    if scale != 1.0:
        im = im.resize((max(1,int(round(w*scale))), max(1,int(round(h*scale)))), Image.Resampling.NEAREST)
    # optional color tweaks
    im = apply_adjustments(im, saturation=saturation, contrast=contrast, gamma=gamma)
    # pixelation
    im = pixelate(im, block=block)
    # quantize to palette (or adaptive 256)
    if not palette_hex:
        anchors = _load_general_anchors()
        im = adaptive_quantize(im, colors=adaptive_colors or 256, dither=dither, anchors=anchors)
    else:
        im = floyd_steinberg_quantize(im, palette_hex, dither=dither)
    return im
