
from __future__ import annotations
import numpy as np
from PIL import Image
from .utils import hex_to_rgb

# ===== Color space helpers (OKLab) =====
def _srgb_to_linear(c):
    c = c/255.0
    return c/12.92 if c <= 0.04045 else ((c+0.055)/1.055)**2.4

def _linear_to_srgb(c):
    return 12.92*c if c <= 0.0031308 else 1.055*(c**(1/2.4)) - 0.055

def rgb_to_oklab(rgb):
    # rgb: (...,3) uint8 or float [0..255]
    r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
    rl = np.vectorize(_srgb_to_linear)(r.astype(np.float32))
    gl = np.vectorize(_srgb_to_linear)(g.astype(np.float32))
    bl = np.vectorize(_srgb_to_linear)(b.astype(np.float32))
    l = 0.4122214708*rl + 0.5363325363*gl + 0.0514459929*bl
    m = 0.2119034982*rl + 0.6806995451*gl + 0.1073969566*bl
    s = 0.0883024619*rl + 0.2817188376*gl + 0.6299787005*bl
    l_, m_, s_ = np.cbrt(l), np.cbrt(m), np.cbrt(s)
    L = 0.2104542553*l_ + 0.7936177850*m_ - 0.0040720468*s_
    a = 1.9779984951*l_ - 2.4285922050*m_ + 0.4505937099*s_
    b = 0.0259040371*l_ + 0.7827717662*m_ - 0.8086757660*s_
    return np.stack([L,a,b], axis=-1).astype(np.float32)

def oklab_to_rgb(Lab):
    L,a,b = Lab[...,0], Lab[...,1], Lab[...,2]
    l_ = L + 0.3963377774*a + 0.2158037573*b
    m_ = L - 0.1055613458*a - 0.0638541728*b
    s_ = L - 0.0894841775*a - 1.2914855480*b
    l = l_**3; m = m_**3; s = s_**3
    rl = +4.0767416621*l - 3.3077115913*m + 0.2309699292*s
    gl = -1.2684380046*l + 2.6097574011*m - 0.3413193965*s
    bl = -0.0041960863*l - 0.7034186147*m + 1.7076147010*s
    r = np.vectorize(_linear_to_srgb)(rl); g = np.vectorize(_linear_to_srgb)(gl); b = np.vectorize(_linear_to_srgb)(bl)
    arr = np.clip(np.stack([r,g,b], axis=-1)*255.0, 0, 255).astype(np.uint8)
    return arr

def _kmeans_oklab(samples_lab, k, iters=20):
    n = len(samples_lab)
    if n == 0 or k <= 0:
        return np.zeros((0,3), dtype=np.float32)
    # init by sampling
    centers = samples_lab[np.random.choice(n, size=min(k,n), replace=False)].astype(np.float32)
    for _ in range(iters):
        d = ((samples_lab[:,None,:]-centers[None,:,:])**2).sum(axis=2)
        lab = d.argmin(axis=1)
        for i in range(len(centers)):
            pts = samples_lab[lab==i]
            if len(pts)>0:
                centers[i] = pts.mean(axis=0)
    return centers

def _nearest_oklab(rgb_arr, pal_rgb):
    lab = rgb_to_oklab(rgb_arr.astype(np.uint8))
    pal_lab = rgb_to_oklab(pal_rgb.astype(np.uint8))
    flat = lab.reshape(-1,3)
    d = ((flat[:,None,:]-pal_lab[None,:,:])**2).sum(axis=2)
    idx = d.argmin(axis=1)
    q = pal_rgb[idx]
    return idx.reshape(lab.shape[:2]), q.reshape(rgb_arr.shape)

# ===== Classic palette helpers =====
def build_palette_rgb(hex_list):
    return np.array([hex_to_rgb(h) for h in hex_list], dtype=np.float32)

def nearest_palette_color(rgb, palette):
    flat = rgb.reshape(-1,3)
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
        serp = (y % 2) == 1
        xs = range(w-1, -1, -1) if serp else range(w)
        for x in xs:
            old = out[y,x]
            _, q = nearest_palette_color(old[None,None,:], pal)
            new = q[0,0]
            out[y,x] = new
            err = old - new
            if serp:
                if x-1 >= 0: out[y, x-1] += err * 7/16
                if y+1 < h and x+1 < w: out[y+1, x+1] += err * 3/16
                if y+1 < h: out[y+1, x] += err * 5/16
                if y+1 < h and x-1 >= 0: out[y+1, x-1] += err * 1/16
            else:
                if x+1 < w: out[y, x+1] += err * 7/16
                if y+1 < h and x-1 >= 0: out[y+1, x-1] += err * 3/16
                if y+1 < h: out[y+1, x] += err * 5/16
                if y+1 < h and x+1 < w: out[y+1, x+1] += err * 1/16
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))

# ===== Adaptive (hybrid) quantizer =====
def adaptive_quantize(im: Image.Image, colors=256, dither=True, anchors=None):
    """
    Hybrid adaptive quantizer:
      - If anchors is provided (list of RGB tuples), they are seeded into the palette.
      - Remaining colors are learned via OKLab k-means from the input image.
      - Dithering uses serpentine Floyd–Steinberg in RGB with OKLab nearest-color search.
    """
    base = im.convert("RGB")
    arr = np.asarray(base)
    H,W,_ = arr.shape

    # Fallback: if colors <= 32 and no anchors, Pillow MEDIANCUT is fine
    if (anchors is None or len(anchors)==0) and colors <= 32:
        dmode = Image.FLOYDSTEINBERG if dither else Image.NONE
        return base.quantize(colors=colors, method=Image.MEDIANCUT, dither=dmode).convert("RGB")

    # Build palette
    samp = arr.reshape(-1,3)
    if len(samp) > 50000:
        idx = np.random.choice(len(samp), 50000, replace=False)
        samp = samp[idx]

    labs = rgb_to_oklab(samp.astype(np.uint8))
    anchor_rgb = np.array(anchors, dtype=np.uint8) if anchors else np.zeros((0,3), dtype=np.uint8)
    k = max(0, colors - len(anchor_rgb))
    # to avoid huge clustering cost, cap k
    k = min(k, 200)
    km = _kmeans_oklab(labs, k=k if k>0 else 1)

    pal_rgb = np.vstack([anchor_rgb, oklab_to_rgb(km)]).astype(np.uint8)
    if len(pal_rgb) > colors:
        pal_rgb = pal_rgb[:colors]

    # Quantize
    out = arr.astype(np.float32).copy()
    if not dither:
        _, q = _nearest_oklab(out.astype(np.uint8), pal_rgb.astype(np.uint8))
        return Image.fromarray(q.astype(np.uint8))

    for y in range(H):
        serp = (y % 2) == 1
        xs = range(W-1, -1, -1) if serp else range(W)
        for x in xs:
            old = np.clip(out[y,x],0,255).astype(np.uint8)[None,None,:]
            _, q = _nearest_oklab(old, pal_rgb.astype(np.uint8))
            new = q[0,0].astype(np.float32)
            out[y,x] = new
            err = old.astype(np.float32)[0,0] - new
            if serp:
                if x-1 >= 0: out[y, x-1] += err * 7/16
                if y+1 < H and x+1 < W: out[y+1, x+1] += err * 3/16
                if y+1 < H: out[y+1, x] += err * 5/16
                if y+1 < H and x-1 >= 0: out[y+1, x-1] += err * 1/16
            else:
                if x+1 < W: out[y, x+1] += err * 7/16
                if y+1 < H and x-1 >= 0: out[y+1, x-1] += err * 3/16
                if y+1 < H: out[y+1, x] += err * 5/16
                if y+1 < H and x+1 < W: out[y+1, x+1] += err * 1/16
    return Image.fromarray(np.clip(out,0,255).astype(np.uint8))

# ===== Main pipeline =====
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
    if block > 1:
        w, h = im.size
        small = im.resize((max(1, w // block), max(1, h // block)), Image.Resampling.NEAREST)
        # upscale back *without interpolation*
        im = small.resize((w, h), Image.Resampling.NEAREST)
    # quantize to palette or adaptive
    if not palette_hex:
        anchors = _load_general_anchors()
        im = adaptive_quantize(im, colors=adaptive_colors or 256, dither=dither, anchors=anchors)
    else:
        im = floyd_steinberg_quantize(im, palette_hex, dither=dither)
    return im
