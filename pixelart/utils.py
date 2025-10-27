from __future__ import annotations
from PIL import Image, ImageEnhance
import numpy as np
import math

def load_image_keep_ratio(path: str, target_long=512):
    im = Image.open(path).convert("RGB")
    w,h = im.size
    scale = target_long / max(w,h)
    if scale != 1.0:
        im = im.resize((max(1,int(round(w*scale))), max(1,int(round(h*scale)))), Image.Resampling.NEAREST)
    return im

def apply_adjustments(im: Image.Image, saturation=1.0, contrast=1.0, gamma=1.0):
    if saturation != 1.0:
        im = ImageEnhance.Color(im).enhance(saturation)
    if contrast != 1.0:
        im = ImageEnhance.Contrast(im).enhance(contrast)
    if gamma != 1.0:
        arr = np.asarray(im).astype(np.float32)/255.0
        arr = np.clip(arr, 1e-6, 1.0) ** (1.0/gamma)
        im = Image.fromarray(np.clip(arr*255,0,255).astype(np.uint8))
    return im

def to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)

def hex_to_rgb(hexstr):
    hexstr = hexstr.lstrip("#")
    return tuple(int(hexstr[i:i+2], 16) for i in (0,2,4))
