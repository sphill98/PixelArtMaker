from __future__ import annotations
from PIL import Image, ImageEnhance
import numpy as np
import imageio.v2 as imageio
import math

def make_subtle_gif(base_im: Image.Image, frames=12, duration_ms=80, mode="wobble"):
    W,H = base_im.size
    imgs = []
    for i in range(frames):
        t = i / frames
        if mode == "wobble":
            dx = int(round(1.5*math.sin(2*math.pi*t)))
            dy = int(round(1.5*math.cos(2*math.pi*t)))
            frame = Image.new("RGB", (W,H))
            frame.paste(base_im, (dx,dy))
            frame = frame.crop((max(0,dx), max(0,dy), max(0,dx)+W, max(0,dy)+H))
        elif mode == "pan":
            dx = int(round(2*math.sin(2*math.pi*t)))
            frame = Image.new("RGB", (W,H))
            frame.paste(base_im, (dx,0))
            frame = frame.crop((max(0,dx), 0, max(0,dx)+W, H))
        else:  # shimmer: slight brightness oscillation
            factor = 1.0 + 0.05*math.sin(2*math.pi*t)
            frame = ImageEnhance.Brightness(base_im).enhance(factor)
        imgs.append(frame)
    # Save via imageio requires numpy arrays
    return imgs, duration_ms

def save_gif(frames, duration_ms, out_path):
    arrs = [np.array(f) for f in frames]
    imageio.mimsave(out_path, arrs, duration=duration_ms/1000.0, loop=0)
