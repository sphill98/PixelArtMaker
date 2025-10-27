# PixelArt Maker (GUI)

Batch-convert photos into **pixel-art styled** images using curated Ghibli/landscape palettes.
- Resize longest side to 512 **without cropping** (as requested).
- Palette quantization with optional Floyd–Steinberg dithering.
- Adjustable "block size" pixelation (downscale/nearest upscale).
- Save & load **custom presets** (JSON).
- Make subtle **animated GIF** loops (pan / wobble / shimmer).

## Quick start
```bash
python app.py
```
(Requires Python 3.10+; install packages from `requirements.txt`.)

## Presets
- Saved to `~/.pixelart_maker/presets.json` by default.
- Each preset stores: palette name, block size, dither on/off, saturation, contrast, gamma, and GIF options.

## v4 Update
- Adds `assets/palette_general_v4.json` built from your Mediterranean reference set.
- `General_Adaptive_256` now uses these anchors + OKLab k-means.
- New preset: **Mediterranean** (block=2, dither=ON, saturation 1.05, contrast 1.08, gamma 0.96).
