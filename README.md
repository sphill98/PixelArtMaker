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
