from __future__ import annotations
import os, json, threading, pathlib
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from pixelart.palettes import PALETTES, DEFAULT_PALETTE, ADAPTIVE_SPECIAL
from pixelart.processor import pipeline
from pixelart.gifmaker import make_subtle_gif, save_gif
from pixelart.presets import DEFAULT_PRESETS

from pixelart.anchors import DEFAULT_ANCHORS_PATH
import numpy as np
from PIL import Image
import glob, json, os

def build_anchors_from_folder(folder, out_path=DEFAULT_ANCHORS_PATH, k=42, max_per=80000):
    # Collect pixels from images in folder
    paths = []
    for ext in ("*.jpg","*.jpeg","*.png","*.webp","*.bmp","*.tif","*.tiff"):
        paths += glob.glob(os.path.join(folder, ext))
    if not paths:
        raise RuntimeError("No images in folder")
    from pixelart.processor import rgb_to_oklab, oklab_to_rgb, _nearest_oklab
    def load_px(p):
        im = Image.open(p).convert("RGB")
        w,h = im.size
        scale = 768 / max(w,h)
        if scale < 1: im = im.resize((int(w*scale), int(h*scale)), Image.Resampling.BILINEAR)
        arr = np.asarray(im).reshape(-1,3)
        N = min(max_per, len(arr))
        idx = np.random.choice(len(arr), N, replace=False)
        return arr[idx]
    arrs = [load_px(p) for p in paths]
    arr = np.vstack(arrs)
    lab = rgb_to_oklab(arr.astype(np.float32))
    # luminance bins
    L = lab[:,0]; bins = np.digitize(L, np.linspace(L.min(), L.max(), 8))
    cents = []
    def kmeans(samples, kk):
        import numpy as np
        n=len(samples); 
        if n==0: return np.zeros((0,3),dtype=np.float32)
        c = samples[np.random.choice(n, size=min(kk,n), replace=False)].astype(np.float32)
        for _ in range(20):
            d=((samples[:,None,:]-c[None,:,:])**2).sum(axis=2)
            lab=d.argmin(axis=1)
            for i in range(len(c)):
                pts=samples[lab==i]
                if len(pts)>0: c[i]=pts.mean(axis=0)
        return c
    for b in range(1,8):
        subset = lab[bins==b]
        if len(subset)<50: continue
        kk = 6 if b in (1,7) else 5
        cents.append(kmeans(subset, kk))
    cents = np.vstack(cents)
    rgb = oklab_to_rgb(cents)
    # add explicit anchors
    extra = np.array([[252,252,252],[12,12,12],[28,110,216],[19,118,204],[100,180,255]], dtype=np.uint8)
    rgb = np.vstack([extra, rgb])
    # dedup coarse
    uniq = {}
    for c in rgb:
        uniq[tuple((c//4).tolist())]=c
    data = {"name":"CustomRefSet","rgb":[v.tolist() for v in uniq.values()]}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path, len(data["rgb"])


PRESET_STORE = os.path.expanduser("~/.pixelart_maker/presets.json")

def load_presets():
    try:
        if os.path.exists(PRESET_STORE):
            with open(PRESET_STORE,"r",encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return DEFAULT_PRESETS.copy()

def save_presets(presets):
    os.makedirs(os.path.dirname(PRESET_STORE), exist_ok=True)
    with open(PRESET_STORE,"w",encoding="utf-8") as f:
        json.dump(presets, f, ensure_ascii=False, indent=2)

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        # UI hotfix: guarantee enough width for controls
        self.minsize(1024, 720)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.title("PixelArt Maker")
        self.geometry("820x520")
        self.resizable(True, True)

        self.in_dir = tk.StringVar()
        self.out_dir = tk.StringVar()
        self.target_long = tk.IntVar(value=512)
        self.block = tk.IntVar(value=3)
        self.dither = tk.BooleanVar(value=True)
        self.saturation = tk.DoubleVar(value=1.05)
        self.contrast = tk.DoubleVar(value=1.05)
        self.gamma = tk.DoubleVar(value=1.0)
        self.adaptive_colors = tk.IntVar(value=256)
        self.palette_name = tk.StringVar(value=DEFAULT_PALETTE)
        self.make_gif = tk.BooleanVar(value=False)
        self.gif_frames = tk.IntVar(value=12)
        self.gif_duration = tk.IntVar(value=80)
        self.gif_mode = tk.StringVar(value="wobble")

        self.presets = load_presets()

        self._build_ui()

    def _build_ui(self):
        pad = {'padx':8, 'pady':6}

        # Paths
        frm_paths = ttk.LabelFrame(self, text="Folders")
        frm_paths.pack(fill='x', **pad)
        ttk.Label(frm_paths, text="Input folder").grid(row=0, column=0, sticky='e')
        ttk.Entry(frm_paths, textvariable=self.in_dir, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm_paths, text="Browse...", command=self.pick_in).grid(row=0, column=2)
        ttk.Label(frm_paths, text="Output folder").grid(row=1, column=0, sticky='e')
        ttk.Entry(frm_paths, textvariable=self.out_dir, width=70).grid(row=1, column=1, sticky='we')
        ttk.Button(frm_paths, text="Browse...", command=self.pick_out).grid(row=1, column=2)
        frm_paths.columnconfigure(1, weight=1)

        # Settings
        # Reference anchors
        frm_ref = ttk.LabelFrame(self, text="Reference Anchors (optional)")
        frm_ref.pack(fill='x', **pad)
        self.ref_dir = tk.StringVar()
        ttk.Label(frm_ref, text="Reference folder").grid(row=0, column=0, sticky='e')
        ttk.Entry(frm_ref, textvariable=self.ref_dir, width=70).grid(row=0, column=1, sticky='we')
        ttk.Button(frm_ref, text="Browse...", command=lambda: self.ref_dir.set(filedialog.askdirectory() or self.ref_dir.get())).grid(row=0, column=2)
        ttk.Button(frm_ref, text="Build anchors", command=self.build_anchors).grid(row=0, column=3)
        frm_ref.columnconfigure(1, weight=1)

        frm_set = ttk.LabelFrame(self, text="Settings")
        frm_set.pack(fill='x', **pad)
        ttk.Label(frm_set, text="Palette").grid(row=0, column=0, sticky='w')
        cmb = ttk.Combobox(frm_set, textvariable=self.palette_name, values=sorted(PALETTES.keys()), state="readonly", width=30)
        cmb.grid(row=0, column=1, sticky='ew')
        def on_palette_change(event=None):
            name = self.palette_name.get()
            if name.startswith("General_") or name in ADAPTIVE_SPECIAL:
                getattr(self, 'spin_adapt', None) and self.spin_adapt.state(["!disabled"])
            else:
                getattr(self, 'spin_adapt', None) and self.spin_adapt.state(["disabled"])
        cmb.bind("<<ComboboxSelected>>", on_palette_change)

        ttk.Label(frm_set, text="Block size").grid(row=0, column=2, sticky='e')
        ttk.Spinbox(frm_set, from_=1, to=16, textvariable=self.block, width=6).grid(row=0, column=3)
        ttk.Checkbutton(frm_set, text="Dither (Floyd–Steinberg)", variable=self.dither).grid(row=0, column=4, sticky='w')

        ttk.Label(frm_set, text="Saturation").grid(row=1, column=0)
        ttk.Spinbox(frm_set, from_=0.1, to=2.0, increment=0.05, textvariable=self.saturation, width=6).grid(row=1, column=1)
        ttk.Label(frm_set, text="Contrast").grid(row=1, column=2, sticky='e')
        ttk.Spinbox(frm_set, from_=0.1, to=2.0, increment=0.05, textvariable=self.contrast, width=6).grid(row=1, column=3)
        ttk.Label(frm_set, text="Gamma").grid(row=1, column=4, sticky='e')
        ttk.Spinbox(frm_set, from_=0.5, to=2.0, increment=0.05, textvariable=self.gamma, width=8, justify='right').grid(row=1, column=5, sticky='ew')
        ttk.Label(frm_set, text="Adaptive").grid(row=1, column=6, sticky='e')
        self.spin_adapt = ttk.Spinbox(frm_set, from_=16, to=512, textvariable=self.adaptive_colors, width=8, justify='right')
        self.spin_adapt.grid(row=1, column=7, sticky='ew')
        # initialize adaptive spin state after creation
        try:
            on_palette_change()
        except Exception:
            pass

        # GIF options
        frm_gif = ttk.LabelFrame(self, text="GIF (optional)")
        frm_gif.pack(fill='x', **pad)
        ttk.Checkbutton(frm_gif, text="Create subtle animated GIF", variable=self.make_gif).grid(row=0, column=0, sticky='w')
        ttk.Label(frm_gif, text="Mode").grid(row=0, column=1)
        ttk.Combobox(frm_gif, textvariable=self.gif_mode, values=["wobble","pan","shimmer"], state="readonly", width=10).grid(row=0, column=2)
        ttk.Label(frm_gif, text="Frames").grid(row=0, column=3)
        ttk.Spinbox(frm_gif, from_=4, to=60, textvariable=self.gif_frames, width=6).grid(row=0, column=4)
        ttk.Label(frm_gif, text="Duration (ms)").grid(row=0, column=5)
        ttk.Spinbox(frm_gif, from_=20, to=300, textvariable=self.gif_duration, width=6).grid(row=0, column=6)

        # Presets
        frm_preset = ttk.LabelFrame(self, text="Presets")
        frm_preset.pack(fill='x', **pad)
        self.cmb_preset = ttk.Combobox(frm_preset, values=list(self.presets.keys()), state="readonly", width=30)
        self.cmb_preset.grid(row=0, column=0)
        ttk.Button(frm_preset, text="Load", command=self.load_preset).grid(row=0, column=1)
        ttk.Button(frm_preset, text="Save current as...", command=self.save_preset_as).grid(row=0, column=2)
        ttk.Button(frm_preset, text="Delete", command=self.delete_preset).grid(row=0, column=3)

        # Action
        frm_act = ttk.Frame(self)
        frm_act.pack(fill='x', **pad)
        self.progress = ttk.Progressbar(frm_act, mode='determinate')
        self.progress.pack(fill='x', side='left', expand=True, padx=8)
        ttk.Button(frm_act, text="Convert", command=self.convert).pack(side='right')

        # Log
        self.log = tk.Text(self, height=10)
        self.log.pack(fill='both', expand=True, padx=8, pady=8)
        self.log.insert('end', "Ready. Pick folders and press Convert.\n")

    # Folder pickers
    def pick_in(self):
        d = filedialog.askdirectory()
        if d: self.in_dir.set(d)

    def pick_out(self):
        d = filedialog.askdirectory()
        if d: self.out_dir.set(d)

    # Preset handlers
    def apply_preset(self, data):
        self.palette_name.set(data.get("palette", self.palette_name.get()))
        self.block.set(int(data.get("block", self.block.get())))
        self.dither.set(bool(data.get("dither", self.dither.get())))
        self.saturation.set(float(data.get("saturation", self.saturation.get())))
        self.contrast.set(float(data.get("contrast", self.contrast.get())))
        self.gamma.set(float(data.get("gamma", self.gamma.get())))

    def load_preset(self):
        name = self.cmb_preset.get()
        if not name: return
        data = self.presets.get(name)
        if data:
            self.apply_preset(data)
            self.log.insert('end', f"Loaded preset: {name}\n")

    def save_preset_as(self):
        name = simple_input(self, "Preset name?", "MyPreset")
        if not name: return
        data = {
            "palette": self.palette_name.get(),
            "block": int(self.block.get()),
            "dither": bool(self.dither.get()),
            "saturation": float(self.saturation.get()),
            "contrast": float(self.contrast.get()),
            "gamma": float(self.gamma.get())
        }
        self.presets[name] = data
        save_presets(self.presets)
        self.cmb_preset.configure(values=list(self.presets.keys()))
        self.cmb_preset.set(name)
        self.log.insert('end', f"Saved preset: {name}\n")

    def delete_preset(self):
        name = self.cmb_preset.get()
        if not name: return
        if name in self.presets:
            del self.presets[name]
            save_presets(self.presets)
            self.cmb_preset.configure(values=list(self.presets.keys()))
            self.cmb_preset.set("")
            self.log.insert('end', f"Deleted preset: {name}\n")


    def build_anchors(self):
        folder = self.ref_dir.get()
        if not folder:
            messagebox.showwarning("Missing folder", "Select a reference folder first.")
            return
        try:
            out, n = build_anchors_from_folder(folder)
            self.log.insert('end', f"Built reference anchors ({n} colors) → {out}\n")
            messagebox.showinfo("Anchors updated", f"Saved {n} anchor colors to\n{out}\nUse palette 'General_Adaptive_RefSet'.")
        except Exception as e:
            messagebox.showerror("Failed", str(e))

    # Conversion
    def convert(self):
        in_dir = pathlib.Path(self.in_dir.get())
        out_dir = pathlib.Path(self.out_dir.get())
        if not in_dir.exists():
            messagebox.showerror("Error", "Input folder doesn't exist.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in in_dir.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}]
        if not files:
            messagebox.showwarning("No images", "No image files found in the selected input folder.")
            return

        adaptive = int(self.adaptive_colors.get()) if (self.palette_name.get() in ADAPTIVE_SPECIAL or self.palette_name.get().startswith('General_')) else None
        params = dict(
            target_long=512,
            block=int(self.block.get()),
            dither=bool(self.dither.get()),
            saturation=float(self.saturation.get()),
            contrast=float(self.contrast.get()),
            gamma=float(self.gamma.get()),
            palette_hex=PALETTES[self.palette_name.get()],
            adaptive_colors=adaptive,
        )

        self.progress.configure(maximum=len(files), value=0)
        self.log.insert('end', f"Converting {len(files)} image(s) with palette '{self.palette_name.get()}'...\n")
        self.update_idletasks()

        def worker():
            for i,p in enumerate(files,1):
                try:
                    im = Image.open(p).convert("RGB")
                    out = pipeline(im, **params)
                    out_path = out_dir / (p.stem + "_px.png")
                    out.save(out_path)
                    if self.make_gif.get():
                        frames, dur = make_subtle_gif(out, frames=int(self.gif_frames.get()), duration_ms=int(self.gif_duration.get()), mode=self.gif_mode.get())
                        gif_path = out_dir / (p.stem + "_px.gif")
                        save_gif(frames, dur, gif_path)
                    self.log.insert('end', f"✓ {p.name}\n")
                except Exception as e:
                    self.log.insert('end', f"✗ {p.name}: {e}\n")
                self.progress['value'] = i
                self.update_idletasks()
            self.log.insert('end', "Done.\n")

        threading.Thread(target=worker, daemon=True).start()

def simple_input(parent, prompt, default=""):
    top = tk.Toplevel(parent)
    top.title("Input")
    ttk.Label(top, text=prompt).pack(padx=10, pady=10)
    var = tk.StringVar(value=default)
    e = ttk.Entry(top, textvariable=var, width=40)
    e.pack(padx=10, pady=4)
    e.focus_set()
    res = {'val': None}
    def ok():
        res['val'] = var.get()
        top.destroy()
    ttk.Button(top, text="OK", command=ok).pack(padx=10, pady=10)
    parent.wait_window(top)
    return res['val']

if __name__ == "__main__":
    app = App()
    app.mainloop()
