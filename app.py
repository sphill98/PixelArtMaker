from __future__ import annotations
import os, json, threading, pathlib
from tkinter import ttk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
from pixelart.palettes import PALETTES, DEFAULT_PALETTE
from pixelart.processor import pipeline
from pixelart.gifmaker import make_subtle_gif, save_gif
from pixelart.presets import DEFAULT_PRESETS

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
        frm_set = ttk.LabelFrame(self, text="Settings")
        frm_set.pack(fill='x', **pad)
        ttk.Label(frm_set, text="Palette").grid(row=0, column=0, sticky='w')
        cmb = ttk.Combobox(frm_set, textvariable=self.palette_name, values=sorted(PALETTES.keys()), state="readonly", width=30)
        cmb.grid(row=0, column=1, sticky='w')
        ttk.Label(frm_set, text="Block size").grid(row=0, column=2, sticky='e')
        ttk.Spinbox(frm_set, from_=1, to=16, textvariable=self.block, width=6).grid(row=0, column=3)
        ttk.Checkbutton(frm_set, text="Dither (Floyd–Steinberg)", variable=self.dither).grid(row=0, column=4, sticky='w')

        ttk.Label(frm_set, text="Saturation").grid(row=1, column=0)
        ttk.Spinbox(frm_set, from_=0.1, to=2.0, increment=0.05, textvariable=self.saturation, width=6).grid(row=1, column=1)
        ttk.Label(frm_set, text="Contrast").grid(row=1, column=2, sticky='e')
        ttk.Spinbox(frm_set, from_=0.1, to=2.0, increment=0.05, textvariable=self.contrast, width=6).grid(row=1, column=3)
        ttk.Label(frm_set, text="Gamma").grid(row=1, column=4, sticky='e')
        ttk.Spinbox(frm_set, from_=0.5, to=2.0, increment=0.05, textvariable=self.gamma, width=6).grid(row=1, column=5)

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

        params = dict(
            target_long=512,
            block=int(self.block.get()),
            dither=bool(self.dither.get()),
            saturation=float(self.saturation.get()),
            contrast=float(self.contrast.get()),
            gamma=float(self.gamma.get()),
            palette_hex=PALETTES[self.palette_name.get()],
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
