# CN Hints Bridge (Forge / Forge Neo)
Generate **Canny** + **OpenPose** hint maps (batch or single), preview them, optionally clean/edit Canny, then **send directly into ControlNet** (txt2img + img2img) inside **Forge / Forge Neo**.

> Built to keep the hint generation dependencies **outside** your main Forge install (separate venv), so your SD setup stays clean.

---

## ✨ Features
### Hint generation
- **Batch** generation (scan a folder recursively)
- **Single-image** generation (solo run)
- Output preserves **relative paths** (source → map pairing just works)

### Pre-processing
- **Overlay before detection**
  - **Transparent PNG overlay** (position/scale/opacity)
  - **Text overlay** (size/color/outline)
- **Canny quality toggles**
  - invert
  - edge thin/thick
  - adaptive threshold/autocontrast
  - clean background (pure B/W)
  - speckle filter (median)

### Post-processing (after Canny)
- **Canny cleanup editor**: paint a mask → **erase edges to black** (no edges) → save as new file or overwrite

### Forge UI workflow
- Browse maps with **preview**
- **Send Canny** to ControlNet (txt2img / img2img)
- **Send BOTH**: Canny → unit A, OpenPose → unit B (user-selectable)
- **Send LATEST pair** one-click (no dropdown needed)
- **Auto-refresh** browser after generation
- **Mini gallery** (latest 8 maps)
- **Unit count auto-detect** (matches your ControlNet slots)
- **Copy path** + **copy relative key**
- **Open output folder** button

### Convenience
- Settings are persisted in `cn_hints_settings.json` (python path, folders, thresholds, toggles, etc.)

---

## ✅ Requirements
- **Forge / Forge Neo** (Stable Diffusion WebUI Forge variant)
- **Python 3.10+** (recommended: 3.10 for widest compatibility)
- OS: Windows works best (Linux also OK if you adjust path opens)

---

## 📦 Installation (Forge Extension)
1. Copy this extension folder into:
   - `sd-webui-forge-neo/extensions/cn_hints_bridge/`
2. Restart Forge / Forge Neo

> If you downloaded a zip release, unzip it so the structure becomes:

```
extensions/
  cn_hints_bridge/
    scripts/
    tools/
    javascript/
```

---

## 🧪 Recommended: Separate Python Environment (venv)
This extension can run hint generation in a **separate** venv so it doesn’t mess with Forge’s own Python packages.

### Option A (Recommended): Create venv **outside** Forge
Example:
- `D:\AI\cn_hints_env\`
- `F:\MyTools\cn_hints\cn_hints_env\`

**Windows (CMD / PowerShell):**
```bat
cd /d F:\MyTools\cn_hints
py -3.10 -m venv cn_hints_env
cn_hints_env\Scripts\activate
python -m pip install -U pip wheel setuptools
```

Install dependencies:
```bat
pip install numpy pillow opencv-python tqdm
pip install controlnet-aux
pip install torch torchvision
```

> ⚠️ If `torch` install fails or you want CUDA builds, install PyTorch from the official PyTorch “Get Started” page for your CUDA version.  
> (CPU-only is fine too; it’ll just be slower for pose.)

Optional (silence warning):
```bat
pip install mediapipe
```

Then in the extension UI, set:
- **Python Path** → `F:\MyTools\cn_hints\cn_hints_env\Scripts\python.exe`

---

### Option B: Create venv **inside** the extension folder (not recommended for GitHub)
You *can* do:
- `extensions/cn_hints_bridge/tools/cn_hints_env/`

But **do not commit** the venv into GitHub (huge + messy). If you do this locally, add it to `.gitignore`.

---

## ⚙️ First-time Setup in the UI
Open Forge → go to the **CN Hints Bridge** tab:

### 1) Python Path
Set the python executable to your separate env, e.g.:
- `...\cn_hints_env\Scripts\python.exe`

### 2) Input / Output folders
- **Input dir (in_dir)**: folder containing images to scan
- **Output dir (out_dir)**: where maps will be written

Output structure example:
```
out_dir/
  canny/...
  openpose/...
```
Relative paths are preserved, e.g.:
```
in_dir/sets/a/001.png
out_dir/canny/sets/a/001.png
out_dir/openpose/sets/a/001.png
```

### 3) Save settings
Enable **Auto-save settings** (recommended) or click **Save settings** manually.

---

## 🧠 Usage Guide

### A) Batch generation
1. Set **in_dir** + **out_dir**
2. Choose what to generate:
   - Canny / OpenPose / Both
3. Adjust:
   - detect res
   - canny thresholds
   - quality toggles (optional)
4. Click **Run**
5. If enabled, browser auto-refreshes and loads newest outputs.

---

### B) Single image (solo) generation
1. In **Single Image** section:
   - pick from `in_dir` list OR browse file
2. Choose:
   - Canny / OpenPose / Both
3. Click **Run (Single)**

---

### C) Overlay before generating maps
Use the **Overlay** section:
- PNG overlay:
  - choose PNG (transparent supported)
  - position/scale/opacity/offset
- Text overlay:
  - text, font size, color
  - optional outline

This overlay is applied **before** detection, so your map includes it.

---

### D) Clean/remove text AFTER Canny (erase edges)
If your source image contains text and it creates ugly edge noise:

1. Generate Canny
2. Open **Canny Cleanup**
3. Load the canny map
4. Paint over the text region (mask)
5. Click **Erase to black (no edges)**
6. Save:
   - overwrite OR save `_edit.png`
7. Send the edited map to ControlNet

> Best practice: save edited versions as `_edit.png` so you keep the original too.

---

### E) Browse + preview maps
Use **Browse maps**:
- filter by type (canny / openpose)
- preview selected map
- see full path + relative key
- copy buttons available

Mini gallery shows latest 8 maps for fast grabbing.

---

### F) Send to ControlNet (Forge)
#### Send one map
- Pick a map in browser
- Choose **unit index**
- Click:
  - **Send to ControlNet (txt2img)** or
  - **Send to ControlNet (img2img)**

#### Send BOTH (consistent workflow)
- Set units:
  - Canny unit (default 0)
  - Pose unit (default 1)
- Click **Send BOTH** for txt2img or img2img

#### Send LATEST pair (fastest)
- Click:
  - **Send LATEST pair (txt2img)** or
  - **Send LATEST pair (img2img)**

#### Unit auto-detect
- Click **Detect units**
- Sliders max will update to match how many ControlNet slots your UI currently has.

---

## 🧩 CLI Usage (optional / debugging)
From:
`extensions/cn_hints_bridge/tools/`

### Batch
```bat
python batch_hints.py --mode batch --in_dir "X:\images" --out_dir "X:\maps" --do_canny 1 --do_openpose 1
```

### Single
```bat
python batch_hints.py --mode single --input "X:\images\one.png" --out_dir "X:\maps" --do_canny 1 --do_openpose 1
```

### Overlays (example)
```bat
python batch_hints.py --mode single --input "X:\images\one.png" --out_dir "X:\maps" ^
  --do_canny 1 ^
  --overlay_png "X:\overlay\logo.png" --overlay_png_pos "br" --overlay_png_scale 0.35 --overlay_png_opacity 0.85 ^
  --overlay_text "SAMPLE" --overlay_text_size 42 --overlay_text_pos "tl" --overlay_text_outline 2
```

---

## 🛠 Troubleshooting

### 1) UnicodeEncodeError (Windows cp1252)
If you ever see:
`UnicodeEncodeError: 'charmap' codec can't encode character ...`

Fix options:
- Run console as UTF-8:
```bat
chcp 65001
set PYTHONIOENCODING=utf-8
```

### 2) “mediapipe is not installed” warning
Safe to ignore unless you’re using features that require mediapipe.  
To silence:
```bat
pip install mediapipe
```

### 3) OpenPose is slow
Install CUDA PyTorch (recommended) and run in GPU-enabled environment. CPU works but slower.

### 4) Send-to-ControlNet doesn’t land in the right unit
- Use **Detect units**
- Adjust unit index sliders
- Some Forge skins/layouts may change DOM selectors; update the JS selectors if needed.

---

## 🧾 GitHub Notes
### Recommended `.gitignore`
```gitignore
# venv
cn_hints_env/
tools/cn_hints_env/
.venv/

# python cache
__pycache__/
*.pyc

# settings
cn_hints_settings.json

# logs / temp
*.log
tmp/
```

### Releases
- Tag versions like `v4.2`, `v4.3`, etc.
- Attach the zipped extension folder as a GitHub Release asset.

---

## 📌 Credits
- Built on top of Forge/Forge Neo + ControlNet workflows
- Uses `controlnet-aux` for detectors

---

