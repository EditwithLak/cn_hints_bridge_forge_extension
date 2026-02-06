# CN Hints Bridge (Forge / Forge Neo Extension)

Adds a **CN Hints** tab to Forge/Forge Neo that runs `tools/batch_hints.py` with UI controls.

### New in v4.2
- 🧱 **Run + Log moved to bottom** (no more mid-page log block)

### New in v4
- ✅ **Single image (solo) generation** (canny / openpose / both)
- ✅ **Overlay before detection**: PNG (alpha) + Text overlay (great for watermarks/censor bars)
- ✅ **Canny cleanup editor**: paint a mask to erase unwanted edges (defaults to **black / no-edges**)
- ✅ **Mini gallery grid** (latest 8 maps) for fast picking
- ✅ **Unit auto-detect** (sizes unit sliders to your current ControlNet slot count)
- ✅ **Copy buttons**: map absolute path + “relative key” (pairing/debug)
- ✅ **Send LATEST pair** one-click (no dropdown picking)

### Still included from v3
- **Send BOTH** one-click: Canny → unit A, OpenPose → unit B (user-selectable)
- **Auto-refresh** map browser after generation completes
- **Remember settings** across restarts (saved to `cn_hints_settings.json` in the extension folder)
- **Source → map pairing**: pick a source image and jump to its matching canny/openpose outputs
- **Canny quality toggles**: invert, thin/thicken, adaptive thresholding, clean background, speckle filtering

## Install
1. Copy `cn_hints_bridge/` into your WebUI folder:
   - `sd-webui-forge-neo/extensions/cn_hints_bridge/`
2. Restart Forge/Forge Neo.
3. Open the UI → tab **CN Hints**.

## Why subprocess?
Your script depends on `controlnet_aux`, `opencv-python`, etc. Running via subprocess lets you point to a separate venv
(e.g. `cn_hints_env\Scripts\python.exe`) without installing those deps into Forge’s env.

## Output structure
The script writes into:
- `<out_dir>/canny/<vertical|landscape>/...`
- `<out_dir>/openpose/<vertical|landscape>/...`

## Send → ControlNet (how it works)
The send buttons try to locate ControlNet image inputs in the target tab and drop the selected hint map into the chosen unit index.

If it can't find the right slot (UI changes between Forge builds), use the unit index slider or disable auto-config and manually drop the file.
