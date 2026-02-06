import sys
import os
import json
import subprocess
from pathlib import Path

import gradio as gr
from modules import script_callbacks

EXT_ROOT = Path(__file__).resolve().parents[1]  # .../extensions/cn_hints_bridge
DEFAULT_SCRIPT = str(EXT_ROOT / "tools" / "batch_hints.py")
SETTINGS_PATH = EXT_ROOT / "cn_hints_settings.json"


def _open_folder(path: str) -> str:
    """Open a folder in the OS file browser (Windows Explorer on win32)."""
    try:
        p = Path((path or "").strip())
        if not p.exists():
            return "❌ Folder not found"
        if sys.platform.startswith("win"):
            subprocess.Popen(["explorer", str(p)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(p)])
        else:
            subprocess.Popen(["xdg-open", str(p)])
        return f"📂 Opened: {p}"
    except Exception as e:
        return f"❌ Failed to open folder: {e}"


def _rel_key_from_map(rel_map_path: str) -> str:
    """Turn 'canny/vertical/foo/bar.png' into 'foo/bar' for pairing/debug."""
    try:
        parts = Path(rel_map_path).with_suffix("").parts
        if len(parts) >= 3 and parts[0] in {"canny", "openpose"} and parts[1] in {"vertical", "landscape"}:
            return str(Path(*parts[2:]))
        return str(Path(rel_map_path).with_suffix(""))
    except Exception:
        return ""


def _abs_map_path(out_dir: str, rel_map_path: str) -> str:
    try:
        return str((Path(out_dir) / rel_map_path).resolve())
    except Exception:
        return ""


def _find_latest_pair(out_dir: str, orient_filter: str = "all"):
    """Find the newest canny map, then try to find matching openpose by key."""
    out_dir = (out_dir or "").strip()
    if not out_dir:
        return None, None
    root = Path(out_dir)
    if not root.exists():
        return None, None

    canny_base = root / "canny"
    pose_base = root / "openpose"
    if (orient_filter or "all") in {"vertical", "landscape"}:
        canny_base = canny_base / orient_filter
        pose_base = pose_base / orient_filter

    canny_files = list(canny_base.rglob("*.png"))
    pose_files = list(pose_base.rglob("*.png"))
    if not canny_files and not pose_files:
        return None, None

    latest_canny = max(canny_files, key=lambda p: p.stat().st_mtime) if canny_files else None
    latest_pose = max(pose_files, key=lambda p: p.stat().st_mtime) if pose_files else None

    def rel(p: Path):
        try:
            return str(p.relative_to(root))
        except Exception:
            return str(p)

    if latest_canny is None:
        return None, rel(latest_pose)

    canny_rel = rel(latest_canny)
    key = _rel_key_from_map(canny_rel)

    # Prefer matching openpose using same orientation if possible
    cand = None
    try:
        parts = Path(canny_rel).parts
        orient = parts[1] if len(parts) > 1 else "vertical"
        p1 = root / "openpose" / orient / (key + ".png")
        if p1.exists():
            cand = p1
        else:
            for o in ("vertical", "landscape"):
                p2 = root / "openpose" / o / (key + ".png")
                if p2.exists():
                    cand = p2
                    break
    except Exception:
        cand = None

    pose_rel = rel(cand) if cand is not None else (rel(latest_pose) if latest_pose is not None else None)
    return canny_rel, pose_rel


def _gallery_items(out_dir: str, rel_paths: list[str]):
    """Return gr.Gallery-friendly items: [(pil, caption), ...]."""
    items = []
    for rp in rel_paths:
        img, _info = _load_preview(out_dir, rp)
        if img is None:
            continue
        items.append((img, rp))
    return items


def _blank_mask_like(out_dir: str, rel_map_path: str):
    try:
        from PIL import Image

        img, _ = _load_preview(out_dir, rel_map_path)
        if img is None:
            return None
        w, h = img.size
        return Image.new("RGB", (w, h), (0, 0, 0))
    except Exception:
        return None


def _apply_erase_mask(out_dir: str, rel_map_path: str, mask_img, erase_to: str, overwrite: bool):
    """Erase regions on a saved map using a user-painted mask (white => erase)."""
    try:
        from PIL import Image

        out_dir = (out_dir or "").strip()
        rel_map_path = (rel_map_path or "").strip()
        if not out_dir or not rel_map_path:
            return None, "❌ Pick a map first", gr.update(), None

        src = Path(out_dir) / rel_map_path
        if not src.exists():
            return None, f"❌ Missing: {src}", gr.update(), None

        if mask_img is None:
            return None, "❌ Provide / paint a mask", gr.update(), None

        img = Image.open(src).convert("RGB")
        mask = mask_img.convert("L")
        # Treat any non-black pixel as a mask
        mask = mask.point(lambda p: 255 if p > 10 else 0)

        fill = (0, 0, 0) if (erase_to or "black").lower().startswith("black") else (255, 255, 255)
        fill_img = Image.new("RGB", img.size, fill)
        out_img = Image.composite(fill_img, img, mask)

        if overwrite:
            dst = src
        else:
            dst = src.with_name(src.stem + "_edit" + src.suffix)
            i = 2
            while dst.exists():
                dst = src.with_name(src.stem + f"_edit{i}" + src.suffix)
                i += 1

        dst.parent.mkdir(parents=True, exist_ok=True)
        out_img.save(dst)

        # Return new rel + reset mask to blank for next edit
        try:
            rel_new = str(dst.relative_to(Path(out_dir)))
        except Exception:
            rel_new = str(dst)

        info = f"✅ Saved edited map: {rel_new}"
        blank = _blank_mask_like(out_dir, rel_new)
        return out_img, info, rel_new, blank
    except Exception as e:
        return None, f"❌ Edit failed: {e}", gr.update(), None


def _load_settings() -> dict:
    try:
        if SETTINGS_PATH.exists():
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        pass
    return {}


def _save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        # Avoid breaking UI on write failures (permissions etc.)
        pass


def _browse_dir():
    """Native folder picker (best on local installs)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        p = filedialog.askdirectory()
        root.destroy()
        return p or ""
    except Exception:
        return ""


def _browse_file():
    """Native file picker."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        p = filedialog.askopenfilename()
        root.destroy()
        return p or ""
    except Exception:
        return ""


def _build_cmd(
    python_exe: str,
    script_path: str,
    in_dir: str,
    out_dir: str,
    input_path: str,
    input_rel: str,
    mode: str,
    detect: int,
    portrait_size: str,
    landscape_size: str,
    canny: bool,
    openpose: bool,
    canny_low: int,
    canny_high: int,
    blur: int,
    canny_invert: bool,
    canny_thickness: str,
    canny_adaptive: bool,
    canny_clean_bg: bool,
    canny_clean_thresh: int,
    canny_speckle: str,
    clahe: bool,
    sharpen: bool,
    denoise: bool,
    hands: bool,
    face: bool,
    device: str,
    recursive: bool,
    skip_existing: bool,
    overlay_png: str,
    overlay_png_pos: str,
    overlay_png_scale: float,
    overlay_png_opacity: float,
    overlay_png_offx: int,
    overlay_png_offy: int,
    overlay_text: str,
    overlay_text_size: int,
    overlay_text_color: str,
    overlay_text_outline: bool,
    overlay_text_outline_color: str,
    overlay_text_pos: str,
    overlay_text_offx: int,
    overlay_text_offy: int,
):
    py = python_exe.strip() or sys.executable
    script = script_path.strip() or DEFAULT_SCRIPT

    cmd = [
        py,
        script,
        "--out_dir",
        out_dir,
        "--mode", mode,
        "--detect", str(int(detect)),
        "--portrait_size", portrait_size,
        "--landscape_size", landscape_size,
        "--canny_low", str(int(canny_low)),
        "--canny_high", str(int(canny_high)),
        "--blur", str(int(blur)),
        "--canny_thickness", str(canny_thickness or "none"),
        "--canny_clean_thresh", str(int(canny_clean_thresh)),
        "--canny_speckle", str(canny_speckle or "none"),
        "--device", device,
    ]

    # Batch vs single input
    if (input_path or "").strip():
        cmd.extend(["--input", input_path])
        if (input_rel or "").strip():
            cmd.extend(["--input_rel", input_rel])
    else:
        cmd.extend(["--in_dir", in_dir])

    # If user picks both, pass both (or you could omit both and let script default)
    if canny:
        cmd.append("--canny")
    if openpose:
        cmd.append("--openpose")

    if clahe:
        cmd.append("--clahe")
    if sharpen:
        cmd.append("--sharpen")
    if denoise:
        cmd.append("--denoise")

    if canny_invert:
        cmd.append("--canny_invert")
    if canny_adaptive:
        cmd.append("--canny_adaptive")
    if canny_clean_bg:
        cmd.append("--canny_clean_bg")

    if hands:
        cmd.append("--hands")
    if face:
        cmd.append("--face")

    if recursive:
        cmd.append("--recursive")
    if skip_existing:
        cmd.append("--skip_existing")

    # Optional overlay (before detection)
    if (overlay_png or "").strip():
        cmd.extend(
            [
                "--overlay_png",
                overlay_png,
                "--overlay_png_pos",
                str(overlay_png_pos or "bottom-right"),
                "--overlay_png_scale",
                str(float(overlay_png_scale or 0.25)),
                "--overlay_png_opacity",
                str(float(overlay_png_opacity or 1.0)),
                "--overlay_png_offx",
                str(int(overlay_png_offx or 0)),
                "--overlay_png_offy",
                str(int(overlay_png_offy or 0)),
            ]
        )

    if (overlay_text or "").strip():
        cmd.extend(
            [
                "--overlay_text",
                overlay_text,
                "--overlay_text_size",
                str(int(overlay_text_size or 28)),
                "--overlay_text_color",
                str(overlay_text_color or "#FFFFFF"),
                "--overlay_text_pos",
                str(overlay_text_pos or "bottom-right"),
                "--overlay_text_offx",
                str(int(overlay_text_offx or 0)),
                "--overlay_text_offy",
                str(int(overlay_text_offy or 0)),
                "--overlay_text_outline_color",
                str(overlay_text_outline_color or "#000000"),
            ]
        )
        if overlay_text_outline:
            cmd.append("--overlay_text_outline")

    return cmd


def _run_subprocess(cmd):
    """Stream subprocess stdout as a generator."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    try:
        for line in iter(p.stdout.readline, ""):
            if not line:
                break
            yield line.rstrip("\n")
    finally:
        try:
            p.stdout.close()
        except Exception:
            pass
        p.wait()


def _scan_hint_files(
    out_dir: str,
    kind: str,
    orient: str,
    name_filter: str,
    sort_mode: str,
    max_items: int,
):
    """Scan <out_dir>/(canny|openpose)/(vertical|landscape)/... for PNG maps."""
    out_dir = (out_dir or "").strip()
    if not out_dir:
        return []

    base = Path(out_dir) / kind
    if orient in {"vertical", "landscape"}:
        base = base / orient

    if not base.exists():
        return []

    files = list(base.rglob("*.png"))

    nf = (name_filter or "").strip().lower()
    if nf:
        files = [p for p in files if nf in p.name.lower() or nf in str(p).lower()]

    if sort_mode == "modified":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        files.sort(key=lambda p: str(p).lower())

    if max_items and max_items > 0:
        files = files[: int(max_items)]

    # Return relative paths from out_dir to keep dropdown clean
    root = Path(out_dir)
    rels = []
    for p in files:
        try:
            rels.append(str(p.relative_to(root)))
        except Exception:
            rels.append(str(p))
    return rels




def _latest_hint_files(out_dir: str, kind: str, orient: str, limit: int = 8):
    """Return newest <limit> PNG maps for <kind>, optionally filtered by orientation."""
    out_dir = (out_dir or "").strip()
    if not out_dir:
        return []

    base = Path(out_dir) / kind
    if orient in {"vertical", "landscape"}:
        base = base / orient

    if not base.exists():
        return []

    files = list(base.rglob("*.png"))
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    files = files[: int(limit or 8)]

    root = Path(out_dir)
    rels = []
    for fp in files:
        try:
            rels.append(str(fp.relative_to(root)))
        except Exception:
            rels.append(str(fp))
    return rels
def _load_preview(out_dir: str, rel_path: str):
    """Load a selected PNG and return (pil_image, info_text)."""
    out_dir = (out_dir or "").strip()
    rel_path = (rel_path or "").strip()
    if not out_dir or not rel_path:
        return None, ""

    p = Path(out_dir) / rel_path
    if not p.exists():
        return None, f"❌ Missing: {p}"

    try:
        from PIL import Image

        img = Image.open(p)
        w, h = img.size
        kb = p.stat().st_size / 1024
        info = f"📄 {rel_path}  •  {w}×{h}  •  {kb:.1f} KB"
        return img, info
    except Exception as e:
        return None, f"❌ Failed to load preview: {e}"


def _scan_source_files(
    in_dir: str,
    recursive: bool,
    name_filter: str,
    sort_mode: str,
    max_items: int,
):
    """Scan <in_dir> (optionally recursive) for common image formats."""
    in_dir = (in_dir or "").strip()
    if not in_dir:
        return []

    base = Path(in_dir)
    if not base.exists():
        return []

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    files = list(base.rglob("*") if recursive else base.iterdir())
    files = [p for p in files if p.is_file() and p.suffix.lower() in exts]

    nf = (name_filter or "").strip().lower()
    if nf:
        files = [p for p in files if nf in p.name.lower() or nf in str(p).lower()]

    if sort_mode == "modified":
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        files.sort(key=lambda p: str(p).lower())

    if max_items and max_items > 0:
        files = files[: int(max_items)]

    rels = []
    for p in files:
        try:
            rels.append(str(p.relative_to(base)))
        except Exception:
            rels.append(str(p))
    return rels


def _load_source_preview(in_dir: str, rel_path: str):
    in_dir = (in_dir or "").strip()
    rel_path = (rel_path or "").strip()
    if not in_dir or not rel_path:
        return None, ""

    p = Path(in_dir) / rel_path
    if not p.exists():
        return None, f"❌ Missing: {p}"

    try:
        from PIL import Image

        img = Image.open(p)
        w, h = img.size
        kb = p.stat().st_size / 1024
        info = f"🖼️ {rel_path}  •  {w}×{h}  •  {kb:.1f} KB"
        return img, info
    except Exception as e:
        return None, f"❌ Failed to load source preview: {e}"


def _pair_maps_for_source(out_dir: str, source_rel: str):
    """Map an input image (relative to in_dir) to its corresponding outputs.

    Output layout is:
      <out_dir>/canny/<vertical|landscape>/<source_rel_without_ext>.png
      <out_dir>/openpose/<vertical|landscape>/<source_rel_without_ext>.png
    We don't assume orientation: we probe both vertical and landscape.
    """
    out_dir = (out_dir or "").strip()
    source_rel = (source_rel or "").strip()
    if not out_dir or not source_rel:
        return None, None

    stem = Path(source_rel).with_suffix("")
    root = Path(out_dir)

    def pick(kind: str):
        for orient in ("vertical", "landscape"):
            p = root / kind / orient / (str(stem) + ".png")
            if p.exists():
                try:
                    return str(p.relative_to(root))
                except Exception:
                    return str(p)
        return None

    return pick("canny"), pick("openpose")


def on_ui_tabs():
    s = _load_settings()

    with gr.Blocks() as ui:
        gr.Markdown("## 🧩 Bulk Canny + OpenPose Hint Generator")
        gr.Markdown("This tab wraps `batch_hints.py` with UI controls and runs it.")

        with gr.Accordion("Runtime / Paths", open=True):
            with gr.Row():
                auto_save = gr.Checkbox(value=s.get("auto_save", True), label="Auto-save settings")
                save_btn = gr.Button("💾 Save settings", scale=0)
                settings_status = gr.Textbox(value="", label="Settings status", lines=1, interactive=False)

            with gr.Row():
                python_exe = gr.Textbox(
                    value=s.get("python_exe", ""),
                    label="Python executable (optional)",
                    placeholder="Leave blank to use Forge's python. Or set cn_hints_env\\Scripts\\python.exe",
                )
                py_pick = gr.Button("Browse python.exe", scale=0)

            with gr.Row():
                script_path = gr.Textbox(value=s.get("script_path", DEFAULT_SCRIPT) or DEFAULT_SCRIPT, label="Script path (batch_hints.py)")
                script_pick = gr.Button("Browse .py", scale=0)

            with gr.Row():
                in_dir = gr.Textbox(value=s.get("in_dir", ""), label="Input folder (in_dir)")
                in_browse = gr.Button("Browse", scale=0)

            with gr.Row():
                out_dir = gr.Textbox(value=s.get("out_dir", ""), label="Output folder (out_dir)")
                out_browse = gr.Button("Browse", scale=0)
                open_out_dir = gr.Button("📂 Open output", scale=0)

        with gr.Accordion("Core Settings", open=True):
            with gr.Row():
                mode = gr.Dropdown(choices=["cover", "contain"], value=s.get("mode", "cover"), label="Fit mode")
                detect = gr.Slider(64, 2048, value=int(s.get("detect", 512)), step=64, label="Detect resolution (speed vs detail)")

            with gr.Row():
                portrait_size = gr.Textbox(value=s.get("portrait_size", "896x1344"), label="Portrait size")
                landscape_size = gr.Textbox(value=s.get("landscape_size", "1344x896"), label="Landscape size")

            with gr.Row():
                do_canny = gr.Checkbox(value=bool(s.get("do_canny", True)), label="Generate Canny")
                do_openpose = gr.Checkbox(value=bool(s.get("do_openpose", True)), label="Generate OpenPose")
                recursive = gr.Checkbox(value=bool(s.get("recursive", False)), label="Recursive")
                skip_existing = gr.Checkbox(value=bool(s.get("skip_existing", True)), label="Skip existing outputs")

            with gr.Row():
                auto_refresh_after_run = gr.Checkbox(value=bool(s.get("auto_refresh_after_run", True)), label="Auto-refresh browser after Run")

        with gr.Accordion("Overlay before detection (optional)", open=False):
            gr.Markdown("Overlay a **PNG (alpha)** or **text** onto the input *before* generating Canny/OpenPose. Useful for stamps, censor blocks, marks, etc.")

            with gr.Row():
                overlay_png = gr.Textbox(value=s.get("overlay_png", ""), label="Overlay PNG path (optional)")
                overlay_png_pick = gr.Button("Browse", scale=0)

            with gr.Row():
                overlay_png_pos = gr.Dropdown(
                    choices=["top-left", "top-right", "top-center", "center", "bottom-left", "bottom-right", "bottom-center"],
                    value=str(s.get("overlay_png_pos", "bottom-right")),
                    label="PNG position",
                )
                overlay_png_scale = gr.Slider(0.05, 1.5, value=float(s.get("overlay_png_scale", 0.25)), step=0.01, label="PNG scale (fraction of width)")
                overlay_png_opacity = gr.Slider(0.0, 1.0, value=float(s.get("overlay_png_opacity", 1.0)), step=0.01, label="PNG opacity")

            with gr.Row():
                overlay_png_offx = gr.Slider(-1024, 1024, value=int(s.get("overlay_png_offx", 0)), step=1, label="PNG offset X")
                overlay_png_offy = gr.Slider(-1024, 1024, value=int(s.get("overlay_png_offy", 0)), step=1, label="PNG offset Y")

            gr.Markdown("### Text overlay")
            with gr.Row():
                overlay_text = gr.Textbox(value=s.get("overlay_text", ""), label="Text (optional)", placeholder="e.g. mask / censor / title")
                overlay_text_size = gr.Slider(6, 256, value=int(s.get("overlay_text_size", 28)), step=1, label="Text size")

            with gr.Row():
                overlay_text_color = gr.Textbox(value=str(s.get("overlay_text_color", "#FFFFFF")), label="Text color (hex)")
                overlay_text_outline = gr.Checkbox(value=bool(s.get("overlay_text_outline", False)), label="Outline")
                overlay_text_outline_color = gr.Textbox(value=str(s.get("overlay_text_outline_color", "#000000")), label="Outline color (hex)")

            with gr.Row():
                overlay_text_pos = gr.Dropdown(
                    choices=["top-left", "top-right", "top-center", "center", "bottom-left", "bottom-right", "bottom-center"],
                    value=str(s.get("overlay_text_pos", "bottom-right")),
                    label="Text position",
                )
                overlay_text_offx = gr.Slider(-1024, 1024, value=int(s.get("overlay_text_offx", 0)), step=1, label="Text offset X")
                overlay_text_offy = gr.Slider(-1024, 1024, value=int(s.get("overlay_text_offy", 0)), step=1, label="Text offset Y")

        with gr.Accordion("Canny Settings", open=False):
            with gr.Row():
                canny_low = gr.Slider(1, 500, value=int(s.get("canny_low", 150)), step=1, label="Low threshold")
                canny_high = gr.Slider(1, 800, value=int(s.get("canny_high", 300)), step=1, label="High threshold")
                blur = gr.Dropdown(choices=[0, 3, 5, 7, 9], value=int(s.get("blur", 3)), label="Pre-blur kernel (odd)")

            with gr.Row():
                clahe = gr.Checkbox(value=bool(s.get("clahe", False)), label="CLAHE")
                sharpen = gr.Checkbox(value=bool(s.get("sharpen", False)), label="Sharpen")
                denoise = gr.Checkbox(value=bool(s.get("denoise", False)), label="Denoise")

            gr.Markdown("### ✨ Canny quality (optional post-process)")
            with gr.Row():
                canny_invert = gr.Checkbox(value=bool(s.get("canny_invert", False)), label="Invert")
                canny_thickness = gr.Dropdown(
                    choices=["none", "thin", "thick", "extra_thick"],
                    value=str(s.get("canny_thickness", "none")),
                    label="Edge thickness",
                )
                canny_speckle = gr.Dropdown(
                    choices=["none", "median3", "median5"],
                    value=str(s.get("canny_speckle", "none")),
                    label="Speckle filter",
                )

            with gr.Row():
                canny_adaptive = gr.Checkbox(value=bool(s.get("canny_adaptive", False)), label="Adaptive threshold (autocontrast)")
                canny_clean_bg = gr.Checkbox(value=bool(s.get("canny_clean_bg", False)), label="Clean background (pure B/W)")
                canny_clean_thresh = gr.Slider(0, 255, value=int(s.get("canny_clean_thresh", 128)), step=1, label="Clean threshold")

        with gr.Accordion("OpenPose Settings", open=False):
            with gr.Row():
                device = gr.Dropdown(choices=["cpu", "cuda"], value=str(s.get("device", "cpu")), label="Device")
                hands = gr.Checkbox(value=bool(s.get("hands", False)), label="Include hands")
                face = gr.Checkbox(value=bool(s.get("face", False)), label="Include face")


        # -------------------------------
        # Single image (solo) generation
        # -------------------------------
        with gr.Accordion("Single image (solo) generation", open=False):
            gr.Markdown("Run **one image** through Canny/OpenPose without batch scanning. Preserves pairing by writing into out_dir using a relative key.")
            with gr.Row():
                single_mode = gr.Radio(
                    choices=["Pick from in_dir", "Browse file"],
                    value="Pick from in_dir",
                    label="Single input source",
                )
                single_refresh = gr.Button("🔄 Refresh list", scale=0)

            single_src_dd = gr.Dropdown(choices=[], value=None, label="Source from in_dir")
            with gr.Row():
                single_preview = gr.Image(value=None, label="Single source preview", type="pil")
                single_info = gr.Textbox(value="", label="Single source info", lines=3, interactive=False)

            with gr.Row():
                single_file = gr.Textbox(value="", label="Or browse a file path")
                single_file_pick = gr.Button("Browse", scale=0)

            run_single_btn = gr.Button("🚀 Run single image", variant="primary")
            single_status = gr.Textbox(value="", label="Single run status", lines=2, interactive=False)

        # Browse buttons
        py_pick.click(lambda: _browse_file(), outputs=[python_exe])
        script_pick.click(lambda: _browse_file(), outputs=[script_path])
        in_browse.click(lambda: _browse_dir(), outputs=[in_dir])
        out_browse.click(lambda: _browse_dir(), outputs=[out_dir])
        overlay_png_pick.click(lambda: _browse_file(), outputs=[overlay_png])
        open_out_dir.click(fn=_open_folder, inputs=[out_dir], outputs=[settings_status])


        def _pack_settings(
            auto_save_v,
            python_exe_v,
            script_path_v,
            in_dir_v,
            out_dir_v,
            mode_v,
            detect_v,
            portrait_v,
            landscape_v,
            do_canny_v,
            do_openpose_v,
            canny_low_v,
            canny_high_v,
            blur_v,
            canny_invert_v,
            canny_thickness_v,
            canny_speckle_v,
            canny_adaptive_v,
            canny_clean_bg_v,
            canny_clean_thresh_v,
            clahe_v,
            sharpen_v,
            denoise_v,
            hands_v,
            face_v,
            device_v,
            recursive_v,
            skip_existing_v,
            auto_refresh_after_run_v,
            overlay_png_v,
            overlay_png_pos_v,
            overlay_png_scale_v,
            overlay_png_opacity_v,
            overlay_png_offx_v,
            overlay_png_offy_v,
            overlay_text_v,
            overlay_text_size_v,
            overlay_text_color_v,
            overlay_text_outline_v,
            overlay_text_outline_color_v,
            overlay_text_pos_v,
            overlay_text_offx_v,
            overlay_text_offy_v,
        ):
            return {
                "auto_save": bool(auto_save_v),
                "python_exe": python_exe_v or "",
                "script_path": script_path_v or DEFAULT_SCRIPT,
                "in_dir": in_dir_v or "",
                "out_dir": out_dir_v or "",
                "mode": mode_v or "cover",
                "detect": int(detect_v) if detect_v is not None else 512,
                "portrait_size": portrait_v or "896x1344",
                "landscape_size": landscape_v or "1344x896",
                "do_canny": bool(do_canny_v),
                "do_openpose": bool(do_openpose_v),
                "canny_low": int(canny_low_v) if canny_low_v is not None else 150,
                "canny_high": int(canny_high_v) if canny_high_v is not None else 300,
                "blur": int(blur_v) if blur_v is not None else 3,
                "clahe": bool(clahe_v),
                "sharpen": bool(sharpen_v),
                "denoise": bool(denoise_v),
                "canny_invert": bool(canny_invert_v),
                "canny_thickness": str(canny_thickness_v or "none"),
                "canny_speckle": str(canny_speckle_v or "none"),
                "canny_adaptive": bool(canny_adaptive_v),
                "canny_clean_bg": bool(canny_clean_bg_v),
                "canny_clean_thresh": int(canny_clean_thresh_v) if canny_clean_thresh_v is not None else 128,
                "device": str(device_v or "cpu"),
                "hands": bool(hands_v),
                "face": bool(face_v),
                "recursive": bool(recursive_v),
                "skip_existing": bool(skip_existing_v),
                "auto_refresh_after_run": bool(auto_refresh_after_run_v),

                "overlay_png": overlay_png_v or "",
                "overlay_png_pos": str(overlay_png_pos_v or "bottom-right"),
                "overlay_png_scale": float(overlay_png_scale_v) if overlay_png_scale_v is not None else 0.25,
                "overlay_png_opacity": float(overlay_png_opacity_v) if overlay_png_opacity_v is not None else 1.0,
                "overlay_png_offx": int(overlay_png_offx_v) if overlay_png_offx_v is not None else 0,
                "overlay_png_offy": int(overlay_png_offy_v) if overlay_png_offy_v is not None else 0,
                "overlay_text": overlay_text_v or "",
                "overlay_text_size": int(overlay_text_size_v) if overlay_text_size_v is not None else 28,
                "overlay_text_color": str(overlay_text_color_v or "#FFFFFF"),
                "overlay_text_outline": bool(overlay_text_outline_v),
                "overlay_text_outline_color": str(overlay_text_outline_color_v or "#000000"),
                "overlay_text_pos": str(overlay_text_pos_v or "bottom-right"),
                "overlay_text_offx": int(overlay_text_offx_v) if overlay_text_offx_v is not None else 0,
                "overlay_text_offy": int(overlay_text_offy_v) if overlay_text_offy_v is not None else 0,
            }

        def _maybe_save_settings(auto_save_v, *vals):
            if not auto_save_v:
                return "⚪ Auto-save is OFF"
            data = _pack_settings(auto_save_v, *vals)
            _save_settings(data)
            return "💾 Auto-saved"

        def _save_settings_click(auto_save_v, *vals):
            data = _pack_settings(auto_save_v, *vals)
            _save_settings(data)
            return "✅ Saved"

        save_btn.click(
            _save_settings_click,
            inputs=[
                auto_save,
                python_exe, script_path, in_dir, out_dir, mode, detect,
                portrait_size, landscape_size, do_canny, do_openpose,
                canny_low, canny_high, blur,
                canny_invert, canny_thickness, canny_speckle, canny_adaptive, canny_clean_bg, canny_clean_thresh,
                clahe, sharpen, denoise,
                hands, face, device,
                recursive, skip_existing, auto_refresh_after_run,
                overlay_png, overlay_png_pos, overlay_png_scale, overlay_png_opacity, overlay_png_offx, overlay_png_offy,
                overlay_text, overlay_text_size, overlay_text_color, overlay_text_outline, overlay_text_outline_color,
                overlay_text_pos, overlay_text_offx, overlay_text_offy,
            ],
            outputs=[settings_status],
        )

        # Auto-save on key changes (keeps the list readable: we wire the same handler)
        _autosave_inputs = [
            auto_save,
            python_exe, script_path, in_dir, out_dir, mode, detect,
            portrait_size, landscape_size, do_canny, do_openpose,
            canny_low, canny_high, blur,
            canny_invert, canny_thickness, canny_speckle, canny_adaptive, canny_clean_bg, canny_clean_thresh,
            clahe, sharpen, denoise,
            hands, face, device,
            recursive, skip_existing, auto_refresh_after_run,
            overlay_png, overlay_png_pos, overlay_png_scale, overlay_png_opacity, overlay_png_offx, overlay_png_offy,
            overlay_text, overlay_text_size, overlay_text_color, overlay_text_outline, overlay_text_outline_color,
            overlay_text_pos, overlay_text_offx, overlay_text_offy,
        ]

        for c in [
            python_exe, script_path, in_dir, out_dir, mode, detect,
            portrait_size, landscape_size, do_canny, do_openpose,
            canny_low, canny_high, blur,
            canny_invert, canny_thickness, canny_speckle, canny_adaptive, canny_clean_bg, canny_clean_thresh,
            clahe, sharpen, denoise,
            hands, face, device,
            recursive, skip_existing, auto_refresh_after_run,
            overlay_png, overlay_png_pos, overlay_png_scale, overlay_png_opacity, overlay_png_offx, overlay_png_offy,
            overlay_text, overlay_text_size, overlay_text_color, overlay_text_outline, overlay_text_outline_color,
            overlay_text_pos, overlay_text_offx, overlay_text_offy,
        ]:
            c.change(
                _maybe_save_settings,
                inputs=_autosave_inputs,
                outputs=[settings_status],
            )

        def run_click(
            auto_save_v,
            python_exe_v,
            script_path_v,
            in_dir_v,
            out_dir_v,
            mode_v,
            detect_v,
            portrait_v,
            landscape_v,
            canny_v,
            openpose_v,
            canny_low_v,
            canny_high_v,
            blur_v,
            canny_invert_v,
            canny_thickness_v,
            canny_speckle_v,
            canny_adaptive_v,
            canny_clean_bg_v,
            canny_clean_thresh_v,
            clahe_v,
            sharpen_v,
            denoise_v,
            hands_v,
            face_v,
            device_v,
            recursive_v,
            skip_existing_v,
            auto_refresh_after_run_v,
            overlay_png_v,
            overlay_png_pos_v,
            overlay_png_scale_v,
            overlay_png_opacity_v,
            overlay_png_offx_v,
            overlay_png_offy_v,
            overlay_text_v,
            overlay_text_size_v,
            overlay_text_color_v,
            overlay_text_outline_v,
            overlay_text_outline_color_v,
            overlay_text_pos_v,
            overlay_text_offx_v,
            overlay_text_offy_v,
            gen_done_v,
        ):
            # Save on run (even if auto-save is off), so settings survive restarts.
            _save_settings(
                _pack_settings(
                    auto_save_v,
                    python_exe_v,
                    script_path_v,
                    in_dir_v,
                    out_dir_v,
                    mode_v,
                    detect_v,
                    portrait_v,
                    landscape_v,
                    canny_v,
                    openpose_v,
                    canny_low_v,
                    canny_high_v,
                    blur_v,
                    canny_invert_v,
                    canny_thickness_v,
                    canny_speckle_v,
                    canny_adaptive_v,
                    canny_clean_bg_v,
                    canny_clean_thresh_v,
                    clahe_v,
                    sharpen_v,
                    denoise_v,
                    hands_v,
                    face_v,
                    device_v,
                    recursive_v,
                    skip_existing_v,
                    auto_refresh_after_run_v,
                    overlay_png_v,
                    overlay_png_pos_v,
                    overlay_png_scale_v,
                    overlay_png_opacity_v,
                    overlay_png_offx_v,
                    overlay_png_offy_v,
                    overlay_text_v,
                    overlay_text_size_v,
                    overlay_text_color_v,
                    overlay_text_outline_v,
                    overlay_text_outline_color_v,
                    overlay_text_pos_v,
                    overlay_text_offx_v,
                    overlay_text_offy_v,
                )
            )

            if not in_dir_v or not out_dir_v:
                yield "❌ Please set both input and output folders."
                return

            if not (canny_v or openpose_v):
                yield "❌ Select at least one: Canny and/or OpenPose."
                return

            cmd = _build_cmd(
                python_exe_v,
                script_path_v,
                in_dir_v,
                out_dir_v,
                "",  # input_path (batch)
                "",  # input_rel  (batch)
                mode_v,
                detect_v,
                portrait_v,
                landscape_v,
                canny_v,
                openpose_v,
                canny_low_v,
                canny_high_v,
                int(blur_v),
                bool(canny_invert_v),
                str(canny_thickness_v or "none"),
                bool(canny_adaptive_v),
                bool(canny_clean_bg_v),
                int(canny_clean_thresh_v),
                str(canny_speckle_v or "none"),
                bool(clahe_v),
                bool(sharpen_v),
                bool(denoise_v),
                bool(hands_v),
                bool(face_v),
                str(device_v or "cpu"),
                bool(recursive_v),
                bool(skip_existing_v),
                overlay_png_v,
                overlay_png_pos_v,
                float(overlay_png_scale_v),
                float(overlay_png_opacity_v),
                int(overlay_png_offx_v),
                int(overlay_png_offy_v),
                overlay_text_v,
                int(overlay_text_size_v),
                overlay_text_color_v,
                bool(overlay_text_outline_v),
                overlay_text_outline_color_v,
                overlay_text_pos_v,
                int(overlay_text_offx_v),
                int(overlay_text_offy_v),
            )

            buf = []
            buf.append("✅ Running:")
            buf.append(" ".join(cmd))
            buf.append("")
            yield "\n".join(buf), int(gen_done_v or 0)
            for line in _run_subprocess(cmd):
                buf.append(line)
                yield "\n".join(buf), int(gen_done_v or 0)
            buf.append("\n✅ Done.")
            yield "\n".join(buf), int(gen_done_v or 0) + 1


        # --- Single image bindings ---
        def _refresh_single_list(in_dir_v, recursive_v):
            items = _scan_source_files(in_dir_v, bool(recursive_v), "", "modified", 2000)
            val = items[0] if items else None
            img, meta = _load_source_preview(in_dir_v, val) if val else (None, "⚠️ No sources found")
            return gr.update(choices=items, value=val), img, meta

        def _load_single_preview(single_mode_v, in_dir_v, rel_v, file_v):
            if (single_mode_v or "").startswith("Pick"):
                return _load_source_preview(in_dir_v, rel_v)
            # Browse file
            fp = (file_v or "").strip()
            if not fp:
                return None, ""
            try:
                from PIL import Image

                img = Image.open(fp)
                w, h = img.size
                kb = Path(fp).stat().st_size / 1024 if Path(fp).exists() else 0
                return img, f"📄 {fp}  •  {w}×{h}  •  {kb:.1f} KB"
            except Exception as e:
                return None, f"❌ Failed to load: {e}"

        def run_single_click(
            auto_save_v,
            python_exe_v,
            script_path_v,
            in_dir_v,
            out_dir_v,
            mode_v,
            detect_v,
            portrait_v,
            landscape_v,
            canny_v,
            openpose_v,
            canny_low_v,
            canny_high_v,
            blur_v,
            canny_invert_v,
            canny_thickness_v,
            canny_speckle_v,
            canny_adaptive_v,
            canny_clean_bg_v,
            canny_clean_thresh_v,
            clahe_v,
            sharpen_v,
            denoise_v,
            hands_v,
            face_v,
            device_v,
            recursive_v,
            skip_existing_v,
            auto_refresh_after_run_v,
            overlay_png_v,
            overlay_png_pos_v,
            overlay_png_scale_v,
            overlay_png_opacity_v,
            overlay_png_offx_v,
            overlay_png_offy_v,
            overlay_text_v,
            overlay_text_size_v,
            overlay_text_color_v,
            overlay_text_outline_v,
            overlay_text_outline_color_v,
            overlay_text_pos_v,
            overlay_text_offx_v,
            overlay_text_offy_v,
            gen_done_v,
            single_mode_v,
            single_src_rel_v,
            single_file_v,
        ):
            # Save settings (if enabled)
            s = _pack_settings(
                auto_save_v,
                python_exe_v,
                script_path_v,
                in_dir_v,
                out_dir_v,
                mode_v,
                detect_v,
                portrait_v,
                landscape_v,
                canny_v,
                openpose_v,
                canny_low_v,
                canny_high_v,
                blur_v,
                canny_invert_v,
                canny_thickness_v,
                canny_speckle_v,
                canny_adaptive_v,
                canny_clean_bg_v,
                canny_clean_thresh_v,
                clahe_v,
                sharpen_v,
                denoise_v,
                hands_v,
                face_v,
                device_v,
                recursive_v,
                skip_existing_v,
                auto_refresh_after_run_v,
                overlay_png_v,
                overlay_png_pos_v,
                overlay_png_scale_v,
                overlay_png_opacity_v,
                overlay_png_offx_v,
                overlay_png_offy_v,
                overlay_text_v,
                overlay_text_size_v,
                overlay_text_color_v,
                overlay_text_outline_v,
                overlay_text_outline_color_v,
                overlay_text_pos_v,
                overlay_text_offx_v,
                overlay_text_offy_v,
            )

            if not out_dir_v:
                yield "❌ Please set an output folder (out_dir).", int(gen_done_v or 0), "❌ Missing out_dir"
                return

            if not (canny_v or openpose_v):
                yield "❌ Select at least one: Canny and/or OpenPose.", int(gen_done_v or 0), "❌ Nothing selected"
                return

            input_path = ""
            input_rel = ""
            if (single_mode_v or "").startswith("Pick"):
                if not in_dir_v or not single_src_rel_v:
                    yield "❌ Pick a source from in_dir (or switch to Browse file).", int(gen_done_v or 0), "❌ No source selected"
                    return
                input_path = str(Path(in_dir_v) / str(single_src_rel_v))
                input_rel = str(single_src_rel_v)
            else:
                if not single_file_v:
                    yield "❌ Browse/select a file path.", int(gen_done_v or 0), "❌ No file"
                    return
                input_path = str(single_file_v)
                input_rel = Path(input_path).name
                if not in_dir_v:
                    in_dir_v = str(Path(input_path).parent)

            cmd = _build_cmd(
                python_exe_v,
                script_path_v,
                in_dir_v,
                out_dir_v,
                input_path,
                input_rel,
                mode_v,
                detect_v,
                portrait_v,
                landscape_v,
                canny_v,
                openpose_v,
                canny_low_v,
                canny_high_v,
                int(blur_v),
                bool(canny_invert_v),
                str(canny_thickness_v or "none"),
                bool(canny_adaptive_v),
                bool(canny_clean_bg_v),
                int(canny_clean_thresh_v),
                str(canny_speckle_v or "none"),
                bool(clahe_v),
                bool(sharpen_v),
                bool(denoise_v),
                bool(hands_v),
                bool(face_v),
                str(device_v or "cpu"),
                bool(recursive_v),
                bool(skip_existing_v),
                overlay_png_v,
                overlay_png_pos_v,
                float(overlay_png_scale_v),
                float(overlay_png_opacity_v),
                int(overlay_png_offx_v),
                int(overlay_png_offy_v),
                overlay_text_v,
                int(overlay_text_size_v),
                overlay_text_color_v,
                bool(overlay_text_outline_v),
                overlay_text_outline_color_v,
                overlay_text_pos_v,
                int(overlay_text_offx_v),
                int(overlay_text_offy_v),
            )

            buf = []
            buf.append("✅ Running single image:")
            buf.append(" ".join(cmd))
            buf.append("")
            yield "\n".join(buf), int(gen_done_v or 0), "🚀 Started"
            for line in _run_subprocess(cmd):
                buf.append(line)
                yield "\n".join(buf), int(gen_done_v or 0), "⏳ Running…"
            buf.append("\n✅ Done.")
            yield "\n".join(buf), int(gen_done_v or 0) + 1, "✅ Done"

        single_refresh.click(
            _refresh_single_list,
            inputs=[in_dir, recursive],
            outputs=[single_src_dd, single_preview, single_info],
        )

        single_src_dd.change(
            _load_single_preview,
            inputs=[single_mode, in_dir, single_src_dd, single_file],
            outputs=[single_preview, single_info],
        )

        single_file_pick.click(lambda: _browse_file(), outputs=[single_file])

        single_file.change(
            _load_single_preview,
            inputs=[single_mode, in_dir, single_src_dd, single_file],
            outputs=[single_preview, single_info],
        )

        # -------------------------------
        # Browse / Preview / Send
        # -------------------------------
        with gr.Accordion("Browse / Pair / Send (Canny + OpenPose)", open=True):
            gr.Markdown(
                "Browse generated **Canny** + **OpenPose** maps with previews, jump from a **source image** to matching outputs, "
                "and send either one (or both) into ControlNet in **txt2img** / **img2img**."
            )

            with gr.Row():
                orient = gr.Dropdown(choices=["all", "vertical", "landscape"], value="all", label="Orientation")
                sort_mode = gr.Dropdown(choices=["modified", "name"], value="modified", label="Sort")
                map_filter = gr.Textbox(value="", label="Map filter (optional)", placeholder="e.g. 00012 or subfolder")
                max_items = gr.Slider(10, 2000, value=500, step=10, label="Max maps")
                refresh_maps_btn = gr.Button("🔄 Refresh maps", scale=0)

            gr.Markdown("### 🔗 Source → map pairing")
            with gr.Row():
                src_filter = gr.Textbox(value="", label="Source filter (optional)", placeholder="match filename/subfolder")
                src_max = gr.Slider(10, 2000, value=500, step=10, label="Max sources")
                refresh_src_btn = gr.Button("🔄 Refresh sources", scale=0)

            src_dd = gr.Dropdown(choices=[], value=None, label="Source image (in_dir)")
            with gr.Row():
                src_preview = gr.Image(value=None, label="Source preview", type="pil")
                src_info = gr.Textbox(value="", label="Source info", lines=3, interactive=False)

            pair_btn = gr.Button("🎯 Jump to matching outputs", variant="secondary")

            gr.Markdown("### 🧩 Output maps")
            with gr.Row():
                canny_dd = gr.Dropdown(choices=[], value=None, label="Canny map")
                pose_dd = gr.Dropdown(choices=[], value=None, label="OpenPose map")

            with gr.Row():
                canny_preview = gr.Image(value=None, label="Canny preview", type="pil", elem_id="cnbridge_preview_canny")
                pose_preview = gr.Image(value=None, label="Pose preview", type="pil", elem_id="cnbridge_preview_pose")

            with gr.Row():
                canny_info = gr.Textbox(value="", label="Canny file info", lines=3, interactive=False)
                pose_info = gr.Textbox(value="", label="Pose file info", lines=3, interactive=False)

            # --- Mini gallery (latest 8) ---
            with gr.Accordion("🖼 Mini gallery (latest 8)", open=False):
                gr.Markdown("Quick-pick from the most recently generated maps. Click a thumbnail to load it.")

                with gr.Row():
                    gallery_refresh_btn = gr.Button("🖼 Refresh gallery", scale=0)

                canny_gallery_state = gr.State([])
                pose_gallery_state = gr.State([])

                with gr.Row():
                    canny_gallery = gr.Gallery(label="Latest Canny (8)", show_label=True)
                    pose_gallery = gr.Gallery(label="Latest OpenPose (8)", show_label=True)
                    try:
                        canny_gallery.style(grid=[4], height="auto")
                        pose_gallery.style(grid=[4], height="auto")
                    except Exception:
                        pass

            # --- Debug helpers (copy path / key) ---
            with gr.Accordion("🔧 Debug helpers (copy path / key)", open=False):
                with gr.Row():
                    canny_abs_txt = gr.Textbox(value="", label="Canny absolute path", interactive=False, elem_id="cnbridge_canny_abs")
                    canny_key_txt = gr.Textbox(value="", label="Canny relative key", interactive=False, elem_id="cnbridge_canny_key")

                with gr.Row():
                    pose_abs_txt = gr.Textbox(value="", label="Pose absolute path", interactive=False, elem_id="cnbridge_pose_abs")
                    pose_key_txt = gr.Textbox(value="", label="Pose relative key", interactive=False, elem_id="cnbridge_pose_key")

                with gr.Row():
                    copy_canny_path_btn = gr.Button("📋 Copy canny path")
                    copy_canny_key_btn = gr.Button("📋 Copy canny key")
                    copy_pose_path_btn = gr.Button("📋 Copy pose path")
                    copy_pose_key_btn = gr.Button("📋 Copy pose key")

            # --- Canny cleanup editor (erase text/objects) ---
            with gr.Accordion("🧼 Canny cleanup editor (erase text/objects)", open=False):
                gr.Markdown("Paint a mask (white = erase). This edits the **selected Canny map** after generation. Default erase-to is **black** (recommended).")

                erase_mask = gr.Image(value=None, label="Mask (paint white)", type="pil", tool="sketch")
                with gr.Row():
                    erase_to = gr.Radio(choices=["black", "white"], value="black", label="Erase to")
                    overwrite_map = gr.Checkbox(value=False, label="Overwrite original file")

                with gr.Row():
                    reset_mask_btn = gr.Button("↩️ Reset mask")
                    apply_erase_btn = gr.Button("✅ Apply erase + save", variant="primary")

                erase_status = gr.Textbox(value="", label="Edit status", lines=2, interactive=False)

            with gr.Row():
                unit_canny = gr.Slider(0, 7, value=0, step=1, label="Canny → ControlNet unit")
                unit_pose = gr.Slider(0, 7, value=1, step=1, label="Pose → ControlNet unit")
                auto_cfg = gr.Checkbox(value=True, label="Auto-set ControlNet module/model (best-effort)")

            with gr.Row():
                detect_units_btn = gr.Button('🔎 Auto-detect ControlNet unit count', scale=0)
                gr.Markdown('Counts available ControlNet slots in **txt2img** and **img2img** and updates the unit sliders.')
                t2i_units_box = gr.Textbox(value='0', visible=False)
                i2i_units_box = gr.Textbox(value='0', visible=False)

            with gr.Row():
                send_canny_t2i = gr.Button("➡️ Send Canny (txt2img)")
                send_pose_t2i = gr.Button("➡️ Send Pose (txt2img)")
                send_both_t2i = gr.Button("⚡ Send BOTH (txt2img)", variant="primary")

            with gr.Row():
                send_canny_i2i = gr.Button("➡️ Send Canny (img2img)")
                send_pose_i2i = gr.Button("➡️ Send Pose (img2img)")
                send_both_i2i = gr.Button("⚡ Send BOTH (img2img)", variant="primary")

            with gr.Row():
                send_latest_t2i_btn = gr.Button("🚀 Send LATEST pair (txt2img)", variant="secondary")
                send_latest_i2i_btn = gr.Button("🚀 Send LATEST pair (img2img)", variant="secondary")

            send_status = gr.Textbox(value="", label="Send status", lines=2, interactive=False)


            def _refresh_maps(out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v):
                canny_items = _scan_hint_files(out_dir_v, "canny", orient_v, map_filter_v, sort_mode_v, int(max_items_v))
                pose_items = _scan_hint_files(out_dir_v, "openpose", orient_v, map_filter_v, sort_mode_v, int(max_items_v))

                # Prefer newest *paired* outputs when possible.
                latest_canny, latest_pose = _find_latest_pair(out_dir_v, orient_v)

                canny_val = latest_canny or (canny_items[0] if canny_items else None)
                pose_val = latest_pose or (pose_items[0] if pose_items else None)

                if canny_val and canny_val not in canny_items:
                    canny_items = [canny_val] + canny_items
                if pose_val and pose_val not in pose_items:
                    pose_items = [pose_val] + pose_items

                canny_img, canny_meta = _load_preview(out_dir_v, canny_val) if canny_val else (None, "⚠️ No canny maps found")
                pose_img, pose_meta = _load_preview(out_dir_v, pose_val) if pose_val else (None, "⚠️ No pose maps found")

                canny_abs = _abs_map_path(out_dir_v, canny_val) if canny_val else ""
                canny_key = _rel_key_from_map(canny_val) if canny_val else ""
                pose_abs = _abs_map_path(out_dir_v, pose_val) if pose_val else ""
                pose_key = _rel_key_from_map(pose_val) if pose_val else ""

                blank = _blank_mask_like(out_dir_v, canny_val) if canny_val else None

                return (
                    gr.update(choices=canny_items, value=canny_val),
                    canny_img,
                    canny_meta,
                    canny_abs,
                    canny_key,
                    blank,
                    gr.update(choices=pose_items, value=pose_val),
                    pose_img,
                    pose_meta,
                    pose_abs,
                    pose_key,
                    "",  # erase_status
                )

            def _pick_canny_full(out_dir_v, rel_path_v):
                img, meta = _load_preview(out_dir_v, rel_path_v)
                abs_p = _abs_map_path(out_dir_v, rel_path_v) if rel_path_v else ""
                key = _rel_key_from_map(rel_path_v) if rel_path_v else ""
                blank = _blank_mask_like(out_dir_v, rel_path_v) if rel_path_v else None
                return img, meta, abs_p, key, blank, ""

            def _pick_pose_full(out_dir_v, rel_path_v):
                img, meta = _load_preview(out_dir_v, rel_path_v)
                abs_p = _abs_map_path(out_dir_v, rel_path_v) if rel_path_v else ""
                key = _rel_key_from_map(rel_path_v) if rel_path_v else ""
                return img, meta, abs_p, key

            def _pick_canny(out_dir_v, rel_path_v):
                return _load_preview(out_dir_v, rel_path_v)

            def _pick_pose(out_dir_v, rel_path_v):
                return _load_preview(out_dir_v, rel_path_v)

            def _refresh_sources(in_dir_v, recursive_v, src_filter_v, sort_mode_v, src_max_v):
                items = _scan_source_files(in_dir_v, bool(recursive_v), src_filter_v, sort_mode_v, int(src_max_v))
                val = items[0] if items else None
                img, meta = _load_source_preview(in_dir_v, val) if val else (None, "⚠️ No sources found")
                return gr.update(choices=items, value=val), img, meta


            def _jump_to_pair(in_dir_v, out_dir_v, src_rel_v, orient_v, map_filter_v, sort_mode_v, max_items_v):
                canny_rel, pose_rel = _pair_maps_for_source(out_dir_v, src_rel_v)

                # Re-scan maps using current filters, but ensure matched files are included.
                canny_items = _scan_hint_files(out_dir_v, 'canny', orient_v, map_filter_v, sort_mode_v, int(max_items_v))
                pose_items  = _scan_hint_files(out_dir_v, 'openpose', orient_v, map_filter_v, sort_mode_v, int(max_items_v))

                if canny_rel and canny_rel not in canny_items:
                    canny_items = [canny_rel] + canny_items
                if pose_rel and pose_rel not in pose_items:
                    pose_items = [pose_rel] + pose_items

                canny_img, canny_meta = _load_preview(out_dir_v, canny_rel) if canny_rel else (None, '⚠️ No matching canny output')
                pose_img, pose_meta = _load_preview(out_dir_v, pose_rel) if pose_rel else (None, '⚠️ No matching pose output')

                canny_abs = _abs_map_path(out_dir_v, canny_rel) if canny_rel else ''
                canny_key = _rel_key_from_map(canny_rel) if canny_rel else ''
                pose_abs = _abs_map_path(out_dir_v, pose_rel) if pose_rel else ''
                pose_key = _rel_key_from_map(pose_rel) if pose_rel else ''

                blank = _blank_mask_like(out_dir_v, canny_rel) if canny_rel else None

                return (
                    gr.update(choices=canny_items, value=canny_rel),
                    canny_img,
                    canny_meta,
                    canny_abs,
                    canny_key,
                    blank,
                    gr.update(choices=pose_items, value=pose_rel),
                    pose_img,
                    pose_meta,
                    pose_abs,
                    pose_key,
                    '',  # erase_status
                )

            refresh_maps_btn.click(
                _refresh_maps,
                inputs=[out_dir, orient, map_filter, sort_mode, max_items],
                outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, pose_dd, pose_preview, pose_info, pose_abs_txt, pose_key_txt, erase_status],
            )

            refresh_src_btn.click(
                _refresh_sources,
                inputs=[in_dir, recursive, src_filter, sort_mode, src_max],
                outputs=[src_dd, src_preview, src_info],
            )

            src_dd.change(_load_source_preview, inputs=[in_dir, src_dd], outputs=[src_preview, src_info])

            pair_btn.click(
                _jump_to_pair,
                inputs=[in_dir, out_dir, src_dd, orient, map_filter, sort_mode, max_items],
                outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, pose_dd, pose_preview, pose_info, pose_abs_txt, pose_key_txt, erase_status],
            )

            # If user selects a source, auto-jump (nice UX)
            src_dd.change(
                _jump_to_pair,
                inputs=[in_dir, out_dir, src_dd, orient, map_filter, sort_mode, max_items],
                outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, pose_dd, pose_preview, pose_info, pose_abs_txt, pose_key_txt, erase_status],
            )

            canny_dd.change(_pick_canny_full, inputs=[out_dir, canny_dd], outputs=[canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, erase_status])
            pose_dd.change(_pick_pose_full, inputs=[out_dir, pose_dd], outputs=[pose_preview, pose_info, pose_abs_txt, pose_key_txt])


            # --- Gallery / debug / erase / unit-detect / send-latest bindings ---
            def _refresh_gallery(out_dir_v, orient_v):
                canny_rels = _latest_hint_files(out_dir_v, "canny", orient_v, limit=8)
                pose_rels = _latest_hint_files(out_dir_v, "openpose", orient_v, limit=8)
                canny_items = _gallery_items(out_dir_v, canny_rels)
                pose_items = _gallery_items(out_dir_v, pose_rels)
                return canny_items, canny_rels, pose_items, pose_rels

            def _pick_from_gallery_canny(out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v, rels, evt: gr.SelectData):
                i = int(getattr(evt, "index", 0) or 0)
                rel_path_v = rels[i] if rels and i < len(rels) else None
                canny_items = _scan_hint_files(out_dir_v, "canny", orient_v, map_filter_v, sort_mode_v, int(max_items_v))
                if rel_path_v and rel_path_v not in canny_items:
                    canny_items = [rel_path_v] + canny_items
                img, meta = _load_preview(out_dir_v, rel_path_v) if rel_path_v else (None, "")
                abs_p = _abs_map_path(out_dir_v, rel_path_v) if rel_path_v else ""
                key = _rel_key_from_map(rel_path_v) if rel_path_v else ""
                blank = _blank_mask_like(out_dir_v, rel_path_v) if rel_path_v else None
                return gr.update(choices=canny_items, value=rel_path_v), img, meta, abs_p, key, blank, ""

            def _pick_from_gallery_pose(out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v, rels, evt: gr.SelectData):
                i = int(getattr(evt, "index", 0) or 0)
                rel_path_v = rels[i] if rels and i < len(rels) else None
                pose_items = _scan_hint_files(out_dir_v, "openpose", orient_v, map_filter_v, sort_mode_v, int(max_items_v))
                if rel_path_v and rel_path_v not in pose_items:
                    pose_items = [rel_path_v] + pose_items
                img, meta = _load_preview(out_dir_v, rel_path_v) if rel_path_v else (None, "")
                abs_p = _abs_map_path(out_dir_v, rel_path_v) if rel_path_v else ""
                key = _rel_key_from_map(rel_path_v) if rel_path_v else ""
                return gr.update(choices=pose_items, value=rel_path_v), img, meta, abs_p, key

            gallery_refresh_btn.click(
                _refresh_gallery,
                inputs=[out_dir, orient],
                outputs=[canny_gallery, canny_gallery_state, pose_gallery, pose_gallery_state],
            )

            # Clicking a thumbnail loads it.
            try:
                canny_gallery.select(
                    _pick_from_gallery_canny,
                    inputs=[out_dir, orient, map_filter, sort_mode, max_items, canny_gallery_state],
                    outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, erase_status],
                )
                pose_gallery.select(
                    _pick_from_gallery_pose,
                    inputs=[out_dir, orient, map_filter, sort_mode, max_items, pose_gallery_state],
                    outputs=[pose_dd, pose_preview, pose_info, pose_abs_txt, pose_key_txt],
                )
            except Exception:
                pass

            # Canny cleanup mask
            reset_mask_btn.click(
                fn=lambda out_dir_v, rel_v: (_blank_mask_like(out_dir_v, rel_v) if rel_v else None, ""),
                inputs=[out_dir, canny_dd],
                outputs=[erase_mask, erase_status],
            )

            def _apply_erase(
                out_dir_v,
                orient_v,
                map_filter_v,
                sort_mode_v,
                max_items_v,
                rel_v,
                mask_img_v,
                erase_to_v,
                overwrite_v,
            ):
                if not rel_v:
                    return gr.update(), gr.update(), "❌ Select a canny map first.", "", "", gr.update(), "❌ No canny selected"

                out_img, info, rel_new, blank = _apply_erase_mask(out_dir_v, rel_v, mask_img_v, erase_to_v, overwrite_v)
                # Refresh canny list and make sure new file is selectable
                canny_items = _scan_hint_files(out_dir_v, "canny", orient_v, map_filter_v, sort_mode_v, int(max_items_v))
                if rel_new and rel_new not in canny_items:
                    canny_items = [rel_new] + canny_items

                # Show updated preview/meta
                img2, meta2 = _load_preview(out_dir_v, rel_new) if rel_new else (out_img, info)
                meta_show = f"{info}\n{meta2}" if meta2 else info

                abs_p = _abs_map_path(out_dir_v, rel_new) if rel_new else ""
                key = _rel_key_from_map(rel_new) if rel_new else ""

                return (
                    gr.update(choices=canny_items, value=rel_new),
                    img2,
                    meta_show,
                    abs_p,
                    key,
                    blank,
                    "✅ Saved edited canny map.",
                )

            apply_erase_btn.click(
                _apply_erase,
                inputs=[out_dir, orient, map_filter, sort_mode, max_items, canny_dd, erase_mask, erase_to, overwrite_map],
                outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, erase_status],
            )

            # Copy helpers (clipboard)
            def _copied_msg(label: str):
                return f"📋 Copied {label} to clipboard."

            copy_canny_path_btn.click(
                fn=lambda v: _copied_msg("canny path"),
                inputs=[canny_abs_txt],
                outputs=[send_status],
                _js="(v)=>{ try{navigator.clipboard.writeText(v);}catch(e){} return [v]; }",
            )
            copy_canny_key_btn.click(
                fn=lambda v: _copied_msg("canny key"),
                inputs=[canny_key_txt],
                outputs=[send_status],
                _js="(v)=>{ try{navigator.clipboard.writeText(v);}catch(e){} return [v]; }",
            )
            copy_pose_path_btn.click(
                fn=lambda v: _copied_msg("pose path"),
                inputs=[pose_abs_txt],
                outputs=[send_status],
                _js="(v)=>{ try{navigator.clipboard.writeText(v);}catch(e){} return [v]; }",
            )
            copy_pose_key_btn.click(
                fn=lambda v: _copied_msg("pose key"),
                inputs=[pose_key_txt],
                outputs=[send_status],
                _js="(v)=>{ try{navigator.clipboard.writeText(v);}catch(e){} return [v]; }",
            )

            # Unit count auto-detect
            def _apply_unit_counts(ct2i_v, ci2i_v, u1_v, u2_v):
                try:
                    ct = int(float(ct2i_v or 0))
                except Exception:
                    ct = 0
                try:
                    ci = int(float(ci2i_v or 0))
                except Exception:
                    ci = 0
                n = max(ct, ci)
                if n <= 0:
                    return gr.update(), gr.update(), "⚠️ Could not detect unit count (is ControlNet open / loaded?)"
                mx = max(0, n - 1)
                u1 = min(int(u1_v or 0), mx)
                u2 = min(int(u2_v or 0), mx)
                return (
                    gr.update(maximum=mx, value=u1),
                    gr.update(maximum=mx, value=u2),
                    f"✅ Detected {n} ControlNet slots (unit index 0–{mx}).",
                )

            detect_units_btn.click(
                fn=_apply_unit_counts,
                inputs=[t2i_units_box, i2i_units_box, unit_canny, unit_pose],
                outputs=[unit_canny, unit_pose, send_status],
                _js="(t2i, i2i, u1, u2) => { return [cnbridgeDetectUnitCount('txt2img'), cnbridgeDetectUnitCount('img2img'), u1, u2]; }",
            )

            # Send latest pair (loads newest pair, then JS fires send after a short delay)
            def _load_latest_pair_for_send(tab_name: str, out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v):
                canny_latest, pose_latest = _find_latest_pair(out_dir_v, orient_v)

                canny_items = _scan_hint_files(out_dir_v, "canny", orient_v, map_filter_v, sort_mode_v, int(max_items_v))
                pose_items = _scan_hint_files(out_dir_v, "openpose", orient_v, map_filter_v, sort_mode_v, int(max_items_v))

                canny_val = canny_latest or (canny_items[0] if canny_items else None)
                pose_val = pose_latest or (pose_items[0] if pose_items else None)

                if canny_val and canny_val not in canny_items:
                    canny_items = [canny_val] + canny_items
                if pose_val and pose_val not in pose_items:
                    pose_items = [pose_val] + pose_items

                canny_img, canny_meta = _load_preview(out_dir_v, canny_val) if canny_val else (None, "⚠️ No canny maps")
                pose_img, pose_meta = _load_preview(out_dir_v, pose_val) if pose_val else (None, "⚠️ No pose maps")

                canny_abs = _abs_map_path(out_dir_v, canny_val) if canny_val else ""
                canny_key = _rel_key_from_map(canny_val) if canny_val else ""
                pose_abs = _abs_map_path(out_dir_v, pose_val) if pose_val else ""
                pose_key = _rel_key_from_map(pose_val) if pose_val else ""

                blank = _blank_mask_like(out_dir_v, canny_val) if canny_val else None

                status = f"🚀 Loaded latest pair; sending to {tab_name}…"
                if not (canny_val or pose_val):
                    status = "❌ No maps found to send."

                return (
                    gr.update(choices=canny_items, value=canny_val),
                    canny_img,
                    canny_meta,
                    canny_abs,
                    canny_key,
                    blank,
                    gr.update(choices=pose_items, value=pose_val),
                    pose_img,
                    pose_meta,
                    pose_abs,
                    pose_key,
                    "",
                    status,
                )

            send_latest_t2i_btn.click(
                fn=lambda out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v, u1, u2, auto: _load_latest_pair_for_send('txt2img', out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v),
                inputs=[out_dir, orient, map_filter, sort_mode, max_items, unit_canny, unit_pose, auto_cfg],
                outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, pose_dd, pose_preview, pose_info, pose_abs_txt, pose_key_txt, erase_status, send_status],
                _js="(out_dir, orient, filter, sort, max_items, u1, u2, auto) => { setTimeout(() => { cnbridgeSendBoth('txt2img', u1, u2, auto, 'cnbridge_preview_canny', 'cnbridge_preview_pose'); }, 900); return [out_dir, orient, filter, sort, max_items, u1, u2, auto]; }",
            )

            send_latest_i2i_btn.click(
                fn=lambda out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v, u1, u2, auto: _load_latest_pair_for_send('img2img', out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v),
                inputs=[out_dir, orient, map_filter, sort_mode, max_items, unit_canny, unit_pose, auto_cfg],
                outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, pose_dd, pose_preview, pose_info, pose_abs_txt, pose_key_txt, erase_status, send_status],
                _js="(out_dir, orient, filter, sort, max_items, u1, u2, auto) => { setTimeout(() => { cnbridgeSendBoth('img2img', u1, u2, auto, 'cnbridge_preview_canny', 'cnbridge_preview_pose'); }, 900); return [out_dir, orient, filter, sort, max_items, u1, u2, auto]; }",
            )

            def _status_one(tab: str, kind: str, unit: int, auto: bool):
                a = "auto-config ON" if auto else "auto-config OFF"
                return f"📤 Attempted send: {kind} → {tab} ControlNet unit {int(unit)} ({a})."

            def _status_both(tab: str, u1: int, u2: int, auto: bool):
                a = "auto-config ON" if auto else "auto-config OFF"
                return f"⚡ Attempted send BOTH → {tab} (canny→unit {int(u1)}, pose→unit {int(u2)}) ({a})."

            send_canny_t2i.click(
                fn=lambda unit, auto: _status_one("txt2img", "canny", unit, auto),
                inputs=[unit_canny, auto_cfg],
                outputs=[send_status],
                _js="(unit, auto) => { cnbridgeSendFromPreview('txt2img', unit, 'canny', auto, 'cnbridge_preview_canny'); return [unit, auto]; }",
            )
            send_pose_t2i.click(
                fn=lambda unit, auto: _status_one("txt2img", "openpose", unit, auto),
                inputs=[unit_pose, auto_cfg],
                outputs=[send_status],
                _js="(unit, auto) => { cnbridgeSendFromPreview('txt2img', unit, 'openpose', auto, 'cnbridge_preview_pose'); return [unit, auto]; }",
            )
            send_both_t2i.click(
                fn=lambda u1, u2, auto: _status_both("txt2img", u1, u2, auto),
                inputs=[unit_canny, unit_pose, auto_cfg],
                outputs=[send_status],
                _js="(u1, u2, auto) => { cnbridgeSendBoth('txt2img', u1, u2, auto, 'cnbridge_preview_canny', 'cnbridge_preview_pose'); return [u1, u2, auto]; }",
            )

            send_canny_i2i.click(
                fn=lambda unit, auto: _status_one("img2img", "canny", unit, auto),
                inputs=[unit_canny, auto_cfg],
                outputs=[send_status],
                _js="(unit, auto) => { cnbridgeSendFromPreview('img2img', unit, 'canny', auto, 'cnbridge_preview_canny'); return [unit, auto]; }",
            )
            send_pose_i2i.click(
                fn=lambda unit, auto: _status_one("img2img", "openpose", unit, auto),
                inputs=[unit_pose, auto_cfg],
                outputs=[send_status],
                _js="(unit, auto) => { cnbridgeSendFromPreview('img2img', unit, 'openpose', auto, 'cnbridge_preview_pose'); return [unit, auto]; }",
            )
            send_both_i2i.click(
                fn=lambda u1, u2, auto: _status_both("img2img", u1, u2, auto),
                inputs=[unit_canny, unit_pose, auto_cfg],
                outputs=[send_status],
                _js="(u1, u2, auto) => { cnbridgeSendBoth('img2img', u1, u2, auto, 'cnbridge_preview_canny', 'cnbridge_preview_pose'); return [u1, u2, auto]; }",
            )



        # -------------------------------
        # Run + Log (moved to bottom)
        # -------------------------------
        with gr.Accordion("Run + Log", open=True):
            with gr.Row():
                run_btn = gr.Button("🚀 Run (batch)", variant="primary")
                clear_btn = gr.Button("Clear log", variant="secondary")
            log = gr.Textbox(lines=18, label="Log (live)", value="", interactive=False)
            gen_done = gr.State(0)

        clear_btn.click(lambda: "", outputs=[log])

        run_btn.click(
            run_click,
            inputs=[
                auto_save,
                python_exe, script_path, in_dir, out_dir, mode, detect,
                portrait_size, landscape_size, do_canny, do_openpose,
                canny_low, canny_high, blur,
                canny_invert, canny_thickness, canny_speckle, canny_adaptive, canny_clean_bg, canny_clean_thresh,
                clahe, sharpen, denoise,
                hands, face, device,
                recursive, skip_existing,
                auto_refresh_after_run,
                overlay_png, overlay_png_pos, overlay_png_scale, overlay_png_opacity, overlay_png_offx, overlay_png_offy,
                overlay_text, overlay_text_size, overlay_text_color, overlay_text_outline, overlay_text_outline_color,
                overlay_text_pos, overlay_text_offx, overlay_text_offy,
                gen_done,
            ],
            outputs=[log, gen_done],
        )

        run_single_btn.click(
            run_single_click,
            inputs=[
                auto_save,
                python_exe, script_path, in_dir, out_dir, mode, detect,
                portrait_size, landscape_size, do_canny, do_openpose,
                canny_low, canny_high, blur,
                canny_invert, canny_thickness, canny_speckle, canny_adaptive, canny_clean_bg, canny_clean_thresh,
                clahe, sharpen, denoise,
                hands, face, device,
                recursive, skip_existing,
                auto_refresh_after_run,
                overlay_png, overlay_png_pos, overlay_png_scale, overlay_png_opacity, overlay_png_offx, overlay_png_offy,
                overlay_text, overlay_text_size, overlay_text_color, overlay_text_outline, overlay_text_outline_color,
                overlay_text_pos, overlay_text_offx, overlay_text_offy,
                gen_done,
                single_mode, single_src_dd, single_file,
            ],
            outputs=[log, gen_done, single_status],
        )

        # Auto-refresh maps after generation completes.
        def _on_gen_done(auto_refresh_v, out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v):
            if not auto_refresh_v:
                return (
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                )
            return _refresh_maps(out_dir_v, orient_v, map_filter_v, sort_mode_v, max_items_v)

        gen_done.change(
            _on_gen_done,
            inputs=[auto_refresh_after_run, out_dir, orient, map_filter, sort_mode, max_items],
            outputs=[canny_dd, canny_preview, canny_info, canny_abs_txt, canny_key_txt, erase_mask, pose_dd, pose_preview, pose_info, pose_abs_txt, pose_key_txt, erase_status],
        )

    return [(ui, "CN Hints", "cn_hints_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
