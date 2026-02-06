(() => {
  // CN Hints Bridge → ControlNet sender (best-effort DOM search)

  async function dataUrlToFile(dataUrl, filename) {
    const res = await fetch(dataUrl);
    const blob = await res.blob();
    return new File([blob], filename, { type: blob.type || "image/png" });
  }

  function pickRoot() {
    try {
      return gradioApp();
    } catch (e) {
      return document;
    }
  }

  function findTab(tabName) {
    const root = pickRoot();
    return (
      root.querySelector(`#tab_${tabName}`) ||
      root.querySelector(`[id="tab_${tabName}"]`) ||
      root.querySelector(`[id*="${tabName}"]`)
    );
  }

  function getPreviewSrcById(elemId) {
    const root = pickRoot();
    const host = root.querySelector(`#${elemId}`);
    if (!host) return null;

    // Gradio Image usually renders an <img> tag.
    const img = host.querySelector("img");
    if (img && img.src) return img.src;

    // Fallback: sometimes canvas is used.
    const canvas = host.querySelector("canvas");
    if (canvas && canvas.toDataURL) return canvas.toDataURL("image/png");

    return null;
  }

  function getPreviewSrc() {
    return getPreviewSrcById("cnbridge_preview_img");
  }

  function nearestControlNetUnit(el) {
    if (!el) return null;
    return (
      el.closest("[id*='controlnet']") ||
      el.closest("[class*='controlnet']") ||
      el.closest("[data-testid*='controlnet']") ||
      el.closest("[id*='cn_']") ||
      el.closest("[class*='cn_']") ||
      el.parentElement
    );
  }

  function pickControlNetFileInputs(tabRoot) {
    const all = Array.from(tabRoot.querySelectorAll("input[type='file']"));
    const cn = all.filter((inp) => {
      const unit = nearestControlNetUnit(inp);
      if (!unit) return false;
      const t = (unit.id || "") + " " + (unit.className || "");
      return t.toLowerCase().includes("controlnet") || t.toLowerCase().includes("cn");
    });

    // If heuristic found nothing, at least return all file inputs as a last resort.
    return cn.length ? cn : all;
  }

  function setFileToInput(fileInput, file) {
    const dt = new DataTransfer();
    dt.items.add(file);
    fileInput.files = dt.files;
    fileInput.dispatchEvent(new Event("change", { bubbles: true, composed: true }));
    fileInput.dispatchEvent(new Event("input", { bubbles: true, composed: true }));
  }

  function setSelectFirstMatch(selectEl, predicate) {
    if (!selectEl || !selectEl.options) return false;
    for (const opt of Array.from(selectEl.options)) {
      if (predicate(opt)) {
        selectEl.value = opt.value;
        selectEl.dispatchEvent(new Event("change", { bubbles: true }));
        selectEl.dispatchEvent(new Event("input", { bubbles: true }));
        return true;
      }
    }
    return false;
  }

  function tryAutoConfig(unitRoot, kind) {
    if (!unitRoot) return;
    const k = String(kind || "").toLowerCase();
    const want = k === "openpose" ? "openpose" : "canny";

    const selects = Array.from(unitRoot.querySelectorAll("select"));

    // 1) If we're sending a precomputed hint map, preprocessor should be NONE.
    for (const sel of selects) {
      const texts = Array.from(sel.options).map((o) => (o.textContent || "").trim().toLowerCase());
      const hasNone = texts.some((t) => t === "none" || t.startsWith("none"));
      const hasKind = texts.some((t) => t.includes("canny") || t.includes("openpose") || t.includes("pose"));
      if (hasNone && hasKind) {
        setSelectFirstMatch(sel, (opt) => {
          const t = (opt.textContent || "").trim().toLowerCase();
          return t === "none" || t.startsWith("none");
        });
      }
    }

    // 2) Try to set a model that matches (contains 'canny' / 'openpose').
    //    We avoid aggressively changing selects that don't look like models.
    for (const sel of selects) {
      const texts = Array.from(sel.options).map((o) => (o.textContent || "").trim().toLowerCase());
      const hasWant = texts.some((t) => t.includes(want) || (want === "openpose" && t.includes("pose")));
      const looksLikeModels = texts.some((t) => t.includes("control")) || texts.some((t) => t.includes("xl")) || texts.some((t) => t.includes("sdxl"));
      if (hasWant && looksLikeModels) {
        setSelectFirstMatch(sel, (opt) => {
          const t = (opt.textContent || "").trim().toLowerCase();
          if (want === "openpose") return t.includes("openpose") || t.includes("pose");
          return t.includes("canny");
        });
      }
    }
  }

  // Global function used by python _js callbacks
  window.cnbridgeSendFromPreview = async function (tabName, unitIdx, kind, autoConfig, previewId) {
    try {
      const src = getPreviewSrcById(previewId || "cnbridge_preview_img");
      if (!src) {
        alert("CN Bridge: preview image not found. Select a hint file first.");
        return;
      }

      // Note: Gradio can sometimes render blob: URLs; fetch still works.
      const file = await dataUrlToFile(src, `cnbridge_${kind || "hint"}.png`);

      const tab = findTab(tabName);
      if (!tab) {
        alert(`CN Bridge: could not find tab '${tabName}'.`);
        return;
      }

      const inputs = pickControlNetFileInputs(tab);
      if (!inputs.length) {
        alert("CN Bridge: could not locate any file inputs in the target tab.");
        return;
      }

      const idx = Math.max(0, Math.min(parseInt(unitIdx || 0, 10), inputs.length - 1));
      const target = inputs[idx];

      setFileToInput(target, file);

      if (autoConfig) {
        const unitRoot = nearestControlNetUnit(target);
        tryAutoConfig(unitRoot, kind);
      }
    } catch (err) {
      console.error(err);
      alert("CN Bridge: send failed. Open DevTools console for details.");
    }
  };

  // Backwards-compatible name used by earlier builds
  window.cnbridgeSendToControlNet = async function (tabName, unitIdx, kind, autoConfig) {
    return window.cnbridgeSendFromPreview(tabName, unitIdx, kind, autoConfig, "cnbridge_preview_img");
  };

  window.cnbridgeSendBoth = async function (tabName, cannyUnitIdx, poseUnitIdx, autoConfig, cannyPreviewId, posePreviewId) {
    // Fire sequentially to keep DOM changes predictable.
    await window.cnbridgeSendFromPreview(tabName, cannyUnitIdx, "canny", autoConfig, cannyPreviewId || "cnbridge_preview_canny");
    await window.cnbridgeSendFromPreview(tabName, poseUnitIdx, "openpose", autoConfig, posePreviewId || "cnbridge_preview_pose");
  };

  // Count available ControlNet units by counting file inputs in the tab.
  // Used to auto-size unit sliders.
  window.cnbridgeDetectUnitCount = function (tabName) {
    try {
      const tab = findTab(tabName);
      if (!tab) return 0;
      const inputs = pickControlNetFileInputs(tab);
      return inputs.length || 0;
    } catch (e) {
      return 0;
    }
  };
})();
