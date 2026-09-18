# RelateAnything — Setup Guide

End-to-end setup for the styled Gradio demo with manual box support and the
full predicate preset set. Written against the exact issues a fresh install
tends to hit.

Tested on: Ubuntu 22.04 · Anaconda · Python 3.12 · NVIDIA driver ≥ 550.

---

## 0. TL;DR

If you just want it running:

    conda create -n relsgg python=3.12 -y
    conda activate relsgg
    git clone https://github.com/Maelic/RelateAnything
    cd RelateAnything
    pip install -e ".[dev,deploy,hub,detector,gradio]"

    # download the model
    python -c "from huggingface_hub import snapshot_download; snapshot_download('maelic/relsgg-vits16', local_dir='model')"

    # download the detector
    mkdir -p checkpoints/detectors
    wget -q -P checkpoints/detectors \
      https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt

    # run it
    python deploy/gradio_app_styled.py --device cpu \
      --ckpt model/model.pth \
      --det checkpoints/detectors/yoloe-11s-seg-pf.pt

Open `http://127.0.0.1:7860`.

---

## 1. Prerequisites

- **Python 3.12 or newer.** The project declares `requires-python = ">=3.12"`
  in `pyproject.toml`. Python 3.10 or 3.11 will fail at install.
- **Conda or Miniconda** (recommended). Or `python3 -m venv` if you prefer.
  Do NOT install into a system Python or into Anaconda's `base` environment —
  see §4 for why.
- **A GPU is optional.** The demo runs on CPU. An NVIDIA GPU with ≥4 GB VRAM
  is faster, and VRAM ≥8 GB allows real-time-ish webcam streaming. On CPU,
  use the Image tab, not Webcam.
- **~3 GB free disk** for the environment plus models.

Check what you have:

    python3 --version           # need >= 3.12
    nvidia-smi                  # optional
    df -h ~                     # need a few GB free

---

## 2. Create the environment

Use a **dedicated conda environment**. Do not install into `base`.

    conda create -n relsgg python=3.12 -y
    conda activate relsgg
    which python                # MUST end in .../envs/relsgg/bin/python
    python -V                   # MUST print Python 3.12.x

If `which python` doesn't show the relsgg env path, `conda activate` didn't
take. Run:

    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate relsgg

...or use the env's Python directly:

    ~/anaconda3/envs/relsgg/bin/python ...

---

## 3. Clone and install

    git clone https://github.com/Maelic/RelateAnything
    cd RelateAnything
    pip install -e ".[dev,deploy,hub,detector,gradio]"

**Why all four extras:**

| extra | what it provides |
|---|---|
| `dev` | pytest, for the sanity test below |
| `deploy` | ONNX Runtime, ONNX, full OpenCV (headless shadows it) |
| `hub` | `huggingface_hub`, to download the model |
| `detector` | `ultralytics` — pulls AGPL-3.0 code; used by the demo's detector |
| `gradio` | the web UI |

**Verify the install:**

    python -m pytest tests/ -q

Expect `73 passed`. If it errors with `ModuleNotFoundError: No module named 'torch'`,
you're running pytest from the wrong Python — see §4.

---

## 4. The #1 mistake: installing into the wrong Python

The single most common failure mode. Symptoms:

- `pytest: command not found` (it's installed but not on PATH)
- `ModuleNotFoundError: No module named 'torch'` even though you installed it
- Traceback shows `/usr/lib/python3.10/` or `.../anaconda3/bin/python`

**Cause:** `pip install` ran against one interpreter, `python -m pytest` against
another. `which python` and `which pip` must both point at the same env.

**Diagnose:**

    python -c "import sys; print(sys.executable)"
    which pip
    python -m pytest tests/ -q        # always use `python -m`, not bare `pytest`

**Fix:** activate the conda env as in §2, then `pip install` inside it.

**Also do not install into Anaconda `base`.** If you do, pip will upgrade
numpy to 2.5 and break `numba`, `gensim`, `streamlit`, `contourpy`, and
anything else living in base. Create the env instead.

---

## 5. Download the model

The model ships on the Hugging Face Hub, ungated. No login needed.

    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('maelic/relsgg-vits16', local_dir='model')"

~250 MB. Lands in `./model/` with `model.pth`, `text_student.pt`, the CLIP
tokenizer, `calibration.json`, and `predicate_embeddings.npz`.

**Model variants:**

| id | params | notes |
|---|---|---|
| `maelic/relsgg-vits16` | 46.1 M | smallest, fastest — good for CPU and 4 GB GPUs |
| `maelic/relsgg-vits16plus` | 53.2 M | **recommended** — matches ViT-B at half the params |
| `maelic/relsgg-vitb16` | 113.8 M | largest — needs ≥8 GB |

If you need a different one, substitute the id in the snapshot command.

---

## 6. Download the detector

YOLOE ships in two flavours. Choose based on your goal:

**Prompt-free (recommended default)** — 4,585 built-in classes, detects
anything without configuration:

    mkdir -p checkpoints/detectors
    wget -q -P checkpoints/detectors \
      https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt

**Text-prompt** — empty vocabulary, detects ONLY the classes you list in the
UI's "object classes" box:

    wget -q -P checkpoints/detectors \
      https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt

Both work. The demo auto-detects `-pf` in the filename and changes its
startup behaviour accordingly.

**License note:** ultralytics is AGPL-3.0. The detector weights are not
redistributed with RelateAnything; you download them yourself.

---

## 7. Run the demo

**CPU** (works on any machine):

    python deploy/gradio_app_styled.py --device cpu \
      --ckpt model/model.pth \
      --det checkpoints/detectors/yoloe-11s-seg-pf.pt

**GPU:**

    python deploy/gradio_app_styled.py --device cuda \
      --ckpt model/model.pth \
      --det checkpoints/detectors/yoloe-11s-seg-pf.pt

**If your GPU has ≤4 GB VRAM**, add:

    --max_objects 8 --final_budget 32

**Tighter still (4 GB + webcam)**:

    --max_objects 6 --final_budget 24 --no_overlap

Open the URL it prints. **Use `http://127.0.0.1:7860`**, not `0.0.0.0:7860`.
`0.0.0.0` is a bind address; browsers treat it as insecure and disable webcam
access.

Startup takes 30–90 s on CPU (model load + first forward pass). Not a hang.

---

## 8. Using the UI

Left panel: the image, the rendered overlay, and three prediction tabs
(`both` / `spatial` / `semantic`).

Right panel, top to bottom:

1. **LABELS** — show detected objects by detector class name, or by colour.
2. **BOXES** — draw boxes manually, or paste coordinates.
3. **PICTURE** — six sample images from the repository.
4. **VOCABULARY** — object classes and predicates, editable live.
5. **INFERENCE** — detector confidence, max triplets, relation threshold.

**Vocabularies re-parameterize live.** Type new strings, hit Apply. No restart,
no retraining.

**Predicate presets:** `All 263`, `Default`, `Spatial`, `Semantic`, `Minimal`.
Clicking one fills the textbox and applies immediately.

**Manual box mode** — bypass the detector entirely:

- Tick "draw on image", then drag rectangles on the picture. Double-click to
  clear.
- Or paste `x1,y1,x2,y2` per line into "my boxes". Values ≤1.5 are read as
  normalised fractions; otherwise pixels.
- Leave "my boxes" empty to fall back to the detector.

**Spatial vs semantic split.** Every relation is coloured by type: orange for
layout (`on`, `behind`, `above`), green for interaction (`holding`, `riding`,
`looking at`). A pair can appear in both — that's the two-graph decode.

---

## 9. Sanity check without the UI

If the UI misbehaves, run the model headless to isolate the problem:

    python - <<'PY'
    from PIL import Image
    from relsgg import RelateAnything

    m = RelateAnything.from_pretrained("maelic/relsgg-vits16", device="cpu")
    img = Image.open("assets/reel/images/catlaptop.jpg").convert("RGB")
    W, H = img.size
    boxes = [[0.30*W, 0.05*H, 0.85*W, 0.55*H],
             [0.10*W, 0.40*H, 0.95*W, 0.95*H]]
    for t in m.predict(img, boxes, topk=10, max_boxes=16):
        print(t)
    PY

Expect something like `(cat) --sitting in [...]--> (laptop)`. If this works
but the UI doesn't, the issue is in the UI layer.

---

## 10. Common problems

### `pytest: command not found`

Use `python -m pytest tests/ -q`. Not on PATH.

### `ModuleNotFoundError: No module named 'torch'` in pytest

Wrong interpreter. See §4.

### `IndexError: list index out of range` at
`deploy/pipeline.py:...preds[int(arg[j])]`

Vocabulary mismatch between the pipeline's cached `predicates` list and the
model's actual `W`. Applied patch: pipeline reads
`self.model.vocab_head.pred_names` instead of `self.ra.predicates`. If you see
this, the pipeline file in your checkout predates the fix — check that
`deploy/pipeline.py` contains the string `vocab_head, 'pred_names'`.

### `torch.cuda.OutOfMemoryError`

Reduce shapes:

    --max_objects 8 --final_budget 32

Or run `--device cpu`.

### Webcam doesn't work

1. Use `http://127.0.0.1:7860` (not `0.0.0.0:7860`).
2. Grant camera permission when the browser asks.
3. Close any other app using the camera (Zoom, Teams, another tab).
4. On CPU, webcam is essentially unusable — 1–2 s per frame. Use the Image tab.

### `SharedArrayBuffer is not defined`

The service worker needs to register before it takes effect. Reload the page
once. (Only relevant if you're running the client-side browser demo, not the
Gradio one.)

### Gradio warning about `theme` / `css` in the constructor

Cosmetic. Gradio 6 moved those to `launch()`. The styled app already passes
them correctly.

### `FutureWarning: torch.jit.load is deprecated`

From YOLOE's internal loading. Cosmetic. Ignore.

### Demo loads but nothing shows up when I click a sample

Check the terminal for a Python traceback. The most common cause is a
missing or malformed manual-boxes input — check that the "my boxes" textbox
is empty if you want the detector path.

---

## 11. Running on Colab or a cloud GPU

Both work. Colab's free T4 (16 GB) runs the whole demo comfortably. For a
public URL, pass `--share` to Gradio:

    python deploy/gradio_app_styled.py --device cuda --share \
      --ckpt model/model.pth --det checkpoints/detectors/yoloe-11s-seg-pf.pt

That prints a `https://xxxx.gradio.live` URL reachable from anywhere.

Colab-specific:

    !git clone https://github.com/Maelic/RelateAnything
    %cd RelateAnything
    !pip install -e ".[dev,deploy,hub,detector,gradio]" --quiet
    !python -c "from huggingface_hub import snapshot_download; \
                snapshot_download('maelic/relsgg-vits16', local_dir='model')"
    !mkdir -p checkpoints/detectors && \
     wget -q -P checkpoints/detectors \
       https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg-pf.pt

Then run the demo in a subprocess so it doesn't block the cell:

    import subprocess, threading
    threading.Thread(target=lambda: subprocess.run([
        "python", "deploy/gradio_app_styled.py", "--device", "cuda", "--share",
        "--ckpt", "model/model.pth",
        "--det", "checkpoints/detectors/yoloe-11s-seg-pf.pt",
    ]), daemon=True).start()

Watch the cell output for the `gradio.live` URL. The session dies when you
close the Colab tab.

For Modal or another serverless GPU provider, wrap the same command with the
provider's web endpoint decorator. The Python code inside doesn't change.

---

## 12. What's in the repo vs what you download

| | in the repo | downloaded separately |
|---|---|---|
| Apache-2.0 code | ✅ | |
| Model weights | | ✅ Hugging Face (`maelic/relsgg-*`) |
| Detector weights | | ✅ ultralytics (AGPL-3.0) |
| Sample images | ✅ `assets/reel/images/` | |
| Test suite | ✅ `tests/` | |
| Evaluation packs | | ✅ `maelic/OV-SGG-Bench` (only if benchmarking) |

---

## 13. Reference: full arg list

Run any entry point with `--help`:

    python deploy/gradio_app_styled.py --help

Common flags:

    --ckpt PATH          model .pth file
    --det PATH           detector .pt file
    --device {cpu,cuda}  compute device
    --max_objects N      boxes per image (default 16)
    --final_budget N     relation pairs kept (default 64)
    --no_overlap         disable detector ∥ backbone overlap (CUDA only)
    --port N             port (default 7860)
    --share              Gradio public tunnel URL

---

## 14. Getting help

- `docs/pitfalls.md` in the repo — every entry there has produced a plausible
  wrong result for someone; short and worth reading before trusting any output.
- `docs/quickstart.md`, `docs/deployment.md`, `docs/evaluation.md` — the
  upstream project's own documentation.
- The GitHub issues page for bugs that are clearly in the library, not your
  setup.
