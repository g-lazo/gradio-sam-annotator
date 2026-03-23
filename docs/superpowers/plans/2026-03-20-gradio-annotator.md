# Gradio Annotator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Gradio web UI that runs on RunPod, lets the user click on objects in images to trigger SAM3 segmentation, assigns a class via radio buttons, and exports a Label Studio-compatible annotation pack.

**Architecture:** Two Python classes (`SAMBackend`, `AnnotationSession`) plus a Gradio UI layer, all in a single script `files/annotator.py`. `AnnotationSession` is pure logic (no GPU, no UI), making it fully unit-testable. `SAMBackend` wraps SAM3 via `transformers`. The Gradio layer is thin — it only wires events to the two classes and calls a render helper.

**Tech Stack:** Python 3.10+, Gradio, transformers (Sam3TrackerModel/Processor), OpenCV (headless), PyTorch, Pillow, NumPy.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `files/annotator.py` | Create | SAMBackend + AnnotationSession + Gradio UI + CLI |
| `files/tests/__init__.py` | Create | Makes tests a package |
| `files/tests/test_annotation_session.py` | Create | Unit tests for AnnotationSession (loading, navigation, export) |
| `files/tests/test_sam_backend.py` | Create | Unit tests for SAMBackend (mocked model) |
| `files/requirements_pipeline.txt` | Modify | Add `gradio>=4.0.0` |

---

### Task 1: Add gradio to requirements and create test scaffolding

**Files:**
- Modify: `files/requirements_pipeline.txt`
- Create: `files/tests/__init__.py`
- Create: `files/tests/test_annotation_session.py` (skeleton)

- [ ] **Step 1: Add gradio to requirements**

Open `files/requirements_pipeline.txt` and add at the end:
```
gradio>=4.0.0
```

- [ ] **Step 2: Create test package**

Create `files/tests/__init__.py` as an empty file.

- [ ] **Step 3: Create test file skeleton**

Create `files/tests/test_annotation_session.py`:
```python
import pytest
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image


def make_test_image(path: Path, w: int = 100, h: int = 80):
    """Create a small solid-color JPEG for testing."""
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    img.save(str(path))
    return w, h
```

- [ ] **Step 4: Commit**

```bash
git add files/requirements_pipeline.txt files/tests/
git commit -m "chore: add gradio dependency and test scaffolding"
```

---

### Task 2: AnnotationSession — image loading and navigation

**Files:**
- Create: `files/annotator.py` (AnnotationSession class only for now)
- Modify: `files/tests/test_annotation_session.py`

- [ ] **Step 1: Write failing tests for image loading and navigation**

Add to `files/tests/test_annotation_session.py`:
```python
# Import will fail until annotator.py exists — that's expected
from annotator import AnnotationSession


@pytest.fixture
def image_dir(tmp_path):
    for name in ["a.jpg", "b.jpg", "c.jpg"]:
        make_test_image(tmp_path / name)
    return tmp_path


def test_session_loads_sorted_image_paths(image_dir):
    session = AnnotationSession(str(image_dir))
    assert len(session.image_paths) == 3
    # Should be sorted
    names = [Path(p).name for p in session.image_paths]
    assert names == sorted(names)


def test_session_ignores_non_image_files(tmp_path):
    make_test_image(tmp_path / "img.jpg")
    (tmp_path / "notes.txt").write_text("hello")
    session = AnnotationSession(str(tmp_path))
    assert len(session.image_paths) == 1


def test_current_image_starts_at_first(image_dir):
    session = AnnotationSession(str(image_dir))
    assert session.index == 0
    assert session.current_image_path() == session.image_paths[0]


def test_next_advances_index(image_dir):
    session = AnnotationSession(str(image_dir))
    session.next_image()
    assert session.index == 1


def test_next_reaches_sentinel_but_not_beyond(image_dir):
    # next_image() allows one-past-last (sentinel); a second call does not advance further
    session = AnnotationSession(str(image_dir))
    session.next_image()
    session.next_image()
    session.next_image()  # reaches sentinel (index == 3 on a 3-image session)
    assert session.is_done()
    assert session.index == 3
    session.next_image()  # extra call — should not advance past sentinel
    assert session.index == 3


def test_prev_goes_back(image_dir):
    session = AnnotationSession(str(image_dir))
    session.next_image()
    session.prev_image()
    assert session.index == 0


def test_prev_does_not_go_before_first(image_dir):
    session = AnnotationSession(str(image_dir))
    session.prev_image()
    assert session.index == 0


def test_is_done_when_past_last(image_dir):
    session = AnnotationSession(str(image_dir))
    assert not session.is_done()
    session.next_image()
    session.next_image()
    session.next_image()
    assert session.is_done()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace && python -m pytest files/tests/test_annotation_session.py -v 2>&1 | head -30
```
Expected: `ModuleNotFoundError: No module named 'annotator'`

- [ ] **Step 3: Implement AnnotationSession — loading and navigation**

Create `files/annotator.py` with just the AnnotationSession class:
```python
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed color palette per class index (BGR for OpenCV)
COLORS = [
    (0, 200, 255),   # orange
    (0, 255, 100),   # green
    (255, 80, 80),   # blue
    (80, 80, 255),   # red
    (255, 0, 200),   # purple
]


class AnnotationSession:

    def __init__(self, image_dir: str):
        self.image_paths = sorted([
            str(p) for p in Path(image_dir).iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])
        self.index = 0
        # path -> list of (class_name, bbox_xyxy, img_w, img_h)
        self._annotations: dict[str, list] = {}

    def current_image_path(self) -> str:
        return self.image_paths[self.index]

    def is_done(self) -> bool:
        return self.index >= len(self.image_paths)

    def next_image(self):
        # Allows advancing one past the last image (sentinel = done)
        if self.index < len(self.image_paths):
            self.index += 1

    def prev_image(self):
        if self.index > 0:
            self.index -= 1

    def add_annotation(self, class_name: str, bbox_xyxy: list, img_w: int, img_h: int):
        path = self.current_image_path()
        if path not in self._annotations:
            self._annotations[path] = []
        self._annotations[path].append((class_name, bbox_xyxy, img_w, img_h))

    def undo_last(self):
        path = self.current_image_path()
        if self._annotations.get(path):
            self._annotations[path].pop()

    def get_annotations(self, image_path: str) -> list:
        return self._annotations.get(image_path, [])

    def export(self, output_dir: str):
        pass  # implemented in Task 3
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /workspace && python -m pytest files/tests/test_annotation_session.py -v -k "not export"
```
Expected: all navigation tests PASS

- [ ] **Step 5: Commit**

```bash
git add files/annotator.py files/tests/test_annotation_session.py
git commit -m "feat: add AnnotationSession with image loading and navigation"
```

---

### Task 3: AnnotationSession — add_annotation, undo, and export

**Files:**
- Modify: `files/tests/test_annotation_session.py`
- Modify: `files/annotator.py` (implement `export`)

- [ ] **Step 1: Write failing tests for annotations and export**

Add to `files/tests/test_annotation_session.py`:
```python
def test_add_annotation_stores_entry(image_dir):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    anns = session.get_annotations(session.current_image_path())
    assert len(anns) == 1
    assert anns[0][0] == "coca-cola"
    assert anns[0][1] == [10.0, 20.0, 50.0, 60.0]


def test_undo_removes_last_annotation(image_dir):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    session.add_annotation("monster", [5.0, 5.0, 30.0, 40.0], 100, 80)
    session.undo_last()
    assert len(session.get_annotations(session.current_image_path())) == 1


def test_undo_on_empty_does_nothing(image_dir):
    session = AnnotationSession(str(image_dir))
    session.undo_last()  # should not raise
    assert session.get_annotations(session.current_image_path()) == []


def test_export_creates_label_studio_pack(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    # Annotate image 0: one bbox at absolute pixels [10, 20, 50, 60] in 100x80 image
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)

    session.export(str(tmp_path))

    pack_dir = tmp_path / "label_studio_pack"
    assert (pack_dir / "tasks.json").exists()
    assert (pack_dir / "tasks_upload.json").exists()
    assert (pack_dir / "images").is_dir()

    tasks = json.loads((pack_dir / "tasks.json").read_text())
    assert len(tasks) == 3  # all 3 images included

    # Find the task for image 0 (has annotation)
    annotated = next(t for t in tasks if Path(t["data"]["image"]).name == "a.jpg")
    result = annotated["predictions"][0]["result"][0]

    assert result["type"] == "rectanglelabels"
    assert result["from_name"] == "label"
    assert result["to_name"] == "image"
    assert result["value"]["rectanglelabels"] == ["coca-cola"]
    assert result["score"] == 1.0

    # Check percentage conversion: x1=10, w=100 → x%=10.0
    assert abs(result["value"]["x"] - 10.0) < 0.01
    assert abs(result["value"]["y"] - 25.0) < 0.01   # 20/80*100
    assert abs(result["value"]["width"] - 40.0) < 0.01  # (50-10)/100*100
    assert abs(result["value"]["height"] - 50.0) < 0.01  # (60-20)/80*100


def test_export_images_are_copied(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.export(str(tmp_path))
    images_dir = tmp_path / "label_studio_pack" / "images"
    copied = list(images_dir.iterdir())
    assert len(copied) == 3


def test_export_unannotated_image_has_empty_result(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    # No annotations added
    session.export(str(tmp_path))
    tasks = json.loads((tmp_path / "label_studio_pack" / "tasks.json").read_text())
    for task in tasks:
        assert task["predictions"][0]["result"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace && python -m pytest files/tests/test_annotation_session.py -v -k "export or undo or add_annotation"
```
Expected: export tests FAIL (method is a stub), undo/add tests should PASS.

- [ ] **Step 3: Implement export()**

Replace `pass` in `AnnotationSession.export()` with:
```python
def export(self, output_dir: str):
    pack_dir = Path(output_dir) / "label_studio_pack"
    images_dir = pack_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    tasks_local = []
    tasks_upload = []

    for image_path in self.image_paths:
        src = Path(image_path)
        dst = images_dir / src.name
        if src != dst:
            shutil.copy2(str(src), str(dst))

        anns = self._annotations.get(image_path, [])
        result = []
        for idx, (class_name, bbox, img_w, img_h) in enumerate(anns):
            x1, y1, x2, y2 = bbox
            result.append({
                "id": f"{src.stem}_{idx}",
                "type": "rectanglelabels",
                "from_name": "label",
                "to_name": "image",
                "value": {
                    "x": (x1 / img_w) * 100,
                    "y": (y1 / img_h) * 100,
                    "width": ((x2 - x1) / img_w) * 100,
                    "height": ((y2 - y1) / img_h) * 100,
                    "rectanglelabels": [class_name],
                },
                "score": 1.0,
            })

        prediction = [{"result": result}]

        tasks_local.append({
            "data": {"image": f"/data/local-files/?d=images/{src.name}"},
            "predictions": prediction,
        })
        tasks_upload.append({
            "data": {"image": src.name},
            "predictions": prediction,
        })

    (pack_dir / "tasks.json").write_text(json.dumps(tasks_local, indent=2))
    (pack_dir / "tasks_upload.json").write_text(json.dumps(tasks_upload, indent=2))
    print(f"Pack exportado en: {pack_dir}")
    print(f"  {len(self.image_paths)} imagenes, {sum(len(v) for v in self._annotations.values())} anotaciones")
```

- [ ] **Step 4: Run all AnnotationSession tests**

```bash
cd /workspace && python -m pytest files/tests/test_annotation_session.py -v
```
Expected: all tests PASS

- [ ] **Step 5: Commit**

```bash
git add files/annotator.py files/tests/test_annotation_session.py
git commit -m "feat: implement AnnotationSession export to Label Studio format"
```

---

### Task 4: SAMBackend — point-prompted segmentation

**Files:**
- Modify: `files/annotator.py` (add SAMBackend class)
- Create: `files/tests/test_sam_backend.py`

- [ ] **Step 1: Write failing tests with mocked model**

Create `files/tests/test_sam_backend.py`:
```python
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from PIL import Image


def make_pil_image(w=200, h=150):
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))


@pytest.fixture
def mock_sam(monkeypatch):
    """Patch Sam3TrackerModel and Sam3TrackerProcessor so no GPU needed."""
    mock_processor = MagicMock()
    mock_model = MagicMock()

    # Simulate processor output as a real dict-like object so __getitem__ works
    mock_inputs = {
        "original_sizes": torch.tensor([[150, 200]]),
    }
    mock_inputs_obj = MagicMock()
    # MagicMock dunder methods must be set on the class, not the instance;
    # use side_effect on the already-wired magic __getitem__ instead:
    mock_inputs_obj.__getitem__ = MagicMock(side_effect=lambda key: mock_inputs[key])
    mock_inputs_obj.to = MagicMock(return_value=mock_inputs_obj)
    mock_processor.return_value = mock_inputs_obj

    # Simulate model output
    mock_outputs = MagicMock()
    mock_outputs.pred_masks = torch.zeros(1, 1, 1, 150, 200)
    mock_model.return_value = mock_outputs

    monkeypatch.setattr("annotator.Sam3TrackerModel", MagicMock(
        from_pretrained=MagicMock(return_value=mock_model)
    ))
    monkeypatch.setattr("annotator.Sam3TrackerProcessor", MagicMock(
        from_pretrained=MagicMock(return_value=mock_processor)
    ))

    return mock_processor, mock_model, mock_outputs


def test_segment_returns_none_for_empty_mask(mock_sam):
    from annotator import SAMBackend
    mock_processor, mock_model, mock_outputs = mock_sam

    # Empty mask (all zeros) → post_process_masks returns all-False mask
    h, w = 150, 200
    empty_mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
    mock_processor.post_process_masks = MagicMock(return_value=[empty_mask])

    backend = SAMBackend("facebook/sam3", device="cpu")
    result = backend.segment(make_pil_image(w, h), (100, 75))
    assert result is None


def test_segment_returns_bbox_from_mask(mock_sam):
    from annotator import SAMBackend
    mock_processor, mock_model, mock_outputs = mock_sam

    h, w = 150, 200
    mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
    # Set a rectangular region: rows 20-60, cols 30-90
    mask[0, 0, 20:61, 30:91] = True
    mock_processor.post_process_masks = MagicMock(return_value=[mask])

    backend = SAMBackend("facebook/sam3", device="cpu")
    bbox = backend.segment(make_pil_image(w, h), (60, 40))

    assert bbox is not None
    x1, y1, x2, y2 = bbox
    assert x1 == 30.0
    assert y1 == 20.0
    assert x2 == 90.0
    assert y2 == 60.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /workspace && python -m pytest files/tests/test_sam_backend.py -v
```
Expected: `ImportError` — SAMBackend doesn't exist yet.

- [ ] **Step 3: Implement SAMBackend**

Add to `files/annotator.py` (before AnnotationSession class):
```python
from transformers import Sam3TrackerModel, Sam3TrackerProcessor


class SAMBackend:

    def __init__(self, model_name: str = "facebook/sam3", device: str = "cuda"):
        self.device = device
        self.model = Sam3TrackerModel.from_pretrained(model_name).to(device)
        self.processor = Sam3TrackerProcessor.from_pretrained(model_name)
        self.model.eval()

    def segment(self, pil_image, point_xy: tuple) -> list | None:
        import torch
        x, y = int(point_xy[0]), int(point_xy[1])

        inputs = self.processor(
            images=pil_image,
            input_points=[[[x, y]]],
            input_labels=[[1]],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)

        masks = self.processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"],
        )[0]  # [1, 1, H, W]

        mask = masks[0, 0].numpy().astype(bool)
        coords = np.where(mask)

        if len(coords[0]) == 0:
            return None

        return [
            float(np.min(coords[1])),  # x1
            float(np.min(coords[0])),  # y1
            float(np.max(coords[1])),  # x2
            float(np.max(coords[0])),  # y2
        ]
```

- [ ] **Step 4: Run SAMBackend tests**

```bash
cd /workspace && python -m pytest files/tests/test_sam_backend.py -v
```
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add files/annotator.py files/tests/test_sam_backend.py
git commit -m "feat: add SAMBackend with point-prompted SAM3 segmentation"
```

---

### Task 5: Image rendering helper

**Files:**
- Modify: `files/annotator.py` (add `render_image` function)

- [ ] **Step 1: Implement render_image**

Add to `files/annotator.py` after the imports:
```python
def render_image(image_path: str, annotations: list, classes: list[str]) -> np.ndarray:
    """
    Draw bboxes and class labels on image copy.
    annotations: list of (class_name, bbox_xyxy, img_w, img_h)
    Returns BGR numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "No se pudo leer la imagen", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img

    for class_name, bbox, _, _ in annotations:
        x1, y1, x2, y2 = [int(c) for c in bbox]
        class_idx = classes.index(class_name) if class_name in classes else 0
        color = COLORS[class_idx % len(COLORS)]

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img
```

- [ ] **Step 2: Verify render_image works manually**

```bash
cd /workspace && python3 -c "
from annotator import render_image
import cv2, glob
imgs = glob.glob('fotos/train/*.jpg')
if imgs:
    result = render_image(imgs[0], [], ['coca-cola', 'monster'])
    print('Shape:', result.shape, 'dtype:', result.dtype)
    print('OK')
"
```
Expected: prints shape and `OK` with no errors.

- [ ] **Step 3: Commit**

```bash
git add files/annotator.py
git commit -m "feat: add render_image helper for bbox overlay"
```

---

### Task 6: Gradio UI — click handler and controls

**Files:**
- Modify: `files/annotator.py` (add `build_ui` function and `main`)

- [ ] **Step 1: Implement build_ui and main**

Add to the bottom of `files/annotator.py`:
```python
import gradio as gr
import argparse


def build_ui(session: AnnotationSession, sam: SAMBackend, classes: list[str]) -> gr.Blocks:

    def get_status():
        total = len(session.image_paths)
        idx = session.index
        anns = session.get_annotations(session.current_image_path())
        return f"Imagen {idx + 1} / {total}  |  Bboxes: {len(anns)}"

    def get_rendered():
        path = session.current_image_path()
        anns = session.get_annotations(path)
        bgr = render_image(path, anns, classes)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def on_click(image_rgb, evt: gr.SelectData, selected_class):
        if selected_class is None:
            return get_rendered(), "Selecciona una clase primero"

        x, y = evt.index[0], evt.index[1]
        pil_image = PILImage.fromarray(image_rgb)
        w, h = pil_image.size

        bbox = sam.segment(pil_image, (x, y))
        if bbox is None:
            return get_rendered(), "No se detecto objeto, intenta otro punto"

        session.add_annotation(selected_class, bbox, w, h)
        return get_rendered(), get_status()

    def on_undo():
        session.undo_last()
        return get_rendered(), get_status()

    def on_next():
        session.next_image()
        return get_rendered(), get_status()

    def on_prev():
        session.prev_image()
        return get_rendered(), get_status()

    def on_save(output_dir):
        session.export(output_dir)
        return get_rendered(), f"Guardado en {output_dir}/label_studio_pack"

    with gr.Blocks(title="Anotador") as demo:
        gr.Markdown("## Anotador — Click en objeto, selecciona clase, navega con los botones")

        with gr.Row():
            with gr.Column(scale=3):
                image_display = gr.Image(
                    value=get_rendered(),
                    label="Imagen",
                    interactive=True,
                )
            with gr.Column(scale=1):
                class_radio = gr.Radio(choices=classes, label="Clase a anotar", value=classes[0])
                status_label = gr.Textbox(value=get_status(), label="Estado", interactive=False)
                undo_btn = gr.Button("Deshacer ultimo")
                prev_btn = gr.Button("← Anterior")
                next_btn = gr.Button("Siguiente →")
                output_dir_input = gr.Textbox(value="./output", label="Directorio de salida")
                save_btn = gr.Button("Guardar y salir", variant="primary")

        image_display.select(on_click, inputs=[image_display, class_radio],
                              outputs=[image_display, status_label])
        undo_btn.click(on_undo, outputs=[image_display, status_label])
        prev_btn.click(on_prev, outputs=[image_display, status_label])
        next_btn.click(on_next, outputs=[image_display, status_label])
        save_btn.click(on_save, inputs=[output_dir_input],
                       outputs=[image_display, status_label])

    return demo


def main():
    parser = argparse.ArgumentParser(description="Anotador Gradio + SAM3")
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--classes", default="object", help="Clases separadas por coma")
    parser.add_argument("--sam_model", default="facebook/sam3")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")]

    print("Cargando SAM3...")
    sam = SAMBackend(model_name=args.sam_model, device=args.device)

    print(f"Cargando imagenes de {args.image_dir}...")
    session = AnnotationSession(args.image_dir)
    print(f"  {len(session.image_paths)} imagenes encontradas")

    demo = build_ui(session, sam, classes)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script can be imported without errors**

```bash
cd /workspace && python3 -c "import importlib.util; spec = importlib.util.spec_from_file_location('annotator', 'files/annotator.py'); print('Import OK')"
```
Expected: `Import OK` (may warn about missing gradio/torch if not installed, but no SyntaxError)

- [ ] **Step 3: Run all tests to confirm nothing broke**

```bash
cd /workspace && python -m pytest files/tests/ -v
```
Expected: all tests PASS

- [ ] **Step 4: Commit**

```bash
git add files/annotator.py
git commit -m "feat: add Gradio UI, click handler, navigation controls, and CLI"
```

---

### Task 7: End-to-end smoke test on RunPod

This task is manual — no unit tests for the Gradio UI itself.

- [ ] **Step 1: Install gradio on the pod**

```bash
pip install "gradio>=4.0.0"
```

- [ ] **Step 2: Launch the annotator**

```bash
cd /workspace
python3 files/annotator.py \
  --image_dir ./fotos/train \
  --output_dir ./output \
  --classes coca-cola,monster \
  --sam_model facebook/sam3 \
  --device cuda \
  --port 7860
```

- [ ] **Step 3: Open the UI**

In RunPod's interface, go to "Connect" → open port 7860. Or use the public URL shown in the terminal output.

- [ ] **Step 4: Annotate a few images**

- Select "coca-cola" in the radio button
- Click on a Coca Cola can in the image
- Verify: bbox appears overlaid on the image
- Click "Siguiente →" — image advances
- Click "Guardar y salir"

- [ ] **Step 5: Verify output**

```bash
ls -la output/label_studio_pack/
cat output/label_studio_pack/tasks.json | python3 -m json.tool | head -40
```
Expected: valid JSON with `rectanglelabels`, `from_name`, `to_name` fields.

- [ ] **Step 6: Load into Label Studio**

Import `tasks_upload.json` into Label Studio (or `tasks.json` with local storage). Verify annotations appear with class labels.

- [ ] **Step 7: Final commit if any fixes were needed**

```bash
git add files/annotator.py
git commit -m "fix: smoke test corrections"
```

---

## Usage Summary

```bash
# On RunPod:
pip install "gradio>=4.0.0"

python3 files/annotator.py \
  --image_dir ./fotos/train \
  --output_dir ./output \
  --classes coca-cola,monster \
  --sam_model facebook/sam3 \
  --device cuda \
  --port 7860
```

Open port 7860 from RunPod UI → annotate → Guardar y salir → download `output/label_studio_pack/` → import into Label Studio.
