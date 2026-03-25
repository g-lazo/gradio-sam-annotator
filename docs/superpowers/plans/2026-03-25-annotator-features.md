# Annotator Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add autosave, discard, mask toggle, YOLO export, enhanced counter, and junk filtering to the Gradio annotator.

**Architecture:** All changes in `files/annotator.py`. Modify existing classes (SAMBackend, AnnotationSession, render_image, build_ui). Annotation tuples gain a 5th element (mask). Autosave writes progress.json on every action. Discard moves images to `_descartadas/`.

**Tech Stack:** Python 3.11, Gradio, OpenCV, numpy, PIL, SAM3 (transformers)

**Spec:** `docs/superpowers/specs/2026-03-25-annotator-features-design.md`

---

## File Structure

- Modify: `files/annotator.py` — all 6 features
- Modify: `files/tests/test_annotation_session.py` — tests for new AnnotationSession methods
- Modify: `files/tests/test_sam_backend.py` — test SAMBackend returns mask

---

### Task 1: Foundation — filter junk, SAMBackend returns mask

**Files:**
- Modify: `files/annotator.py:23-53` (SAMBackend.segment)
- Modify: `files/annotator.py:97-106` (AnnotationSession.__init__)
- Modify: `files/annotator.py:123-129` (AnnotationSession.add_annotation)
- Modify: `files/tests/test_sam_backend.py`
- Modify: `files/tests/test_annotation_session.py`

- [ ] **Step 1: Update SAMBackend.segment to return (bbox, mask) tuple**

```python
def segment(self, pil_image, point_xy: tuple) -> tuple[list, np.ndarray] | tuple[None, None]:
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
        inputs["original_sizes"].cpu(),
    )[0]  # [1, 1, H, W]

    mask = masks[0, 0].numpy().astype(bool)
    coords = np.where(mask)

    if len(coords[0]) == 0:
        return None, None

    bbox = [
        float(np.min(coords[1])),  # x1
        float(np.min(coords[0])),  # y1
        float(np.max(coords[1])),  # x2
        float(np.max(coords[0])),  # y2
    ]
    return bbox, mask
```

- [ ] **Step 2: Filter junk in AnnotationSession.__init__**

```python
def __init__(self, image_dir: str):
    self.image_dir = image_dir
    self.image_paths = sorted([
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
        and not p.name.startswith(".") and not p.name.startswith("_")
    ])
    self.index = 0
    # path -> list of (class_name, bbox_xyxy, img_w, img_h, mask_or_None)
    self._annotations: dict[str, list] = {}
    self._discarded: set[str] = set()
```

- [ ] **Step 3: Update add_annotation to accept mask**

```python
def add_annotation(self, class_name: str, bbox_xyxy: list, img_w: int, img_h: int, mask: np.ndarray = None):
    if self.is_done():
        return
    path = self.current_image_path()
    if path not in self._annotations:
        self._annotations[path] = []
    self._annotations[path].append((class_name, bbox_xyxy, img_w, img_h, mask))
```

- [ ] **Step 4: Update get_annotations unpacking in render_image and export**

In `render_image`, change the unpacking to handle 5-element tuples:

```python
for ann in annotations:
    class_name, bbox = ann[0], ann[1]
    x1, y1, x2, y2 = [int(c) for c in bbox]
    # ... rest same
```

In `export`, change:

```python
for idx, ann in enumerate(anns):
    class_name, bbox, img_w, img_h = ann[0], ann[1], ann[2], ann[3]
```

- [ ] **Step 5: Update tests**

In `test_sam_backend.py`, update `test_segment_returns_bbox_from_mask`:
```python
def test_segment_returns_bbox_and_mask(mock_sam):
    from annotator import SAMBackend
    mock_processor, mock_model, mock_outputs = mock_sam

    h, w = 150, 200
    mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
    mask[0, 0, 20:61, 30:91] = True
    mock_processor.post_process_masks = MagicMock(return_value=[mask])

    backend = SAMBackend("facebook/sam3", device="cpu")
    bbox, returned_mask = backend.segment(make_pil_image(w, h), (60, 40))

    assert bbox is not None
    assert returned_mask is not None
    assert bbox == [30.0, 20.0, 90.0, 60.0]
    assert returned_mask.shape == (h, w)
    assert returned_mask[40, 60] == True

def test_segment_returns_none_tuple_for_empty_mask(mock_sam):
    from annotator import SAMBackend
    mock_processor, mock_model, mock_outputs = mock_sam

    h, w = 150, 200
    empty_mask = torch.zeros(1, 1, h, w, dtype=torch.bool)
    mock_processor.post_process_masks = MagicMock(return_value=[empty_mask])

    backend = SAMBackend("facebook/sam3", device="cpu")
    bbox, mask = backend.segment(make_pil_image(w, h), (100, 75))
    assert bbox is None
    assert mask is None
```

In `test_annotation_session.py`, add:
```python
def test_session_filters_dotfiles_and_underscored(tmp_path):
    make_test_image(tmp_path / "good.jpg")
    make_test_image(tmp_path / ".hidden.jpg")
    make_test_image(tmp_path / "_discarded.jpg")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    session = AnnotationSession(str(tmp_path))
    assert len(session.image_paths) == 1
```

- [ ] **Step 6: Run tests**

Run: `cd files && python -m pytest tests/ -v`

- [ ] **Step 7: Commit**

```bash
git add files/annotator.py files/tests/
git commit -m "feat(annotator): return mask from SAM, filter junk files, 5-element annotations"
```

---

### Task 2: Autosave

**Files:**
- Modify: `files/annotator.py` (AnnotationSession)
- Modify: `files/tests/test_annotation_session.py`

- [ ] **Step 1: Add save_progress and load_progress methods**

Add to `AnnotationSession`:

```python
def save_progress(self, output_dir: str):
    progress_path = Path(output_dir) / "progress.json"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    # Convert annotations to serializable format (drop masks)
    serializable_anns = {}
    for path, anns in self._annotations.items():
        serializable_anns[path] = [
            (a[0], a[1], a[2], a[3]) for a in anns
        ]
    data = {
        "index": self.index,
        "annotations": serializable_anns,
        "discarded": list(self._discarded),
    }
    progress_path.write_text(json.dumps(data, indent=2))

def load_progress(self, output_dir: str) -> bool:
    progress_path = Path(output_dir) / "progress.json"
    if not progress_path.exists():
        return False
    data = json.loads(progress_path.read_text())
    self.index = data.get("index", 0)
    self._discarded = set(data.get("discarded", []))
    for path, anns in data.get("annotations", {}).items():
        self._annotations[path] = [
            (a[0], a[1], a[2], a[3], None) for a in anns
        ]
    return True
```

- [ ] **Step 2: Add tests**

```python
def test_save_and_load_progress(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    session.next_image()
    session.save_progress(str(tmp_path))

    assert (tmp_path / "progress.json").exists()

    session2 = AnnotationSession(str(image_dir))
    loaded = session2.load_progress(str(tmp_path))
    assert loaded is True
    assert session2.index == 1
    anns = session2.get_annotations(session2.image_paths[0])
    assert len(anns) == 1
    assert anns[0][0] == "coca-cola"

def test_load_progress_returns_false_when_no_file(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    assert session.load_progress(str(tmp_path)) is False
```

- [ ] **Step 3: Run tests**

Run: `cd files && python -m pytest tests/test_annotation_session.py -v`

- [ ] **Step 4: Commit**

```bash
git add files/annotator.py files/tests/test_annotation_session.py
git commit -m "feat(annotator): add autosave with progress.json"
```

---

### Task 3: Discard image

**Files:**
- Modify: `files/annotator.py` (AnnotationSession)
- Modify: `files/tests/test_annotation_session.py`

- [ ] **Step 1: Add discard_image method and update navigation**

```python
def discard_image(self, output_dir: str):
    if self.is_done():
        return
    path = self.current_image_path()
    discard_dir = Path(output_dir) / "_descartadas"
    discard_dir.mkdir(parents=True, exist_ok=True)
    src = Path(path)
    dst = discard_dir / src.name
    shutil.move(str(src), str(dst))
    self._discarded.add(path)
    # Remove any annotations for this image
    self._annotations.pop(path, None)
    # Skip to next non-discarded
    self._skip_discarded_forward()

def _skip_discarded_forward(self):
    while not self.is_done() and self.current_image_path() in self._discarded:
        self.index += 1

def _skip_discarded_backward(self):
    while self.index > 0 and self.image_paths[self.index] in self._discarded:
        self.index -= 1
```

Update `next_image` and `prev_image`:

```python
def next_image(self):
    if self.index < len(self.image_paths):
        self.index += 1
    self._skip_discarded_forward()

def prev_image(self):
    if self.index > 0:
        self.index -= 1
    self._skip_discarded_backward()
```

- [ ] **Step 2: Add tests**

```python
def test_discard_image_moves_file(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    first_path = session.current_image_path()
    first_name = Path(first_path).name
    session.discard_image(str(tmp_path))
    assert (tmp_path / "_descartadas" / first_name).exists()
    assert not Path(first_path).exists()
    assert first_path in session._discarded

def test_discard_skips_to_next(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.discard_image(str(tmp_path))
    # Should now be on the second image (index may still be 0 but the first is discarded)
    assert not session.is_done()
    assert session.current_image_path() not in session._discarded

def test_navigation_skips_discarded(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.next_image()  # go to b.jpg
    discarded_path = session.current_image_path()
    session.discard_image(str(tmp_path))  # discard b.jpg
    # Should skip past b.jpg
    session.prev_image()
    assert session.current_image_path() != discarded_path
```

- [ ] **Step 3: Run tests**

Run: `cd files && python -m pytest tests/test_annotation_session.py -v`

- [ ] **Step 4: Commit**

```bash
git add files/annotator.py files/tests/test_annotation_session.py
git commit -m "feat(annotator): add discard image with move to _descartadas"
```

---

### Task 4: render_image with mask overlay

**Files:**
- Modify: `files/annotator.py:68-94` (render_image)

- [ ] **Step 1: Add mask overlay support**

```python
def render_image(image_path: str, annotations: list, classes: list[str], show_masks: bool = False) -> np.ndarray:
    img = cv2.imread(image_path)
    if img is None:
        h, w = 480, 640
        img = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(img, "No se pudo leer la imagen", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return img

    for ann in annotations:
        class_name, bbox = ann[0], ann[1]
        mask = ann[4] if len(ann) > 4 else None
        x1, y1, x2, y2 = [int(c) for c in bbox]
        class_idx = classes.index(class_name) if class_name in classes else 0
        color = COLORS[class_idx % len(COLORS)]

        # Draw mask overlay if enabled and mask exists
        if show_masks and mask is not None:
            overlay = img.copy()
            overlay[mask] = color
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = class_name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return img
```

- [ ] **Step 2: Commit**

```bash
git add files/annotator.py
git commit -m "feat(annotator): add mask overlay toggle to render_image"
```

---

### Task 5: YOLO export

**Files:**
- Modify: `files/annotator.py` (AnnotationSession.export)
- Modify: `files/tests/test_annotation_session.py`

- [ ] **Step 1: Add YOLO export to export method**

Add at the end of `export()`, before the print statements:

```python
# YOLO labels export
labels_dir = Path(output_dir) / "labels"
labels_dir.mkdir(parents=True, exist_ok=True)
for image_path in self.image_paths:
    if image_path in self._discarded:
        continue
    anns = self._annotations.get(image_path, [])
    image_name = Path(image_path).stem
    lines = []
    for ann in anns:
        class_name, bbox, img_w, img_h = ann[0], ann[1], ann[2], ann[3]
        class_id = classes.index(class_name) if class_name in classes else 0
        x1, y1, x2, y2 = bbox
        cx = ((x1 + x2) / 2) / img_w
        cy = ((y1 + y2) / 2) / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    (labels_dir / f"{image_name}.txt").write_text("\n".join(lines))
```

Update `export` signature to accept classes:

```python
def export(self, output_dir: str, classes: list[str] = None):
```

Also update the Label Studio export to skip discarded images:

```python
for image_path in self.image_paths:
    if image_path in self._discarded:
        continue
    # ... rest of existing export logic
```

- [ ] **Step 2: Add test**

```python
def test_export_yolo_labels(image_dir, tmp_path):
    session = AnnotationSession(str(image_dir))
    session.add_annotation("coca-cola", [10.0, 20.0, 50.0, 60.0], 100, 80)
    session.export(str(tmp_path), classes=["coca-cola", "monster"])

    labels_dir = tmp_path / "labels"
    assert labels_dir.exists()
    label_file = labels_dir / "a.txt"
    assert label_file.exists()
    content = label_file.read_text().strip()
    parts = content.split()
    assert parts[0] == "0"  # class_id for coca-cola
    assert len(parts) == 5  # class_id cx cy w h
```

- [ ] **Step 3: Run tests**

Run: `cd files && python -m pytest tests/test_annotation_session.py -v`

- [ ] **Step 4: Commit**

```bash
git add files/annotator.py files/tests/test_annotation_session.py
git commit -m "feat(annotator): add YOLO label export alongside Label Studio"
```

---

### Task 6: UI updates — all new buttons and handlers

**Files:**
- Modify: `files/annotator.py:199-307` (build_ui, main)

- [ ] **Step 1: Update build_ui with all new features**

Replace the entire `build_ui` function:

```python
def build_ui(session: "AnnotationSession", sam: "SAMBackend", classes: list[str], output_dir: str):
    show_masks_state = {"enabled": False}

    def get_status():
        if session.is_done():
            annotated = sum(1 for p in session.image_paths if session.get_annotations(p))
            return f"Terminado | Anotadas: {annotated} | Descartadas: {len(session._discarded)}"
        total = len(session.image_paths)
        idx = session.index
        anns = session.get_annotations(session.current_image_path())
        annotated = sum(1 for p in session.image_paths if session.get_annotations(p))
        return f"Imagen {idx + 1}/{total} | Anotadas: {annotated} | Descartadas: {len(session._discarded)} | Bboxes aqui: {len(anns)}"

    def get_rendered():
        if session.is_done():
            return np.zeros((100, 100, 3), dtype=np.uint8)
        path = session.current_image_path()
        anns = session.get_annotations(path)
        bgr = render_image(path, anns, classes, show_masks=show_masks_state["enabled"])
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def autosave():
        session.save_progress(output_dir)

    def on_click(image_rgb, evt: "gr.SelectData", selected_class):
        if session.is_done():
            return get_rendered(), "Sesion terminada, usa Exportar"
        if selected_class is None:
            return get_rendered(), "Selecciona una clase primero"

        x, y = evt.index[0], evt.index[1]
        pil_image = PILImage.open(session.current_image_path()).convert("RGB")
        w, h = pil_image.size

        bbox, mask = sam.segment(pil_image, (x, y))
        if bbox is None:
            return get_rendered(), "No se detecto objeto, intenta otro punto"

        session.add_annotation(selected_class, bbox, w, h, mask)
        autosave()
        return get_rendered(), get_status()

    def on_undo():
        if not session.is_done():
            session.undo_last()
            autosave()
        return get_rendered(), get_status()

    def on_next():
        session.next_image()
        autosave()
        return get_rendered(), get_status()

    def on_prev():
        session.prev_image()
        autosave()
        return get_rendered(), get_status()

    def on_discard():
        if not session.is_done():
            session.discard_image(output_dir)
            autosave()
        return get_rendered(), get_status()

    def on_toggle_masks():
        show_masks_state["enabled"] = not show_masks_state["enabled"]
        label = "Ocultar mascaras" if show_masks_state["enabled"] else "Mostrar mascaras"
        return get_rendered(), get_status(), label

    def on_export():
        session.export(output_dir, classes=classes)
        return get_rendered(), f"Exportado en {output_dir}/label_studio_pack y {output_dir}/labels"

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
                discard_btn = gr.Button("Descartar imagen")
                toggle_masks_btn = gr.Button("Mostrar mascaras")
                with gr.Row():
                    prev_btn = gr.Button("← Anterior")
                    next_btn = gr.Button("Siguiente →")
                export_btn = gr.Button("Exportar", variant="primary")

        image_display.select(on_click, inputs=[image_display, class_radio],
                              outputs=[image_display, status_label])
        undo_btn.click(on_undo, outputs=[image_display, status_label])
        discard_btn.click(on_discard, outputs=[image_display, status_label])
        toggle_masks_btn.click(on_toggle_masks, outputs=[image_display, status_label, toggle_masks_btn])
        prev_btn.click(on_prev, outputs=[image_display, status_label])
        next_btn.click(on_next, outputs=[image_display, status_label])
        export_btn.click(on_export, outputs=[image_display, status_label])

    return demo
```

- [ ] **Step 2: Update main() to load progress and pass output_dir**

```python
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

    if session.load_progress(args.output_dir):
        annotated = sum(1 for p in session.image_paths if session.get_annotations(p))
        print(f"  Progreso cargado: imagen {session.index + 1}, {annotated} anotadas, {len(session._discarded)} descartadas")
    else:
        print("  Sin progreso previo, empezando de cero")

    demo = build_ui(session, sam, classes, args.output_dir)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)
```

- [ ] **Step 3: Run all tests**

Run: `cd files && python -m pytest tests/ -v`

- [ ] **Step 4: Commit**

```bash
git add files/annotator.py
git commit -m "feat(annotator): add UI for discard, mask toggle, autosave, enhanced status, export"
```
