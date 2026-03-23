# Gradio Annotator — Design Spec

**Date:** 2026-03-20
**Status:** Approved

## Overview

A web-based annotation tool that runs on RunPod (RTX 4090) and is accessed from a local browser. The user clicks on objects in images, SAM3 segments them, and the tool outputs a Label Studio-compatible pack when done.

## Context

- Images live in a folder on the RunPod pod (e.g., `./fotos`)
- SAM3 runs on the pod GPU (RTX 4090, 24GB VRAM)
- User accesses the Gradio UI from their Mac browser via RunPod's port forwarding
- Output must match the existing Label Studio pack format used by `pipeline_yolo_sam.py`

## Components

### `annotator.py`

Single script, two classes:

**`SAMBackend`**
- Loads SAM3 via `transformers` (`Sam3TrackerModel` + `Sam3TrackerProcessor`) on startup
- Exposes one method: `segment(pil_image, point_xy) -> bbox_xyxy`
  - Calls processor with `input_points=[[[x, y]]]`, `input_labels=[[1]]` (foreground), `return_tensors="pt"`
  - Calls `post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])` to get full-resolution mask
  - Returns bounding box `[x1, y1, x2, y2]` in absolute pixels derived from `np.where(mask)` min/max
  - Returns `None` if mask is empty (no True pixels)

**`AnnotationSession`**
- Holds the list of image paths (sorted from `--image_dir`)
- Tracks current image index and a dict mapping image path → list of `(class_name, bbox_xyxy, img_w, img_h)`
  - `img_w` and `img_h` are captured when the image is first loaded, to avoid re-reading at export time
- Methods: `current_image()`, `add_annotation()`, `undo_last()`, `next_image()`, `prev_image()`, `export()`
- `export()` uses stored `img_w`/`img_h` to convert absolute pixel bboxes to percentage coordinates for Label Studio JSON

### Gradio UI

Two-column layout:

- **Left:** `gr.Image` component in `select` mode — captures click coordinates on image
- **Right:** `gr.Radio` for class selection, status label (image N/total, bbox count), and buttons: Undo, Previous, Next, Save & Exit

### Interaction flow

1. User selects class via radio button
2. User clicks on image → Gradio fires `select` event with `(x, y)` pixel coords
3. `SAMBackend.segment()` runs with that point
4. If valid mask: `AnnotationSession.add_annotation()` stores it; UI redraws image with all bboxes for current image overlaid using `cv2.rectangle` + `cv2.putText`
5. Undo removes the last annotation for current image and redraws
6. Next/Previous saves nothing extra (state already in memory) and loads adjacent image
7. Save & Exit calls `export()` then closes Gradio

### Image rendering

Bboxes drawn directly on a copy of the image using OpenCV before passing to the Gradio image component. Each class gets a fixed color (cycle through a small palette). Class name and confidence label drawn above each box.

## CLI

```bash
python annotator.py \
  --image_dir ./fotos \
  --output_dir ./output \
  --classes coca-cola,monster \
  --sam_model facebook/sam3 \
  --device cuda \
  --port 7860
```

All arguments have defaults (`device=cuda`, `port=7860`, `sam_model=facebook/sam3`). `--classes` is comma-separated; order determines the radio button order and the class IDs (0-indexed) written to Label Studio JSON.

## Output format

Matches `AnnotationExporter.pack_for_local_review()` from `pipeline_yolo_sam.py`:

```
output/
└── label_studio_pack/
    ├── images/            # copies of annotated images
    ├── tasks.json         # for Label Studio with local storage
    └── tasks_upload.json  # for direct upload import
```

Each task entry:
```json
{
  "data": { "image": "/data/local-files/?d=images/<filename>" },
  "predictions": [{
    "result": [{
      "id": "<stem>_<idx>",
      "type": "rectanglelabels",
      "from_name": "label",
      "to_name": "image",
      "value": {
        "x": ..., "y": ..., "width": ..., "height": ...,
        "rectanglelabels": ["<class_name>"]
      },
      "score": 1.0  // always 1.0 — no inference confidence in manual annotation
    }]
  }]
}
```

Images with zero annotations are included in the export with an empty `result` list so they appear in Label Studio for review.

## Output format — tasks_upload.json detail

`tasks_upload.json` is identical to `tasks.json` except `data.image` is the bare filename (no path prefix), matching the pipeline's behavior where Label Studio handles the URL when files are uploaded directly.

## Dependencies

No new dependencies beyond what the pipeline already uses:
- `gradio` (add to requirements)
- `transformers`, `torch`, `opencv-python-headless`, `Pillow` — already in `requirements_pipeline.txt`

## Error handling

- If SAM3 returns empty mask: show a brief status message "No se detectó objeto, intenta otro punto" and do not add an annotation
- If image can't be read: skip it and log a warning, continue to next
- The only safe exit path is the "Guardar y salir" button — Gradio has no reliable server-side event for browser tab close, so that case is not handled. Closing the tab without saving loses unsaved state.

## What this is NOT

- Not a replacement for Label Studio — class corrections and fine-grained adjustments still happen there
- Not a batch processor — requires one click per object
- No auto-advance: user controls pacing via Next button
