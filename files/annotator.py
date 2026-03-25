import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

try:
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
except ImportError:  # pragma: no cover
    Sam3TrackerModel = None  # type: ignore
    Sam3TrackerProcessor = None  # type: ignore


class SAMBackend:

    def __init__(self, model_name: str = "facebook/sam3", device: str = "cuda"):
        self.device = device
        self.model = Sam3TrackerModel.from_pretrained(model_name).to(device)
        self.processor = Sam3TrackerProcessor.from_pretrained(model_name)
        self.model.eval()

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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed color palette per class index (BGR for OpenCV)
COLORS = [
    (0, 200, 255),   # orange
    (0, 255, 100),   # green
    (255, 80, 80),   # blue
    (80, 80, 255),   # red
    (255, 0, 200),   # purple
]


def render_image(image_path: str, annotations: list, classes: list[str], show_masks: bool = False) -> np.ndarray:
    """
    Draw bboxes and class labels on image copy.
    annotations: list of (class_name, bbox_xyxy, img_w, img_h, mask_or_None)
    Returns BGR numpy array.
    """
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


class AnnotationSession:

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

    def current_image_path(self) -> str:
        return self.image_paths[self.index]

    def is_done(self) -> bool:
        return self.index >= len(self.image_paths)

    def _skip_discarded_forward(self):
        while not self.is_done() and self.current_image_path() in self._discarded:
            self.index += 1

    def _skip_discarded_backward(self):
        while self.index > 0 and self.image_paths[self.index] in self._discarded:
            self.index -= 1

    def next_image(self):
        if self.index < len(self.image_paths):
            self.index += 1
        self._skip_discarded_forward()

    def prev_image(self):
        if self.index > 0:
            self.index -= 1
        self._skip_discarded_backward()

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
        self._annotations.pop(path, None)
        self._skip_discarded_forward()

    def add_annotation(self, class_name: str, bbox_xyxy: list, img_w: int, img_h: int, mask: np.ndarray = None):
        if self.is_done():
            return
        path = self.current_image_path()
        if path not in self._annotations:
            self._annotations[path] = []
        self._annotations[path].append((class_name, bbox_xyxy, img_w, img_h, mask))

    def undo_last(self):
        if self.is_done():
            return
        path = self.current_image_path()
        if self._annotations.get(path):
            self._annotations[path].pop()

    def get_annotations(self, image_path: str) -> list:
        return self._annotations.get(image_path, [])

    def save_progress(self, output_dir: str):
        progress_path = Path(output_dir) / "progress.json"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
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

    def export(self, output_dir: str, classes: list[str] = None):
        pack_dir = Path(output_dir) / "label_studio_pack"
        images_dir = pack_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        tasks_local = []
        tasks_upload = []

        for image_path in self.image_paths:
            if image_path in self._discarded:
                continue
            src = Path(image_path)
            dst = images_dir / src.name
            if src != dst:
                shutil.copy2(str(src), str(dst))

            anns = self._annotations.get(image_path, [])
            result = []
            for idx, ann in enumerate(anns):
                class_name, bbox, img_w, img_h = ann[0], ann[1], ann[2], ann[3]
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
                class_id = classes.index(class_name) if classes and class_name in classes else 0
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / img_w
                cy = ((y1 + y2) / 2) / img_h
                bw = (x2 - x1) / img_w
                bh = (y2 - y1) / img_h
                lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (labels_dir / f"{image_name}.txt").write_text("\n".join(lines))

        total_anns = sum(len(v) for v in self._annotations.values())
        print(f"Pack exportado en: {pack_dir}")
        print(f"Labels YOLO en: {labels_dir}")
        print(f"  {len(self.image_paths) - len(self._discarded)} imagenes, {total_anns} anotaciones")


try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

import argparse


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


if __name__ == "__main__":
    main()
