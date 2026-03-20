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
            inputs["original_sizes"].cpu(),
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed color palette per class index (BGR for OpenCV)
COLORS = [
    (0, 200, 255),   # orange
    (0, 255, 100),   # green
    (255, 80, 80),   # blue
    (80, 80, 255),   # red
    (255, 0, 200),   # purple
]


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
        if self.is_done():
            return
        path = self.current_image_path()
        if path not in self._annotations:
            self._annotations[path] = []
        self._annotations[path].append((class_name, bbox_xyxy, img_w, img_h))

    def undo_last(self):
        if self.is_done():
            return
        path = self.current_image_path()
        if self._annotations.get(path):
            self._annotations[path].pop()

    def get_annotations(self, image_path: str) -> list:
        return self._annotations.get(image_path, [])

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


try:
    import gradio as gr
except ImportError:  # pragma: no cover
    gr = None

import argparse


def build_ui(session: "AnnotationSession", sam: "SAMBackend", classes: list[str]):

    def get_status():
        if session.is_done():
            return f"Terminado  |  Total: {len(session.image_paths)} imagenes"
        total = len(session.image_paths)
        idx = session.index
        anns = session.get_annotations(session.current_image_path())
        return f"Imagen {idx + 1} / {total}  |  Bboxes: {len(anns)}"

    def get_rendered():
        if session.is_done():
            return np.zeros((100, 100, 3), dtype=np.uint8)
        path = session.current_image_path()
        anns = session.get_annotations(path)
        bgr = render_image(path, anns, classes)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def on_click(image_rgb, evt: "gr.SelectData", selected_class):
        if session.is_done():
            return get_rendered(), "Sesion terminada, usa Guardar y salir"
        if selected_class is None:
            return get_rendered(), "Selecciona una clase primero"

        x, y = evt.index[0], evt.index[1]
        # Use the clean source image (not the rendered copy with boxes drawn on it)
        pil_image = PILImage.open(session.current_image_path()).convert("RGB")
        w, h = pil_image.size

        bbox = sam.segment(pil_image, (x, y))
        if bbox is None:
            return get_rendered(), "No se detecto objeto, intenta otro punto"

        session.add_annotation(selected_class, bbox, w, h)
        return get_rendered(), get_status()

    def on_undo():
        if not session.is_done():
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
