import argparse

import cv2
import numpy as np
import gradio as gr
from PIL import Image as PILImage

from sam_backend import SAMBackend, render_image
from session import AnnotationSession


def build_ui(session: AnnotationSession, sam: SAMBackend, classes: list[str], output_dir: str):
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

    def on_click(image_rgb, evt: gr.SelectData, selected_class):
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
    parser.add_argument("--share", action="store_true", help="Generar link publico de Gradio")
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
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
