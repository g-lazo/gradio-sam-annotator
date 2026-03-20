import os
import json
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float]
    refined_bbox: Optional[list[float]] = None
    mask: Optional[np.ndarray] = None
    auto_approved: bool = False


@dataclass
class FrameResult:
    image_path: str
    detections: list[Detection] = field(default_factory=list)
    all_high_confidence: bool = False


class VideoFrameExtractor:

    def __init__(self, output_dir: str, frame_interval: float = 0.5, blur_threshold: float = 100.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_interval = frame_interval
        self.blur_threshold = blur_threshold

    def _is_blurry(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.blur_threshold

    def _frames_are_similar(self, frame_a: np.ndarray, frame_b: np.ndarray, threshold: float = 0.95) -> bool:
        gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
        hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
        hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])
        cv2.normalize(hist_a, hist_a)
        cv2.normalize(hist_b, hist_b)
        score = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        return score > threshold

    def extract(self, video_path: str) -> list[str]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"No se pudo abrir el video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(fps * self.frame_interval))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        saved_paths = []
        last_saved_frame = None
        frame_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_skip != 0:
                frame_idx += 1
                continue

            if self._is_blurry(frame):
                frame_idx += 1
                continue

            if last_saved_frame is not None and self._frames_are_similar(frame, last_saved_frame):
                frame_idx += 1
                continue

            filename = f"frame_{saved_count:05d}.jpg"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), frame)
            saved_paths.append(str(filepath))
            last_saved_frame = frame.copy()
            saved_count += 1
            frame_idx += 1

        cap.release()
        print(f"Video: {total_frames} frames totales, {saved_count} frames utiles extraidos")
        return saved_paths


class YOLOSAMPipeline:

    def __init__(
        self,
        yolo_model_path: str,
        sam_model_type: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        sam_checkpoint: str = "sam2_hiera_large.pt",
        confidence_threshold: float = 0.85,
        auto_approve_enabled: bool = False,
        device: str = "cuda",
    ):
        self.confidence_threshold = confidence_threshold
        self.auto_approve_enabled = auto_approve_enabled
        self.device = device
        self._load_yolo(yolo_model_path)
        self._load_sam(sam_model_type, sam_checkpoint)

    def _load_yolo(self, model_path: str):
        from ultralytics import YOLO
        self.yolo = YOLO(model_path)

    def _load_sam(self, model_type: str, checkpoint: str):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2 = build_sam2(model_type, checkpoint, device=self.device)
        self.sam_predictor = SAM2ImagePredictor(sam2)

    def _refine_bbox_with_sam(self, bbox_xyxy: list[float]) -> tuple[list[float], np.ndarray]:
        x1, y1, x2, y2 = bbox_xyxy
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        input_point = np.array([[center_x, center_y]])
        input_label = np.array([1])
        input_box = np.array([x1, y1, x2, y2])

        masks, scores, _ = self.sam_predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )

        best_idx = np.argmax(scores)
        best_mask = masks[best_idx]

        coords = np.where(best_mask)
        if len(coords[0]) == 0:
            return bbox_xyxy, best_mask

        refined_bbox = [
            float(np.min(coords[1])),
            float(np.min(coords[0])),
            float(np.max(coords[1])),
            float(np.max(coords[0])),
        ]

        return refined_bbox, best_mask

    def process_image(self, image_path: str) -> FrameResult:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"No se pudo leer la imagen: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.yolo(image_rgb, verbose=False)[0]

        frame_result = FrameResult(image_path=image_path)

        if results.boxes:
            self.sam_predictor.set_image(image_rgb)

        for box in results.boxes:
            det = Detection(
                class_id=int(box.cls[0]),
                class_name=self.yolo.names[int(box.cls[0])],
                confidence=float(box.conf[0]),
                bbox_xyxy=[float(c) for c in box.xyxy[0]],
            )

            refined_bbox, mask = self._refine_bbox_with_sam(det.bbox_xyxy)
            det.refined_bbox = refined_bbox
            det.mask = mask
            det.auto_approved = (
                self.auto_approve_enabled and det.confidence >= self.confidence_threshold
            )

            frame_result.detections.append(det)

        frame_result.all_high_confidence = (
            self.auto_approve_enabled
            and len(frame_result.detections) > 0
            and all(d.auto_approved for d in frame_result.detections)
        )
        return frame_result

    def process_batch(self, image_paths: list[str]) -> tuple[list[FrameResult], list[FrameResult]]:
        auto_approved = []
        needs_review = []

        for i, path in enumerate(image_paths):
            result = self.process_image(path)
            if result.all_high_confidence and len(result.detections) > 0:
                auto_approved.append(result)
            else:
                needs_review.append(result)

            if (i + 1) % 10 == 0:
                print(f"Procesadas {i + 1}/{len(image_paths)} imagenes")

        print(f"Auto-aprobadas: {len(auto_approved)}, Revision: {len(needs_review)}")
        return auto_approved, needs_review


class AnnotationExporter:

    @staticmethod
    def to_yolo_format(results: list[FrameResult], output_dir: str, use_refined: bool = True):
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for result in results:
            image = cv2.imread(result.image_path)
            h, w = image.shape[:2]
            image_name = Path(result.image_path).stem

            lines = []
            for det in result.detections:
                bbox = det.refined_bbox if (use_refined and det.refined_bbox) else det.bbox_xyxy
                x1, y1, x2, y2 = bbox
                cx = ((x1 + x2) / 2) / w
                cy = ((y1 + y2) / 2) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            label_file = output_path / f"{image_name}.txt"
            label_file.write_text("\n".join(lines))

    @staticmethod
    def to_label_studio_json(
        results: list[FrameResult],
        use_refined: bool = True,
        image_base_path: str = "/data/local-files/?d=",
    ) -> list[dict]:
        tasks = []

        for result in results:
            image = cv2.imread(result.image_path)
            h, w = image.shape[:2]
            image_filename = Path(result.image_path).name

            annotations = []
            for idx, det in enumerate(result.detections):
                bbox = det.refined_bbox if (use_refined and det.refined_bbox) else det.bbox_xyxy
                x1, y1, x2, y2 = bbox

                annotations.append({
                    "id": f"{Path(result.image_path).stem}_{idx}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "value": {
                        "x": (x1 / w) * 100,
                        "y": (y1 / h) * 100,
                        "width": ((x2 - x1) / w) * 100,
                        "height": ((y2 - y1) / h) * 100,
                        "rectanglelabels": [det.class_name],
                    },
                    "score": det.confidence,
                })

            task = {
                "data": {"image": f"{image_base_path}{image_filename}"},
                "predictions": [{"result": annotations}],
            }
            tasks.append(task)

        return tasks

    @staticmethod
    def pack_for_local_review(
        results: list[FrameResult],
        output_dir: str,
        use_refined: bool = True,
    ):
        pack_dir = Path(output_dir) / "label_studio_pack"
        images_dir = pack_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        import shutil
        for result in results:
            src = Path(result.image_path)
            dst = images_dir / src.name
            if src != dst:
                shutil.copy2(str(src), str(dst))

        tasks = AnnotationExporter.to_label_studio_json(
            results,
            use_refined=use_refined,
            image_base_path="/data/local-files/?d=images/",
        )

        json_path = pack_dir / "tasks.json"
        json_path.write_text(json.dumps(tasks, indent=2))

        tasks_upload = AnnotationExporter.to_label_studio_json(
            results,
            use_refined=use_refined,
            image_base_path="",
        )
        for task, result in zip(tasks_upload, results):
            task["data"]["image"] = Path(result.image_path).name

        json_upload_path = pack_dir / "tasks_upload.json"
        json_upload_path.write_text(json.dumps(tasks_upload, indent=2))

        print(f"Pack listo en: {pack_dir}")
        print(f"  {len(results)} imagenes copiadas a {images_dir}")
        print(f"  tasks.json -> para Label Studio con local storage")
        print(f"  tasks_upload.json -> para importar subiendo imagenes directo")
        return str(pack_dir)


def run_pipeline_from_video(
    video_path: str,
    yolo_model_path: str,
    sam_checkpoint: str = "sam2_hiera_large.pt",
    output_dir: str = "./pipeline_output",
    frame_interval: float = 0.5,
    confidence_threshold: float = 0.85,
    auto_approve_enabled: bool = False,
    sam_model_type: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
):
    output_base = Path(output_dir)
    frames_dir = output_base / "frames"
    labels_approved_dir = output_base / "labels_approved"
    labels_review_dir = output_base / "labels_review"

    print("=== Paso 1: Extraccion de frames ===")
    extractor = VideoFrameExtractor(
        output_dir=str(frames_dir),
        frame_interval=frame_interval,
    )
    frame_paths = extractor.extract(video_path)

    if not frame_paths:
        print("No se extrajeron frames utiles del video")
        return

    print(f"\n=== Paso 2: YOLO + SAM en {len(frame_paths)} frames ===")
    pipeline = YOLOSAMPipeline(
        yolo_model_path=yolo_model_path,
        sam_model_type=sam_model_type,
        sam_checkpoint=sam_checkpoint,
        confidence_threshold=confidence_threshold,
        auto_approve_enabled=auto_approve_enabled,
        device=device,
    )

    auto_approved, needs_review = pipeline.process_batch(frame_paths)

    print("\n=== Paso 3: Exportando anotaciones ===")
    exporter = AnnotationExporter()

    exporter.to_yolo_format(auto_approved, str(labels_approved_dir))
    exporter.to_yolo_format(needs_review, str(labels_review_dir))

    all_results = auto_approved + needs_review
    exporter.pack_for_local_review(all_results, str(output_base))

    if needs_review:
        exporter.pack_for_local_review(
            needs_review, str(output_base), use_refined=True,
        )

    print(f"\nResultados en: {output_base}")
    print(f"  Frames extraidos: {len(frame_paths)}")
    print(f"  Auto-aprobadas: {len(auto_approved)}")
    print(f"  Para revision: {len(needs_review)}")
    print(f"  Labels YOLO (aprobadas): {labels_approved_dir}")
    print(f"  Labels YOLO (revision): {labels_review_dir}")
    print(f"  Pack Label Studio: {output_base / 'label_studio_pack'}")


def run_pipeline_from_images(
    image_dir: str,
    yolo_model_path: str,
    sam_checkpoint: str = "sam2_hiera_large.pt",
    output_dir: str = "./pipeline_output",
    confidence_threshold: float = 0.85,
    auto_approve_enabled: bool = False,
    sam_model_type: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    device: str = "cuda",
):
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted([
        str(p) for p in Path(image_dir).iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        print(f"No se encontraron imagenes en {image_dir}")
        return

    output_base = Path(output_dir)
    labels_approved_dir = output_base / "labels_approved"
    labels_review_dir = output_base / "labels_review"

    print(f"=== Procesando {len(image_paths)} imagenes ===")
    pipeline = YOLOSAMPipeline(
        yolo_model_path=yolo_model_path,
        sam_model_type=sam_model_type,
        sam_checkpoint=sam_checkpoint,
        confidence_threshold=confidence_threshold,
        auto_approve_enabled=auto_approve_enabled,
        device=device,
    )

    auto_approved, needs_review = pipeline.process_batch(image_paths)

    print("\n=== Exportando anotaciones ===")
    exporter = AnnotationExporter()

    exporter.to_yolo_format(auto_approved, str(labels_approved_dir))
    exporter.to_yolo_format(needs_review, str(labels_review_dir))

    all_results = auto_approved + needs_review
    exporter.pack_for_local_review(all_results, str(output_base))

    print(f"\nAuto-aprobadas: {len(auto_approved)}, Revision: {len(needs_review)}")
    print(f"Pack Label Studio: {output_base / 'label_studio_pack'}")


if __name__ == "__main__":
    run_pipeline_from_images(
        image_dir="./fotos/train",
        yolo_model_path="best.pt",
        output_dir="./output",
        confidence_threshold=0.5,
        auto_approve_enabled=False,
    )
