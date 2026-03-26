import json
import shutil
import numpy as np
from pathlib import Path

from sam_backend import IMAGE_EXTENSIONS


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
