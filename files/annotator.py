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
