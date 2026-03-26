import cv2
import numpy as np
from PIL import Image as PILImage

try:
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
except ImportError:
    Sam3TrackerModel = None
    Sam3TrackerProcessor = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Fixed color palette per class index (BGR for OpenCV)
COLORS = [
    (0, 200, 255),   # orange
    (0, 255, 100),   # green
    (255, 80, 80),   # blue
    (80, 80, 255),   # red
    (255, 0, 200),   # purple
]


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
            input_points=[[[[x, y]]]],
            input_labels=[[[1]]],
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
