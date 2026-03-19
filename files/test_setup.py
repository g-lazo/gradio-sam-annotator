import sys
from pathlib import Path


def check_dependencies():
    errors = []

    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[OK] PyTorch {torch.__version__} | GPU: {gpu} ({vram:.1f}GB)")
        else:
            print(f"[!!] PyTorch {torch.__version__} | Sin GPU (CPU only, sera lento)")
    except ImportError:
        errors.append("PyTorch no instalado")

    try:
        import ultralytics
        print(f"[OK] Ultralytics {ultralytics.__version__}")
    except ImportError:
        errors.append("ultralytics no instalado: pip install ultralytics")

    try:
        import segment_anything
        print(f"[OK] segment-anything instalado")
    except ImportError:
        errors.append("segment-anything no instalado: pip install git+https://github.com/facebookresearch/segment-anything.git")

    try:
        import cv2
        print(f"[OK] OpenCV {cv2.__version__}")
    except ImportError:
        errors.append("opencv no instalado: pip install opencv-python-headless")

    return errors


def check_files(yolo_path: str, sam_path: str):
    errors = []

    if Path(yolo_path).exists():
        size_mb = Path(yolo_path).stat().st_size / (1024 * 1024)
        print(f"[OK] Modelo YOLO: {yolo_path} ({size_mb:.1f}MB)")
    else:
        errors.append(f"Modelo YOLO no encontrado: {yolo_path}")

    if Path(sam_path).exists():
        size_gb = Path(sam_path).stat().st_size / (1024**3)
        print(f"[OK] Checkpoint SAM: {sam_path} ({size_gb:.1f}GB)")
    else:
        errors.append(f"Checkpoint SAM no encontrado: {sam_path}")

    return errors


def quick_inference_test(yolo_path: str, sam_path: str):
    import numpy as np
    import cv2

    print("\n=== Test de inferencia rapida ===")

    print("Cargando YOLO...")
    from ultralytics import YOLO
    yolo = YOLO(yolo_path)
    print(f"  Clases del modelo: {list(yolo.names.values())}")

    dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    results = yolo(dummy_image, verbose=False)
    print(f"  Inferencia YOLO en imagen dummy: OK ({len(results[0].boxes)} detecciones)")

    print("Cargando SAM...")
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    model_type = "vit_h" if "vit_h" in sam_path else "vit_l" if "vit_l" in sam_path else "vit_b"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sam = sam_model_registry[model_type](checkpoint=sam_path)
    sam.to(device)
    predictor = SamPredictor(sam)

    predictor.set_image(dummy_image)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[320, 320]]),
        point_labels=np.array([1]),
        multimask_output=True,
    )
    print(f"  Inferencia SAM en imagen dummy: OK ({masks.shape[0]} mascaras)")

    print("\n[OK] Ambos modelos cargan y corren correctamente.")


if __name__ == "__main__":
    yolo_path = sys.argv[1] if len(sys.argv) > 1 else "best.pt"
    sam_path = sys.argv[2] if len(sys.argv) > 2 else "sam_vit_h_4b8939.pth"

    print("=== Verificacion de dependencias ===\n")
    dep_errors = check_dependencies()

    print("\n=== Verificacion de archivos ===\n")
    file_errors = check_files(yolo_path, sam_path)

    all_errors = dep_errors + file_errors
    if all_errors:
        print("\n=== ERRORES ===")
        for e in all_errors:
            print(f"  [X] {e}")
        print("\nCorrige los errores y vuelve a correr este test.")
        sys.exit(1)

    quick_inference_test(yolo_path, sam_path)
    print("\n=== Todo listo. Puedes correr el pipeline. ===")
