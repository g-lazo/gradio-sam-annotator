#!/bin/bash
set -e

echo "=== Setup Pipeline YOLO + SAM ==="
echo ""

WORKSPACE="/workspace"
cd "$WORKSPACE"

echo "[1/4] Instalando dependencias..."
pip install -q ultralytics opencv-python-headless
pip install -q git+https://github.com/facebookresearch/segment-anything.git

echo "[2/4] Verificando checkpoint de SAM..."
SAM_CHECKPOINT="sam_vit_h_4b8939.pth"
if [ ! -f "$SAM_CHECKPOINT" ]; then
    echo "  Descargando SAM vit_h (2.4GB)..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
else
    echo "  Checkpoint ya existe, saltando descarga."
fi

echo "[3/4] Verificando archivos necesarios..."
MISSING=0

if [ ! -f "pipeline_yolo_sam.py" ]; then
    echo "  FALTA: pipeline_yolo_sam.py"
    MISSING=1
fi

if [ ! -f "best.pt" ]; then
    echo "  FALTA: best.pt (tu modelo YOLO entrenado)"
    echo "  Sube tu modelo con: scp -P <puerto> best.pt root@<pod-ip>:/workspace/"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Sube los archivos faltantes y vuelve a correr este script."
    exit 1
fi

echo "  Todos los archivos presentes."

echo "[4/4] Verificando GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f'  GPU: {gpu_name} ({vram:.1f}GB VRAM)')
else:
    print('  ADVERTENCIA: No se detecto GPU. SAM correra en CPU (muy lento).')
"

echo ""
echo "=== Setup completo ==="
echo ""
echo "Para probar con un video:"
echo "  python3 -c \""
echo "  from pipeline_yolo_sam import run_pipeline_from_video"
echo "  run_pipeline_from_video("
echo "      video_path='tu_video.mp4',"
echo "      yolo_model_path='best.pt',"
echo "      sam_checkpoint='sam_vit_h_4b8939.pth',"
echo "      output_dir='./output',"
echo "  )"
echo "  \""
echo ""
echo "Para probar con imagenes:"
echo "  python3 -c \""
echo "  from pipeline_yolo_sam import run_pipeline_from_images"
echo "  run_pipeline_from_images("
echo "      image_dir='./mis_fotos',"
echo "      yolo_model_path='best.pt',"
echo "      sam_checkpoint='sam_vit_h_4b8939.pth',"
echo "      output_dir='./output',"
echo "  )"
echo "  \""
