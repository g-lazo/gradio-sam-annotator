# Pipeline YOLO + SAM: Guia completa para RunPod

## Setup en RunPod

### 1. Crear pod

Elige un template con PyTorch preinstalado (por ejemplo `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`). GPU recomendada: RTX 3090 o A5000 (24GB VRAM, sobra para SAM vit_h + YOLO). Si quieres gastar menos, una RTX 3060 (12GB) jala con SAM vit_b.

### 2. Conectarte al pod

Desde la interfaz de RunPod, click en "Connect" > "Start Web Terminal" o usa SSH:

```bash
ssh root@<tu-pod-ip> -p <puerto> -i ~/.ssh/tu_llave
```

### 3. Instalar dependencias

```bash
pip install ultralytics opencv-python-headless
pip install git+https://github.com/facebookresearch/segment-anything.git
```

Nota: `opencv-python-headless` en lugar de `opencv-python` porque RunPod no tiene display.

### 4. Descargar checkpoint de SAM

```bash
cd /workspace
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Para GPU con poca VRAM (<=8GB), usa vit_b:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### 5. Subir tus archivos

Sube `pipeline_yolo_sam.py`, tu modelo YOLO (`best.pt`), y tus videos/imagenes.

Opcion A - Desde la terminal web de RunPod, usa el file manager.

Opcion B - Con scp desde tu maquina local:

```bash
scp -P <puerto> pipeline_yolo_sam.py root@<tu-pod-ip>:/workspace/
scp -P <puerto> best.pt root@<tu-pod-ip>:/workspace/
scp -P <puerto> mi_video.mp4 root@<tu-pod-ip>:/workspace/
```

Opcion C - Si tus archivos estan en Google Drive:

```bash
pip install gdown
gdown https://drive.google.com/uc?id=<FILE_ID> -O best.pt
```

### 6. Verificar que todo esta en su lugar

```bash
cd /workspace
ls -la
# Deberias ver:
# pipeline_yolo_sam.py
# best.pt
# sam_vit_h_4b8939.pth
# mi_video.mp4 (o tu carpeta de imagenes)
```

## Uso

### Probar con video

```bash
cd /workspace
python3 -c "
from pipeline_yolo_sam import run_pipeline_from_video

run_pipeline_from_video(
    video_path='mi_video.mp4',
    yolo_model_path='best.pt',
    sam_checkpoint='sam_vit_h_4b8939.pth',
    output_dir='./output',
    frame_interval=0.5,
    confidence_threshold=0.85,
    auto_approve_enabled=False,
)
"
```

### Probar con imagenes

```bash
python3 -c "
from pipeline_yolo_sam import run_pipeline_from_images

run_pipeline_from_images(
    image_dir='./mis_fotos',
    yolo_model_path='best.pt',
    sam_checkpoint='sam_vit_h_4b8939.pth',
    output_dir='./output',
    confidence_threshold=0.85,
    auto_approve_enabled=False,
)
"
```

## El flag auto_approve_enabled

Este es el interruptor de seguridad. Controla si el pipeline puede mandar imagenes directo al dataset sin revision humana.

| Valor | Comportamiento |
|-------|---------------|
| `False` (default) | TODO va a revision. No importa que YOLO diga 0.99, igual va a Label Studio para que lo revises. Usa esto mientras tu modelo esta subentrenado. |
| `True` | Se activa la separacion por confianza. Detecciones arriba del threshold van a auto-aprobadas, el resto a revision. Solo activar cuando confies en tu modelo. |

**Cuando activarlo:** Despues de al menos 2-3 ciclos de reentrenamiento donde veas que las pre-anotaciones son consistentemente buenas. No hay un numero magico, es juicio tuyo revisando resultados.

## Estructura de salida

```
output/
├── frames/                          # Frames extraidos del video
├── labels_approved/                 # Labels YOLO auto-aprobadas (vacio si auto_approve=False)
├── labels_review/                   # Labels YOLO para revision
├── label_studio_review_tasks.json   # Pre-anotaciones para Label Studio
└── label_studio_approved_tasks.json # Anotaciones auto-aprobadas en formato LS
```

## Importar en Label Studio

1. Crear proyecto en Label Studio
2. Settings > Labeling Interface > Code, pegar tu config con RectangleLabels
3. Import > Upload file > `label_studio_review_tasks.json`
4. Cada imagen aparece con las cajas ya dibujadas
5. Solo revisar, corregir si hace falta, y dar Submit

## Descargar resultados del pod

```bash
scp -r -P <puerto> root@<tu-pod-ip>:/workspace/output ./output_local
```

## Parametros

| Parametro | Default | Descripcion |
|-----------|---------|-------------|
| frame_interval | 0.5 | Segundos entre frames extraidos del video |
| confidence_threshold | 0.85 | Umbral para separar auto-aprobadas vs revision |
| auto_approve_enabled | False | Interruptor de auto-aprobacion |
| sam_model_type | vit_h | Variante de SAM (vit_h, vit_l, vit_b) |
| blur_threshold | 100.0 | Umbral de nitidez para descartar frames borrosos |

## Notas sobre VRAM

- SAM vit_h: ~7GB
- SAM vit_b: ~3GB
- YOLO: ~2GB adicional
- Total: 12GB+ para vit_h, 8GB+ para vit_b
