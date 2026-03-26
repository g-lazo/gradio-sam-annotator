# Gradio SAM Annotator

Anotador interactivo de imagenes con Gradio y SAM3. Click en un objeto para generar un bounding box automatico via segmentacion, asigna una clase, y exporta en formato Label Studio o YOLO.

## Como funciona

1. Cargas un directorio de imagenes
2. Haces click en el objeto que quieres anotar
3. SAM3 genera una mascara y extrae el bounding box
4. Seleccionas la clase del objeto
5. Exportas las anotaciones en formato Label Studio JSON o YOLO txt

## Requisitos

- Python 3.11+
- GPU con CUDA (para SAM3)

```bash
pip install gradio transformers torch opencv-python numpy Pillow
```

## Uso

```bash
python app.py \
  --image_dir ./mis_imagenes \
  --output_dir ./output \
  --classes "coca_cola,monster,sabritas" \
  --device cuda \
  --port 7860
```

### Argumentos

| Argumento | Default | Descripcion |
|-----------|---------|-------------|
| `--image_dir` | (requerido) | Directorio con las imagenes a anotar |
| `--output_dir` | `./output` | Directorio de salida para exports y progreso |
| `--classes` | `object` | Clases separadas por coma |
| `--sam_model` | `facebook/sam3` | Modelo SAM a usar |
| `--device` | `cuda` | Dispositivo (`cuda` o `cpu`) |
| `--port` | `7860` | Puerto del servidor Gradio |
| `--share` | off | Generar link publico de Gradio |

## Controles en la UI

- **Click en imagen** - Genera bbox con SAM3 en ese punto
- **Deshacer ultimo** - Quita la ultima anotacion
- **Descartar imagen** - Mueve la imagen a `_descartadas/` y la salta
- **Mostrar mascaras** - Toggle para ver las mascaras de SAM sobre la imagen
- **Anterior / Siguiente** - Navegar entre imagenes
- **Exportar** - Genera `label_studio_pack/` (JSON) y `labels/` (YOLO txt)

## Export

La exportacion genera dos formatos:

- **Label Studio**: `output/label_studio_pack/tasks.json` - listo para importar como pre-anotaciones
- **YOLO**: `output/labels/*.txt` - formato `class_id cx cy w h` normalizado

El progreso se guarda automaticamente en `output/progress.json`.