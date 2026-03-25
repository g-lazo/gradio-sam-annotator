# Anotador Gradio — Features Spec

## Objetivo

Agregar 6 features al anotador Gradio + SAM3 para hacerlo usable en producción.

## Features

### 1. Autosave

- Archivo `progress.json` en el `output_dir`
- Se guarda automáticamente en cada acción: anotar, deshacer, descartar, navegar
- Al iniciar, si existe `progress.json`, carga el estado y retoma (índice, anotaciones, descartadas)
- Formato: `{"index": int, "annotations": {path: [...]}, "discarded": [path, ...]}`

### 2. Skip/descartar imagen

- Botón "Descartar imagen" en la UI
- Mueve la imagen a `{output_dir}/_descartadas/`
- Se registra en el progreso (autosave)
- Al navegar se saltan automáticamente las descartadas
- El export las ignora

### 3. Contador mejorado

- Status bar: `Imagen 15/436 | Anotadas: 10 | Descartadas: 3 | Bboxes aquí: 2`
- Cuenta imágenes con al menos una anotación como "anotadas"
- Cuenta imágenes movidas a descartadas

### 4. Máscara con toggle

- `SAMBackend.segment()` retorna `(bbox, mask)` en vez de solo `bbox`
- Las máscaras se almacenan en `AnnotationSession` junto con las anotaciones
- Botón toggle "Mostrar máscaras" en la UI
- Cuando activo, `render_image` dibuja overlay semitransparente de la máscara sobre la imagen
- Cuando inactivo, solo dibuja bboxes (comportamiento actual)
- Las máscaras NO se guardan en `progress.json` (son pesadas, se regeneran si es necesario)

### 5. Export YOLO labels

- Además del Label Studio JSON, exporta en formato YOLO
- Directorio: `{output_dir}/labels/`
- Un archivo `.txt` por imagen: `{class_id} cx cy w h` (normalizado 0-1)
- El `class_id` se asigna por orden en la lista de clases del CLI

### 6. Filtrar basura del directorio

- Al cargar imágenes de `image_dir`, ignorar archivos/carpetas que empiecen con `.` o `_`
- Esto filtra: `.ipynb_checkpoints`, `_descartadas`, `.DS_Store`, etc.

## Cambios por componente

### SAMBackend

- `segment()` retorna `(bbox_xyxy, mask_ndarray)` o `(None, None)` si no detecta

### AnnotationSession

- Nuevos métodos: `save_progress(output_dir)`, `load_progress(output_dir)`, `discard_image(output_dir)`
- Las anotaciones almacenan la máscara opcionalmente: `(class_name, bbox, img_w, img_h, mask)`
- `discarded_paths: set[str]` para trackear imágenes descartadas
- Navegación salta imágenes descartadas
- `export()` agrega export YOLO además de Label Studio
- Constructor filtra archivos con prefijo `.` o `_`

### render_image

- Nuevo parámetro `show_masks: bool = False`
- Nuevo parámetro `masks: list[ndarray] = None`
- Cuando `show_masks=True`, dibuja overlay semitransparente por cada máscara

### build_ui

- Nuevo botón "Descartar imagen"
- Nuevo botón toggle "Mostrar máscaras"
- Status bar mejorado con contadores
- Autosave en cada handler

## Archivo

Todo en `files/annotator.py`. No se crean archivos nuevos.
