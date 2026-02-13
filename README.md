# Waste Detection Training Job

Job de entrenamiento incremental de **YOLOv8** para detección de desechos. Se ejecuta como un **Kubernetes Job** en GKE con GPU, consume las inferencias verificadas almacenadas por la [Inference API](#) y produce nuevas versiones del modelo.

## Clases

| ID | Clase         |
|----|---------------|
| 0  | Biodegradable |
| 1  | Cartón        |
| 2  | Vidrio        |
| 3  | Metal         |
| 4  | Papel         |
| 5  | Plástico      |

## Flujo de entrenamiento

```
┌──────────────────────────────────────────────────────────────────┐
│  kubectl apply -f training-job.yaml                              │
└──────────┬───────────────────────────────────────────────────────┘
           │
           ▼
   1. Consulta PostgreSQL
      ├─ ¿Hay trainings previos? → usa inferencias nuevas (incremental)
      └─ ¿Primer training?       → usa todas las verificadas
           │
           ▼
   2. Descarga imágenes + anotaciones YOLO desde GCS
      └─ Filtra por verificadas/corregidas si --only-verified
           │
           ▼
   3. Arma dataset YOLO (train/val split)
           │
           ▼
   4. Descarga modelo latest de GCS como base (fine-tuning)
           │
           ▼
   5. Entrena con YOLOv8 + early stopping
           │
           ▼
   6. Evalúa el modelo (mAP50, precision, recall)
           │
           ▼
   7. Sube modelo versionado a GCS
      └─ Actualiza models/latest/
           │
           ▼
   8. Registra versión y métricas en PostgreSQL
```

Después del training, reiniciar la API para que cargue el nuevo modelo:

```bash
kubectl rollout restart deployment/inference-api -n waste-detection
```

## Estructura del proyecto

```
├── train.py              # Servicio de entrenamiento (pipeline completo)
├── requirements.txt      # Dependencias Python
├── Dockerfile            # Multi-stage build (Python 3.10 + CUDA 12.1)
└── training-job.yaml     # Manifest de Kubernetes Job
```

## Modos de entrenamiento

### Desde inferencias (recomendado)

Usa las imágenes e inferencias que la API fue almacenando en GCS, filtradas por estado de verificación. Soporta entrenamiento incremental automático.

```bash
python train.py \
  --from-inferences \
  --only-verified \
  --epochs 50 \
  --batch-size 16 \
  --patience 10
```

### Desde dataset pre-estructurado

Usa un dataset YOLO estándar (local o en GCS).

```bash
python train.py \
  --dataset /path/to/dataset \
  --epochs 50 \
  --batch-size 16
```

## Argumentos CLI

| Argumento | Default | Descripción |
|-----------|---------|-------------|
| `--epochs` | `50` | Número de epochs |
| `--batch-size` | `16` | Tamaño del batch |
| `--learning-rate` | `0.01` | Learning rate inicial |
| `--img-size` | `640` | Tamaño de imagen de entrada |
| `--patience` | `10` | Early stopping patience |
| `--device` | auto | `cuda` o `cpu` (auto-detecta GPU) |
| `--base-model` | `yolov8n.pt` | Modelo base para fine-tuning |
| `--from-inferences` | `false` | Entrenar desde inferencias almacenadas en GCS |
| `--only-verified` | `false` | Solo usar inferencias verificadas o corregidas |
| `--all-time` | `false` | Ignorar fecha de último training (usar todo) |
| `--train-split` | `0.8` | Porcentaje de datos para entrenamiento |
| `--min-detections` | `1` | Mínimo de detecciones por imagen |
| `--dataset` | — | Ruta al dataset (local o `gs://...`) |

## Variables de entorno

| Variable | Descripción | Requerido |
|----------|-------------|-----------|
| `DATABASE_URL` | Connection string de PostgreSQL | Sí |
| `GCS_MODELS_BUCKET` | Bucket para modelos (.pt + metadata) | Sí |
| `GCS_IMAGES_BUCKET` | Bucket con imágenes e inferencias | Sí (modo `--from-inferences`) |
| `GCS_DATASETS_BUCKET` | Bucket con datasets pre-estructurados | Solo modo dataset |

## Docker

La imagen usa un **multi-stage build** para optimizar el tamaño: la etapa builder instala PyTorch con CUDA 12.1 y limpia artefactos de compilación antes de copiar al runtime.

```bash
# Build
docker build -t training .

# Run (requiere nvidia-docker)
docker run --gpus all \
  -e DATABASE_URL="postgresql://..." \
  -e GCS_MODELS_BUCKET="mi-bucket-modelos" \
  -e GCS_IMAGES_BUCKET="mi-bucket-imagenes" \
  training \
  --from-inferences --only-verified --epochs 50
```

## Despliegue en Kubernetes

El job se ejecuta con GPU (`nvidia.com/gpu: 1`) y usa el mismo namespace, secrets y service account que la Inference API, provisionados desde el repositorio de IaC.

```bash
kubectl create -f training-job.yaml
```

> Se usa `kubectl create` (no `apply`) porque el manifest usa `generateName` para nombres únicos por ejecución.

### Monitorear

```bash
# Ver logs en tiempo real
kubectl logs -f job/training-job-xxxxx -n waste-detection

# Ver estado
kubectl get jobs -n waste-detection
```

## Modelo base y fine-tuning

El job siempre intenta descargar `models/latest/yolov8n_waste.pt` de GCS como punto de partida. Si no existe (primer entrenamiento), usa `yolov8n.pt` preentrenado de Ultralytics. Cada entrenamiento posterior hace fine-tuning sobre la versión más reciente, acumulando el conocimiento de iteraciones anteriores.

## Stack tecnológico

- **YOLOv8** (Ultralytics) — entrenamiento y evaluación
- **PyTorch** + **CUDA 12.1** — backend de GPU
- **asyncpg** — consultas a PostgreSQL
- **Google Cloud Storage** — modelos, imágenes y datasets
- **Kubernetes Job** — orquestación en GKE con GPU