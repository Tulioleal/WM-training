"""
Training Service - Servicio de Entrenamiento de YOLOv8 para Detección de Desechos
"""

import os
import sys
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import yaml
import torch
import asyncio
import asyncpg
from ultralytics import YOLO

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingService:
    """
    Servicio para entrenar modelos YOLOv8 para detección de desechos.
    """
    
    # Clases de desechos
    CLASSES = [
        "plastico", "papel", "carton", "vidrio", "metal",
        "organico", "textil", "electronico", "peligroso", "otros"
    ]
    
    def __init__(
        self,
        base_model: str = "yolov8n.pt",
        data_dir: str = "/app/data",
        output_dir: str = "/app/output",
        models_bucket: Optional[str] = None,
        datasets_bucket: Optional[str] = None,
        images_bucket: Optional[str] = None,
        database_url: Optional[str] = None
    ):
        self.base_model = base_model
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.models_bucket = models_bucket
        self.datasets_bucket = datasets_bucket
        self.images_bucket = images_bucket
        self.database_url = database_url
        
        # Crear directorios
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Información del entrenamiento
        self.training_info: Dict[str, Any] = {}
        self.version: str = ""
        
        # Inicializar clientes cloud si están configurados
        self.storage_client = None
        self.db_pool = None
        
        if models_bucket:
            from google.cloud import storage
            self.storage_client = storage.Client()
    
    def generate_version(self) -> str:
        """Genera una versión única basada en timestamp"""
        return datetime.utcnow().strftime("v%Y%m%d_%H%M%S")
    
    def prepare_dataset(self, dataset_path: Optional[str] = None) -> Path:
        """
        Prepara el dataset para entrenamiento.
        """
        logger.info("Preparando dataset...")
        
        if dataset_path and dataset_path.startswith("gs://"):
            local_dataset = self._download_dataset(dataset_path)
        elif dataset_path:
            local_dataset = Path(dataset_path)
        else:
            local_dataset = self.data_dir / "waste_dataset"
        
        if not local_dataset.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {local_dataset}")
        
        data_yaml = local_dataset / "data.yaml"
        
        if not data_yaml.exists():
            data_config = {
                "path": str(local_dataset.absolute()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": len(self.CLASSES),
                "names": self.CLASSES
            }
            
            with open(data_yaml, 'w') as f:
                yaml.dump(data_config, f, default_flow_style=False)
            
            logger.info(f"Archivo data.yaml creado: {data_yaml}")
        
        train_images = list((local_dataset / "images" / "train").glob("*"))
        val_images = list((local_dataset / "images" / "val").glob("*"))
        
        self.training_info["train_images"] = len(train_images)
        self.training_info["val_images"] = len(val_images)
        
        logger.info(f"Dataset: {len(train_images)} train, {len(val_images)} val")
        
        return data_yaml
    
    def _download_dataset(self, gcs_path: str) -> Path:
        """Descarga dataset desde GCS"""
        logger.info(f"Descargando dataset desde {gcs_path}")
        
        path = gcs_path.replace("gs://", "")
        bucket_name, prefix = path.split("/", 1)
        
        bucket = self.storage_client.bucket(bucket_name)
        local_dir = self.data_dir / "downloaded_dataset"
        local_dir.mkdir(parents=True, exist_ok=True)
        
        blobs = bucket.list_blobs(prefix=prefix)
        
        for blob in blobs:
            relative_path = blob.name[len(prefix):].lstrip("/")
            if not relative_path:
                continue
            
            local_file = local_dir / relative_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file))
        
        return local_dir
    
    def _get_last_training_start(self) -> Optional[datetime]:
        """
        Consulta la DB para obtener el started_at del último entrenamiento completado.
        Si no hay entrenamientos previos, retorna None (usar todo).
        """
        if not self.database_url:
            return None
        
        async def _query():
            conn = await asyncpg.connect(self.database_url)
            try:
                return await conn.fetchval("""
                    SELECT started_at FROM trainings
                    WHERE status = 'completed'
                    ORDER BY started_at DESC
                    LIMIT 1
                """)
            finally:
                await conn.close()
        
        try:
            result = asyncio.run(_query())
            if result:
                logger.info(f"Último entrenamiento iniciado: {result}")
            else:
                logger.info("No hay entrenamientos previos — usando todas las inferencias")
            return result
        except Exception as e:
            logger.warning(f"Error consultando último entrenamiento: {e}")
            return None

    def _get_valid_request_ids(
        self,
        only_verified: bool = False,
        since: Optional[datetime] = None
    ) -> list:
        """
        Consulta la DB para obtener los request_ids de inferencias válidas.
        Si la DB no está disponible, intenta leer el archivo exportado de GCS.
        
        Args:
            only_verified: Solo incluir verified + corrected
            since: Solo incluir inferencias posteriores a esta fecha
            
        Returns:
            Lista de dicts con request_id e image_url
        """
        # Intento 1: Consultar DB directamente
        if self.database_url:
            try:
                results = self._get_valid_request_ids_from_db(only_verified, since)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Error consultando DB: {e}")
        
        # Intento 2: Leer JSON exportado de GCS
        if self.storage_client and self.images_bucket:
            try:
                results = self._get_valid_request_ids_from_gcs()
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Error leyendo export de GCS: {e}")
        
        logger.warning("Sin DB ni export de GCS — no se puede filtrar")
        return []

    def _get_valid_request_ids_from_gcs(self) -> list:
        """
        Lee los request_ids desde el JSON exportado en GCS.
        Archivo: gs://{images_bucket}/training_exports/verified_requests.json
        """
        bucket = self.storage_client.bucket(self.images_bucket)
        blob = bucket.blob("training_exports/verified_requests.json")
        
        if not blob.exists():
            logger.warning("No existe archivo de export en GCS")
            return []
        
        content = blob.download_as_string().decode('utf-8')
        export_data = json.loads(content)
        
        logger.info(
            f"Leídos {export_data['total_records']} registros del export de GCS "
            f"(exportado: {export_data['exported_at']})"
        )
        
        return [
            {"request_id": d["request_id"], "image_url": d.get("image_url")}
            for d in export_data.get("data", [])
        ]

    def _get_valid_request_ids_from_db(
        self,
        only_verified: bool = False,
        since: Optional[datetime] = None
    ) -> list:
        """Consulta la DB directamente para obtener request_ids válidos."""
        async def _query():
            conn = await asyncpg.connect(self.database_url)
            try:
                conditions = ["image_url IS NOT NULL"]
                params = []
                param_idx = 1
                
                if only_verified:
                    conditions.append("verification_status IN ('verified', 'corrected')")
                
                if since:
                    conditions.append(f"timestamp > ${param_idx}")
                    params.append(since)
                    param_idx += 1
                
                where_clause = " AND ".join(conditions)
                
                rows = await conn.fetch(f"""
                    SELECT request_id, image_url, detection_count
                    FROM inferences
                    WHERE {where_clause}
                    ORDER BY timestamp ASC
                """, *params)
                
                return [dict(r) for r in rows]
            finally:
                await conn.close()
        
        results = asyncio.run(_query())
        logger.info(f"Inferencias válidas desde DB: {len(results)}")
        return results

    def prepare_dataset_from_inferences(
        self, 
        images_bucket: str,
        train_split: float = 0.8,
        min_detections: int = 1,
        only_verified: bool = False,
        since_last_training: bool = False
    ) -> Path:
        """
        Prepara un dataset YOLO a partir de las inferencias guardadas.
        
        Args:
            images_bucket: Nombre del bucket de imágenes
            train_split: Porcentaje para entrenamiento (0.8 = 80%)
            min_detections: Mínimo de detecciones por imagen
            only_verified: Solo usar inferencias verificadas/corregidas
            since_last_training: Solo usar inferencias posteriores al último training
            
        Returns:
            Path al archivo data.yaml
        """
        import random
        
        # Determinar fecha de corte
        since_date = None
        if since_last_training:
            since_date = self._get_last_training_start()
        
        # Obtener request_ids válidos desde la DB
        valid_records = self._get_valid_request_ids(
            only_verified=only_verified,
            since=since_date
        )
        
        if valid_records:
            valid_request_ids = {r['request_id'] for r in valid_records}
            logger.info(
                f"Filtrado DB: {len(valid_request_ids)} inferencias válidas"
                f"{' (solo verificadas)' if only_verified else ''}"
                f"{f' (desde {since_date})' if since_date else ''}"
            )
        else:
            valid_request_ids = None  # Sin DB, no filtrar
            logger.warning("Sin filtrado de DB — descargando todo del bucket")
        
        logger.info(f"Preparando dataset desde inferencias en gs://{images_bucket}/inferences/")
        
        # Crear estructura de directorios YOLO
        dataset_dir = self.data_dir / "inference_dataset"
        (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)
        
        # Obtener lista de archivos de inferencia
        bucket = self.storage_client.bucket(images_bucket)
        blobs = list(bucket.list_blobs(prefix="inferences/"))
        
        # Agrupar por request_id (imagen + anotación)
        inference_pairs = {}
        for blob in blobs:
            if blob.name.endswith('.txt') or blob.name.endswith('.jpeg') or blob.name.endswith('.jpg') or blob.name.endswith('.png'):
                filename = blob.name.split('/')[-1]
                request_id = filename.rsplit('.', 1)[0]
                
                # Filtrar por request_ids válidos si tenemos la lista
                if valid_request_ids is not None and request_id not in valid_request_ids:
                    continue
                
                if request_id not in inference_pairs:
                    inference_pairs[request_id] = {}
                
                if blob.name.endswith('.txt'):
                    inference_pairs[request_id]['annotation'] = blob
                else:
                    inference_pairs[request_id]['image'] = blob
        
        # Filtrar solo pares completos (imagen + anotación)
        complete_pairs = [
            (rid, data) for rid, data in inference_pairs.items()
            if 'image' in data and 'annotation' in data
        ]
        
        logger.info(f"Encontrados {len(complete_pairs)} pares imagen+anotación")
        
        if len(complete_pairs) == 0:
            raise ValueError("No se encontraron pares completos de imagen+anotación")
        
        # Mezclar aleatoriamente
        random.shuffle(complete_pairs)
        
        # Dividir en train/val
        split_idx = int(len(complete_pairs) * train_split)
        train_pairs = complete_pairs[:split_idx]
        val_pairs = complete_pairs[split_idx:]
        
        logger.info(f"División: {len(train_pairs)} train, {len(val_pairs)} val")
        
        # Descargar y organizar archivos
        downloaded_train = 0
        downloaded_val = 0
        
        for request_id, data in train_pairs:
            if self._download_inference_pair(data, dataset_dir, "train", min_detections):
                downloaded_train += 1
        
        for request_id, data in val_pairs:
            if self._download_inference_pair(data, dataset_dir, "val", min_detections):
                downloaded_val += 1
        
        logger.info(f"Descargados: {downloaded_train} train, {downloaded_val} val")
        
        if downloaded_train == 0:
            raise ValueError("No se descargaron imágenes de entrenamiento")
        
        self.training_info["train_images"] = downloaded_train
        self.training_info["val_images"] = downloaded_val
        self.training_info["source"] = "inferences"
        self.training_info["only_verified"] = only_verified
        self.training_info["since_last_training"] = since_last_training
        self.training_info["cutoff_date"] = since_date.isoformat() if since_date else None
        
        # Crear data.yaml
        data_yaml = dataset_dir / "data.yaml"
        data_config = {
            "path": str(dataset_dir.absolute()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(self.CLASSES),
            "names": self.CLASSES
        }
        
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Dataset preparado: {data_yaml}")
        
        return data_yaml
    
    def _download_inference_pair(
        self, 
        data: dict, 
        dataset_dir: Path, 
        split: str,
        min_detections: int
    ) -> bool:
        """
        Descarga un par imagen+anotación.
        
        Returns:
            True si se descargó correctamente, False si no cumple criterios
        """
        try:
            image_blob = data['image']
            annotation_blob = data['annotation']
            
            # Verificar que la anotación tenga suficientes detecciones
            annotation_content = annotation_blob.download_as_string().decode('utf-8').strip()
            num_detections = len(annotation_content.split('\n')) if annotation_content else 0
            
            if num_detections < min_detections:
                return False
            
            # Determinar nombre de archivo
            image_filename = image_blob.name.split('/')[-1]
            base_name = image_filename.rsplit('.', 1)[0]
            image_ext = image_filename.rsplit('.', 1)[1]
            
            # Descargar imagen
            image_path = dataset_dir / "images" / split / f"{base_name}.{image_ext}"
            image_blob.download_to_filename(str(image_path))
            
            # Descargar anotación
            label_path = dataset_dir / "labels" / split / f"{base_name}.txt"
            with open(label_path, 'w') as f:
                f.write(annotation_content)
            
            return True
            
        except Exception as e:
            logger.warning(f"Error descargando par: {e}")
            return False
    
    def _get_base_model(self) -> str:
        """
        Obtiene el modelo base para fine-tuning.
        Intenta descargar el modelo 'latest' de GCS.
        Si no existe, usa el modelo base yolov8n.pt.
        """
        if not self.storage_client or not self.models_bucket:
            logger.info(f"GCS no configurado, usando modelo base: {self.base_model}")
            return self.base_model
        
        try:
            bucket = self.storage_client.bucket(self.models_bucket)
            blob = bucket.blob("models/latest/yolov8n_waste.pt")
            
            if blob.exists():
                local_path = self.data_dir / "base_model.pt"
                blob.download_to_filename(str(local_path))
                logger.info(f"Modelo base descargado de GCS: gs://{self.models_bucket}/models/latest/yolov8n_waste.pt")
                return str(local_path)
            else:
                logger.info(f"No existe modelo en GCS, usando modelo base: {self.base_model}")
                return self.base_model
                
        except Exception as e:
            logger.warning(f"Error descargando modelo de GCS: {e}. Usando modelo base: {self.base_model}")
            return self.base_model

    def train(
        self,
        data_yaml: Path,
        epochs: int = 50,
        batch_size: int = 16,
        learning_rate: float = 0.01,
        img_size: int = 640,
        patience: int = 10,
        device: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Ejecuta el entrenamiento del modelo.
        """
        self.version = self.generate_version()
        logger.info(f"Iniciando entrenamiento - Versión: {self.version}")
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Dispositivo: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.training_info.update({
            "version": self.version,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "img_size": img_size,
            "device": device,
            "base_model": self.base_model
        })
        
        os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

        # Obtener modelo base para fine-tuning
        base_model_path = self._get_base_model()
        self.training_info["base_model"] = base_model_path
        
        model = YOLO(base_model_path)
        run_dir = self.output_dir / self.version
        
        start_time = time.time()
        
        try:
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=learning_rate,
                patience=patience,
                device=device,
                project=str(self.output_dir),
                name=self.version,
                exist_ok=True,
                verbose=True,
                plots=True,
                save=True,
                **kwargs
            )
            
            training_time = time.time() - start_time
            
            metrics = self._extract_metrics(results, run_dir)
            metrics["training_time_seconds"] = round(training_time, 2)
            
            self.training_info.update(metrics)
            self.training_info["status"] = "completed"
            
            logger.info(f"Entrenamiento completado en {training_time:.2f}s")
            logger.info(f"Métricas: mAP50={metrics.get('map50', 'N/A')}")
            
            return self.training_info
            
        except Exception as e:
            self.training_info["status"] = "failed"
            self.training_info["error"] = str(e)
            logger.error(f"Error en entrenamiento: {e}")
            raise
    
    def _extract_metrics(self, results, run_dir: Path) -> Dict[str, Any]:
        """Extrae métricas del entrenamiento"""
        metrics = {}
        
        if hasattr(results, 'results_dict'):
            rd = results.results_dict
            metrics["map50"] = rd.get("metrics/mAP50(B)", None)
            metrics["map50_95"] = rd.get("metrics/mAP50-95(B)", None)
            metrics["precision"] = rd.get("metrics/precision(B)", None)
            metrics["recall"] = rd.get("metrics/recall(B)", None)
        
        results_csv = run_dir / "results.csv"
        if results_csv.exists():
            import pandas as pd
            df = pd.read_csv(results_csv)
            if not df.empty:
                last_row = df.iloc[-1]
                for col in df.columns:
                    col_clean = col.strip()
                    if 'mAP50' in col_clean and 'mAP50-95' not in col_clean:
                        metrics["map50"] = metrics.get("map50") or float(last_row[col])
                    elif 'mAP50-95' in col_clean:
                        metrics["map50_95"] = metrics.get("map50_95") or float(last_row[col])
        
        return metrics
    
    def evaluate(self, data_yaml: Path) -> Dict[str, Any]:
        """Evalúa el modelo entrenado"""
        logger.info("Evaluando modelo...")
        
        best_model_path = self.output_dir / self.version / "weights" / "best.pt"
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {best_model_path}")
        
        model = YOLO(str(best_model_path))
        
        results = model.val(data=str(data_yaml), verbose=True)
        
        eval_metrics = {
            "map50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
            "map50_95": float(results.box.map) if hasattr(results.box, 'map') else None,
            "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
            "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None,
        }
        
        self.training_info["evaluation"] = eval_metrics
        
        return eval_metrics
    
    def save_model(self) -> str:
        """Guarda el modelo y metadata en GCS"""
        logger.info("Guardando modelo...")
        
        best_model_path = self.output_dir / self.version / "weights" / "best.pt"
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"Modelo no encontrado: {best_model_path}")
        
        if self.storage_client and self.models_bucket:
            gcs_path = self._upload_to_gcs(best_model_path)
            self._upload_metadata()
            return gcs_path
        else:
            final_path = self.output_dir / f"yolov8n_waste_{self.version}.pt"
            import shutil
            shutil.copy(best_model_path, final_path)
            self._save_metadata_local()
            return str(final_path)
    
    def _upload_to_gcs(self, model_path: Path) -> str:
        """Sube modelo a GCS"""
        bucket = self.storage_client.bucket(self.models_bucket)
        
        gcs_model_path = f"models/{self.version}/yolov8n_waste.pt"
        blob = bucket.blob(gcs_model_path)
        blob.upload_from_filename(str(model_path))
        
        logger.info(f"Modelo subido: gs://{self.models_bucket}/{gcs_model_path}")
        
        return f"gs://{self.models_bucket}/{gcs_model_path}"
    
    def _upload_metadata(self):
        """Sube metadata a GCS"""
        bucket = self.storage_client.bucket(self.models_bucket)
        
        metadata = {
            "version": self.version,
            "created_at": datetime.utcnow().isoformat(),
            "training_info": self.training_info,
            "classes": self.CLASSES
        }
        
        gcs_path = f"models/{self.version}/metadata.json"
        blob = bucket.blob(gcs_path)
        blob.upload_from_string(
            json.dumps(metadata, indent=2, default=str),
            content_type="application/json"
        )
        
        logger.info(f"Metadata subida: gs://{self.models_bucket}/{gcs_path}")
    
    def _update_latest_model(self):
        """
        Actualiza el modelo 'latest' copiando el modelo recién entrenado.
        Esto permite que la API siempre use el modelo más reciente.
        """
        logger.info("Actualizando modelo 'latest'...")
        
        bucket = self.storage_client.bucket(self.models_bucket)
        
        # Copiar modelo a latest/
        source_blob = bucket.blob(f"models/{self.version}/yolov8n_waste.pt")
        dest_blob = bucket.blob("models/latest/yolov8n_waste.pt")
        
        # Rewrite (copia dentro del mismo bucket)
        dest_blob.rewrite(source_blob)
        
        # Copiar metadata a latest/
        metadata = {
            "version": self.version,
            "created_at": datetime.utcnow().isoformat(),
            "training_info": self.training_info,
            "classes": self.CLASSES,
            "source_version": self.version
        }
        
        metadata_blob = bucket.blob("models/latest/metadata.json")
        metadata_blob.upload_from_string(
            json.dumps(metadata, indent=2, default=str),
            content_type="application/json"
        )
        
        logger.info(f"Modelo 'latest' actualizado desde {self.version}")
    
    def _save_metadata_local(self):
        """Guarda metadata localmente"""
        metadata = {
            "version": self.version,
            "created_at": datetime.utcnow().isoformat(),
            "training_info": self.training_info,
            "classes": self.CLASSES
        }
        
        metadata_path = self.output_dir / self.version / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dumps(metadata, f, indent=2, default=str)
        
        logger.info(f"Metadata guardada: {metadata_path}")
    
    def register_in_database(self):
        """Registra el modelo en la base de datos"""
        if not self.database_url:
            logger.warning("Database URL no configurada, saltando registro")
            return
        
        import asyncio
        import asyncpg
        
        async def _register():
            conn = await asyncpg.connect(self.database_url)
            try:
                await conn.execute("""
                    INSERT INTO models 
                    (version, created_at, accuracy, map50, map50_95, 
                     training_epochs, training_time_seconds, dataset_size, notes, gcs_path)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                    ON CONFLICT (version) DO UPDATE SET
                        accuracy = EXCLUDED.accuracy,
                        map50 = EXCLUDED.map50,
                        map50_95 = EXCLUDED.map50_95
                """,
                    self.version,
                    datetime.utcnow(),
                    self.training_info.get("precision"),
                    self.training_info.get("map50"),
                    self.training_info.get("map50_95"),
                    self.training_info.get("epochs"),
                    self.training_info.get("training_time_seconds"),
                    self.training_info.get("train_images", 0) + self.training_info.get("val_images", 0),
                    json.dumps(self.training_info),
                    self.training_info.get("gcs_path")
                )
                
                await conn.execute("""
                    INSERT INTO trainings 
                    (job_id, started_at, finished_at, status, model_version,
                     epochs, batch_size, learning_rate, final_accuracy, final_map50)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                """,
                    f"training-{self.version}",
                    datetime.utcnow(),
                    datetime.utcnow(),
                    self.training_info.get("status", "completed"),
                    self.version,
                    self.training_info.get("epochs"),
                    self.training_info.get("batch_size"),
                    self.training_info.get("learning_rate"),
                    self.training_info.get("precision"),
                    self.training_info.get("map50")
                )
                
                logger.info(f"Modelo registrado en base de datos: {self.version}")
                
            finally:
                await conn.close()
        
        asyncio.run(_register())


def main():
    """Punto de entrada principal"""
    parser = argparse.ArgumentParser(description="Servicio de Entrenamiento YOLOv8")
    
    parser.add_argument("--epochs", type=int, default=50, help="Número de epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Tamaño del batch")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=640, help="Tamaño de imagen")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--dataset", type=str, help="Ruta al dataset (local o gs://)")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt", help="Modelo base")
    parser.add_argument("--device", type=str, help="Dispositivo (cuda/cpu)")
    
    # Nuevos argumentos para entrenamiento desde inferencias
    parser.add_argument("--from-inferences", action="store_true", 
                        help="Entrenar usando las inferencias guardadas automáticamente")
    parser.add_argument("--only-verified", action="store_true",
                        help="Solo usar inferencias verificadas o corregidas")
    parser.add_argument("--all-time", action="store_true",
                        help="Ignorar fecha de último entrenamiento y usar todas las inferencias")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Porcentaje de datos para entrenamiento (default: 0.8)")
    parser.add_argument("--min-detections", type=int, default=1,
                        help="Mínimo de detecciones por imagen para incluirla (default: 1)")
    
    args = parser.parse_args()
    
    # Configuración desde variables de entorno
    models_bucket = os.environ.get("GCS_MODELS_BUCKET")
    datasets_bucket = os.environ.get("GCS_DATASETS_BUCKET")
    images_bucket = os.environ.get("GCS_IMAGES_BUCKET")  # Nuevo: bucket de imágenes
    database_url = os.environ.get("DATABASE_URL")
    
    # Crear servicio
    service = TrainingService(
        base_model=args.base_model,
        models_bucket=models_bucket,
        datasets_bucket=datasets_bucket,
        images_bucket=images_bucket,
        database_url=database_url
    )
    
    try:
        # 1. Preparar dataset
        if args.from_inferences:
            # Entrenar desde inferencias guardadas automáticamente
            if not images_bucket:
                raise ValueError("GCS_IMAGES_BUCKET es requerido para --from-inferences")
            
            logger.info("="*60)
            logger.info("MODO: Entrenamiento desde inferencias")
            logger.info(f"Bucket de imágenes: {images_bucket}")
            logger.info(f"Solo verificadas: {args.only_verified}")
            logger.info(f"Incremental: {not args.all_time}")
            logger.info(f"Train split: {args.train_split}")
            logger.info(f"Min detecciones: {args.min_detections}")
            logger.info("="*60)
            
            data_yaml = service.prepare_dataset_from_inferences(
                images_bucket=images_bucket,
                train_split=args.train_split,
                min_detections=args.min_detections,
                only_verified=args.only_verified,
                since_last_training=not args.all_time
            )
        else:
            # Modo tradicional: dataset pre-estructurado
            data_yaml = service.prepare_dataset(args.dataset)
        
        # 2. Entrenar modelo
        service.train(
            data_yaml=data_yaml,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            img_size=args.img_size,
            patience=args.patience,
            device=args.device
        )
        
        # 3. Evaluar modelo
        service.evaluate(data_yaml)
        
        # 4. Guardar modelo
        model_path = service.save_model()
        
        # 5. Actualizar modelo "latest" en GCS
        if service.storage_client and models_bucket:
            service._update_latest_model()
        
        # 6. Registrar en base de datos
        service.register_in_database()
        
        logger.info(f"Pipeline completado exitosamente. Modelo: {model_path}")
        
        # Imprimir resumen
        print("\n" + "="*60)
        print("RESUMEN DEL ENTRENAMIENTO")
        print("="*60)
        print(json.dumps(service.training_info, indent=2, default=str))
        print("="*60)
        
    except Exception as e:
        logger.error(f"Error en pipeline de entrenamiento: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()