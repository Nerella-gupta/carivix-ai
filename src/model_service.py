"""Model loading and inference service for the CARIVIX prediction API."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger("CARIVIX_AI")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_MODELS_DIR = PROJECT_ROOT / "models"

REQUIRED_FIELDS = (
    "age",
    "income",
    "credit_score",
    "loan_amount",
    "years_employed",
    "education",
    "employment_status",
    "marital_status",
    "housing_type",
    "application_date",
)
NUMERIC_FIELDS = ("age", "income", "credit_score", "loan_amount", "years_employed")
CATEGORICAL_FIELDS = ("education", "employment_status", "marital_status", "housing_type")
BASE_FEATURES = NUMERIC_FIELDS + CATEGORICAL_FIELDS


class ModelService:
    """Discover persisted baseline models and serve validated feature rows."""

    def __init__(self, models_dir: Optional[str] = None, config_path: Optional[str] = None) -> None:
        self.project_root = PROJECT_ROOT
        self.models_dir = Path(models_dir) if models_dir is not None else DEFAULT_MODELS_DIR
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.load_errors: List[Dict[str, str]] = []
        self.default_model_name: Optional[str] = None
        self.default_model_path: Optional[str] = None
        self.config = self._load_config()
        self._reference_data = self._load_reference_data()
        self._discover_and_load_models()

    def _load_config(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with self.config_path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _load_reference_data(self) -> pd.DataFrame:
        dataset_path = self.project_root / self.config.get("dataset_path", "data/raw/dataset.csv")
        if dataset_path.exists():
            return pd.read_csv(dataset_path)

        baseline = self.config.get("baseline", {})
        processed_dir = self.project_root / baseline.get("processed_data_path", "data/processed")
        processed_files = sorted(processed_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
        if processed_files:
            return pd.read_csv(processed_files[0])
        raise FileNotFoundError("No reference dataset is available for inference preprocessing.")

    @staticmethod
    def _model_name(path: Path) -> str:
        stem = path.stem
        return stem.split("_processed_")[0].split("_dataset_")[0]

    @staticmethod
    def _model_type(model: Any) -> str:
        return "classification" if hasattr(model, "classes_") or hasattr(model, "predict_proba") else "regression"

    def _discover_and_load_models(self) -> None:
        if not self.models_dir.exists():
            return
        candidates = sorted(
            {path for pattern in ("*.pkl", "*.joblib", "*.sav", "*.model") for path in self.models_dir.rglob(pattern)},
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates and self.models_dir == DEFAULT_MODELS_DIR:
            experiments_file = self.project_root / "experiments" / "experiments.csv"
            if experiments_file.exists():
                with experiments_file.open("r", encoding="utf-8") as handle:
                    for row in csv.DictReader(handle):
                        path = (self.project_root / row.get("model_path", "")).resolve()
                        if path.exists():
                            candidates.append(path)

        for path in candidates:
            try:
                model = joblib.load(path)
            except Exception as exc:
                self.load_errors.append({"path": path.name, "error": str(exc)})
                logger.warning("Unable to load model %s: %s", path.name, exc)
                continue
            name = self._model_name(path)
            base_name = name
            suffix = 1
            while name in self.loaded_models:
                name = f"{base_name}_{suffix}"
                suffix += 1
            self.loaded_models[name] = {
                "model": model,
                "path": str(path),
                "algorithm": type(model).__name__,
                "type": self._model_type(model),
                "status": "loaded",
            }
            if self.default_model_name is None:
                self.default_model_name = name
                self.default_model_path = str(path)

    def reload_models(self) -> None:
        self.loaded_models.clear()
        self.load_errors.clear()
        self.default_model_name = None
        self.default_model_path = None
        self._discover_and_load_models()

    def has_loaded_models(self) -> bool:
        return bool(self.loaded_models)

    def get_load_errors(self) -> List[Dict[str, str]]:
        return list(self.load_errors)

    def list_models(self) -> List[Dict[str, str]]:
        return [{"name": name, "type": data["type"], "status": data["status"]} for name, data in self.loaded_models.items()]

    def get_model(self, model_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        name = model_name or self.default_model_name
        return self.loaded_models.get(name) if name else None

    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        record = self.get_model(model_name)
        if record is None:
            raise RuntimeError("No trained model is currently available.")
        return {
            "name": model_name or self.default_model_name,
            "type": record["type"],
            "task_type": record["type"],
            "algorithm": record["algorithm"],
            "status": record["status"],
            "supported_prediction_mode": record["type"],
        }

    def _prepare_features(self, input_data: Dict[str, Any], model: Any) -> pd.DataFrame:
        if not isinstance(input_data, dict):
            raise ValueError("Input data must be a dictionary of feature values.")
        missing = [field for field in REQUIRED_FIELDS if field not in input_data or input_data[field] is None]
        if missing:
            raise ValueError(f"Missing required fields: {', '.join(missing)}")

        row = dict(input_data)
        try:
            for field in NUMERIC_FIELDS:
                row[field] = float(row[field])
        except (TypeError, ValueError) as exc:
            raise ValueError("Numeric feature values must be valid numbers.") from exc
        for field in CATEGORICAL_FIELDS:
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{field} must be a non-empty string.")

        features = {}
        for field in NUMERIC_FIELDS:
            values = pd.to_numeric(self._reference_data[field], errors="coerce")
            mean, std = values.mean(), values.std(ddof=0) or 1.0
            features[field] = (row[field] - mean) / std
        for field in CATEGORICAL_FIELDS:
            values = self._reference_data[field].dropna().astype(str)
            categories = sorted(values.unique())
            code = categories.index(row[field]) if row[field] in categories else -1
            codes = values.map({category: index for index, category in enumerate(categories)})
            features[field] = (code - codes.mean()) / (codes.std(ddof=0) or 1.0)

        features["application_date_infrequent_sklearn"] = 1.0
        for index, left in enumerate(BASE_FEATURES):
            for right in BASE_FEATURES[index + 1:]:
                features[f"{left}_x_{right}"] = features[left] * features[right]

        expected = list(getattr(model, "feature_names_in_", ()))
        if not expected:
            raise RuntimeError("The loaded model does not expose feature metadata.")
        missing_model_features = [feature for feature in expected if feature not in features]
        if missing_model_features:
            raise ValueError("Input data cannot be transformed into the model feature set.")
        return pd.DataFrame([[features[feature] for feature in expected]], columns=expected)

    def predict(self, input_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        record = self.get_model(model_name)
        if record is None:
            raise RuntimeError("No trained model is currently available.")
        try:
            features = self._prepare_features(input_data, record["model"])
            model = record["model"]
            prediction = model.predict(features)[0]
            prediction_value = prediction.item() if hasattr(prediction, "item") else prediction
            result: Dict[str, Any] = {
                "success": True,
                "model": model_name or self.default_model_name,
                "prediction": prediction_value,
            }
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(features)[0]
                result["confidence"] = float(np.max(probabilities))
                if hasattr(model, "classes_"):
                    labels = list(model.classes_)
                    result["probability"] = float(probabilities[labels.index(prediction)])
            return result
        except ValueError:
            raise
        except Exception as exc:
            logger.exception("Model inference failed")
            raise RuntimeError("Model inference failed") from exc

    def predict_batch(self, records: List[Dict[str, Any]], model_name: Optional[str] = None) -> Dict[str, Any]:
        if not records:
            raise ValueError("At least one record is required for batch prediction.")
        selected = model_name or self.default_model_name
        predictions = [self.predict(record, selected)["prediction"] for record in records]
        return {"success": True, "model": selected, "predictions": predictions, "count": len(predictions)}


service = ModelService()

__all__ = ["ModelService", "service"]