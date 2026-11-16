"""HistGradientBoosting: estimador candidato de umbral orientado a precisión."""

from pathlib import Path

import yaml
from sklearn.ensemble import HistGradientBoostingClassifier


def crear_modelo() -> HistGradientBoostingClassifier:
    archivo = Path(__file__).with_name("hiperparametros.yaml")
    parametros = yaml.safe_load(archivo.read_text(encoding="utf-8"))
    return HistGradientBoostingClassifier(**parametros)
