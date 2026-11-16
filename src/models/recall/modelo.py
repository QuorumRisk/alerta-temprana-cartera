"""Logística regularizada con normalización aprendida solo en desarrollo."""

from pathlib import Path

import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def crear_modelo():
    archivo = Path(__file__).with_name("hiperparametros.yaml")
    parametros = yaml.safe_load(archivo.read_text(encoding="utf-8"))
    return make_pipeline(StandardScaler(), LogisticRegression(**parametros))
