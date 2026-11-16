"""Combinación ilustrativa, fijada antes de evaluar el holdout."""

from pathlib import Path

import numpy as np
import yaml


def combinar(p_precision: np.ndarray, p_recall: np.ndarray) -> np.ndarray:
    parametros = yaml.safe_load(Path(__file__).with_name("hiperparametros.yaml").read_text(encoding="utf-8"))
    w1, w2 = float(parametros["peso_precision"]), float(parametros["peso_recall"])
    if not np.isclose(w1 + w2, 1.0) or min(w1, w2) < 0:
        raise ValueError("Los pesos deben sumar 1 y no pueden ser negativos")
    return w1 * np.asarray(p_precision) + w2 * np.asarray(p_recall)
