"""Evaluación transversal, umbrales OOF y reglas de priorización sin ML."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve,
)


def ks_statistic(y: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, prob)
    return float(np.max(tpr - fpr))


def elegir_umbral(y: np.ndarray, prob: np.ndarray, objetivo: str) -> dict:
    """Escoger en desarrollo OOF exclusivamente, nunca sobre el holdout."""
    tasa_base = float(np.mean(y))
    cuadricula = np.unique(np.quantile(prob, np.linspace(0.03, 0.98, 191)))
    opciones = []
    for umbral in cuadricula:
        pred = prob >= umbral
        if not pred.any():
            continue
        opciones.append({
            "threshold": float(umbral),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "contact_rate": float(pred.mean()),
        })
    if objetivo == "precision":
        elegibles = [o for o in opciones if o["contact_rate"] >= 0.05 and o["recall"] >= 0.12]
        clave = lambda o: (o["precision"], o["recall"])
    elif objetivo == "recall":
        elegibles = [o for o in opciones if o["precision"] >= tasa_base + 0.08 and o["contact_rate"] <= 0.45]
        clave = lambda o: (o["recall"], o["precision"])
    elif objetivo == "f1":
        elegibles = [o for o in opciones if o["contact_rate"] <= 0.40]
        clave = lambda o: (o["f1"], o["precision"])
    else:
        raise ValueError(f"Objetivo de umbral desconocido: {objetivo}")
    if not elegibles:
        raise ValueError(f"No se encontró umbral viable para {objetivo}")
    return max(elegibles, key=clave)


def tabla_ganancias(y: np.ndarray, prob: np.ndarray, modelo: str) -> pd.DataFrame:
    indices = np.argsort(-prob, kind="stable")
    ordenadas = np.asarray(y)[indices]
    total = ordenadas.sum()
    n = len(ordenadas)
    filas = []
    for numero, grupo in enumerate(np.array_split(np.arange(n), 10), start=1):
        eventos = int(ordenadas[grupo].sum())
        filas.append({
            "modelo": modelo,
            "decil": numero,
            "clientes": len(grupo),
            "eventos": eventos,
            "tasa_evento": float(eventos / len(grupo)),
            "lift": float((eventos / len(grupo)) / (total / n)),
            "captura_acumulada": float(ordenadas[: grupo[-1] + 1].sum() / total),
        })
    return pd.DataFrame(filas)


def metricas_modelo(y: np.ndarray, prob: np.ndarray, umbral: float, nombre: str) -> dict:
    pred = prob >= umbral
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "modelo": nombre,
        "umbral": round(float(umbral), 5),
        "roc_auc": round(float(roc_auc_score(y, prob)), 5),
        "pr_auc_average_precision": round(float(average_precision_score(y, prob)), 5),
        "ks": round(ks_statistic(y, prob), 5),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 5),
        "f1": round(float(f1_score(y, pred, zero_division=0)), 5),
        "brier": round(float(brier_score_loss(y, prob)), 5),
        "contact_rate": round(float(pred.mean()), 5),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def bins_calibracion(y: np.ndarray, prob: np.ndarray) -> tuple[list[float], list[float]]:
    tabla = pd.DataFrame({"y": y, "p": prob})
    tabla["bin"] = pd.qcut(tabla["p"], q=10, duplicates="drop")
    agrupado = tabla.groupby("bin", observed=True).agg(observed=("y", "mean"), predicted=("p", "mean"))
    return agrupado["predicted"].tolist(), agrupado["observed"].tolist()


def reglas_sin_ml(variables: pd.DataFrame) -> dict[str, np.ndarray]:
    """Ranking sin etiquetas. Reglas fijadas ANTES del test.

    Mayor utilización; o mayor código positivo de atraso histórico y,
    en empate, utilización (0..5) como criterio secundario.
    """
    uso = variables["utilizacion_reciente"].to_numpy(dtype=float)
    atraso = variables["max_atraso_6m"].to_numpy(dtype=float)
    if not np.isfinite(uso).all() or not np.isfinite(atraso).all():
        raise ValueError("Las reglas requieren predictores observados y finitos")
    return {
        "utilizacion_descendente": uso,
        "peor_atraso_6m_desempate_utilizacion": 10.0 * atraso + uso,
    }


def captura_top_20_eventos(y: np.ndarray, prioridad: np.ndarray) -> float:
    """% de eventos capturados en el 20% de cuentas; NO efecto causal."""
    y = np.asarray(y)
    orden = np.asarray(prioridad, dtype=float)
    if len(y) != len(orden) or not np.isfinite(orden).all():
        raise ValueError("Longitud incompatible o prioridad inválida")
    if not len(y) or y.sum() == 0:
        raise ValueError("La captura requiere cuentas y al menos un evento")
    n = int(np.ceil(len(y) * 0.20))
    indices = np.argsort(-orden, kind="stable")[:n]
    return round(float(100 * y[indices].sum() / y.sum()), 6)
