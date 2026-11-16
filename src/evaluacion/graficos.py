"""Visualizaciones auditables para el holdout transversal de UCI."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.evaluacion.metricas import bins_calibracion, captura_top_20_eventos

PINO, DORADO = "#276653", "#c98c22"
PALETA = {"precision": PINO, "recall": DORADO, "combinacion": "#596c84"}


def guardar_figuras(
    y: np.ndarray,
    predicciones: dict[str, np.ndarray],
    lift: pd.DataFrame,
    importancia: pd.DataFrame,
    destino: Path,
    reglas: dict[str, np.ndarray],
) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for nombre, prob in predicciones.items():
        precision, recall, _ = precision_recall_curve(y, prob)
        ap = average_precision_score(y, prob)
        ax.plot(recall, precision, color=PALETA[nombre], lw=2, label=f"{nombre.title()} · AP={ap:.3f}")
    ax.axhline(np.mean(y), color="#919d94", ls="--", label="Tasa base")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Trade-off: precisión vs. recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=.14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino / "precision_recall.png", dpi=155)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#97a69b", label="Calibración ideal")
    for nombre, prob in predicciones.items():
        x, observado = bins_calibracion(y, prob)
        ax.plot(x, observado, "o-", ms=4, lw=2, color=PALETA[nombre], label=nombre.title())
    ax.set(xlabel="Probabilidad estimada", ylabel="Frecuencia observada", title="Calibración de riesgo (deciles, zoom 0–0,45)")
    ax.set_xlim(0, .45)
    ax.set_ylim(0, .45)
    ax.grid(alpha=.14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino / "calibracion.png", dpi=155)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for nombre in predicciones:
        datos = lift.loc[lift["modelo"].eq(nombre)]
        ax.plot(datos["decil"], datos["captura_acumulada"] * 100,
                marker="o", color=PALETA[nombre], lw=2, label=nombre.title())
    ax.plot([1, 10], [10, 100], "--", color="#97a69b", label="Selección aleatoria")
    ax.set(xlabel="Deciles contactados (1 = mayor riesgo)", ylabel="Eventos capturados acumulados (%)",
           title="Curva de ganancias · cohorte de prueba")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0, 105)
    ax.grid(alpha=.14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destino / "ganancias.png", dpi=155)
    plt.close(fig)

    top = importancia.sort_values("importancia_ap", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.9))
    ax.barh(top["variable"], top["importancia_ap"],
            color=[PINO if x >= 0 else DORADO for x in top["importancia_ap"]])
    ax.set(xlabel="Caída de AP al permutar (prueba)", title="Importancia por permutación · modelo de precisión")
    ax.grid(axis="x", alpha=.14)
    fig.tight_layout()
    fig.savefig(destino / "variables.png", dpi=155)
    plt.close(fig)

    nombres_reglas = {
        "utilizacion_descendente": "Utilización (regla)",
        "peor_atraso_6m_desempate_utilizacion": "Peor atraso previo (regla)",
    }
    valores = [(nombres_reglas[n], captura_top_20_eventos(y, p), "#9caaa2") for n, p in reglas.items()]
    referencia = max(v for _, v, _ in valores)
    valores.extend((n.title(), captura_top_20_eventos(y, p), PALETA[n]) for n, p in predicciones.items())
    fig, ax = plt.subplots(figsize=(9.5, 5.2), layout="constrained")
    barras = ax.barh([n for n, _, _ in valores], [v for _, v, _ in valores],
                     color=[c for _, _, c in valores], height=.65)
    ax.invert_yaxis()
    ax.axvline(referencia, color="#455e51", ls="--", label="Mejor regla sin ML")
    for i, (barra, (_, v, _)) in enumerate(zip(barras, valores)):
        nota = "" if i < len(reglas) else f" (Δ {v - referencia:+.2f} pp)"
        ax.text(v + .4, barra.get_y() + barra.get_height() / 2, f"{v:.2f}%{nota}", va="center", fontsize=9)
    ax.set(title="Eventos futuros capturados en top 20% · reglas vs. modelos",
           xlabel="% de defaults del mes siguiente capturados (no efecto causal)",
           xlim=(0, min(100, max(v for _, v, _ in valores) + 28)))
    ax.grid(axis="x", alpha=.15)
    ax.legend(loc="lower right")
    fig.savefig(destino / "valor_incremental.png", dpi=155)
    plt.close(fig)
