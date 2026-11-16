"""Demostración pública y reproducible de alerta temprana de cartera.

Código original de Quórum Risk; dataset UCI #350, CC BY 4.0.
Ejecutar: python run_demo.py
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SOURCE_DOI = "https://doi.org/10.24432/C55S3H"
RANDOM_STATE = 42
GREEN = "#276653"
GOLD = "#c98c22"
NAVY = "#203e36"
PALETTE = {"precision": GREEN, "recall": GOLD, "combinacion": "#596c84"}
TARGET = "default payment next month"
STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
REQUIRED = ["ID", "LIMIT_BAL", TARGET, *STATUS_COLS, *BILL_COLS, *PAYMENT_COLS]

DESCRIPTIONS = {
    "max_atraso_6m": "Mayor código de atraso positivo en los seis meses",
    "atrasos_6m": "Meses con código de atraso positivo",
    "atrasos_3m": "Meses recientes con código de atraso positivo",
    "recencia_ultimo_atraso": "Meses transcurridos desde el último atraso positivo (7 si ninguno)",
    "tendencia_atrasos": "Atrasos recientes menos atrasos antiguos",
    "severidad_reciente": "Mayor severidad positiva de los últimos tres meses",
    "persistencia_atraso": "Racha de atrasos positivos hacia el corte",
    "meses_estado_no_positivo": "Meses con código 0, -1 o -2 (semántica parcial)",
    "limite_log": "Logaritmo del límite de crédito",
    "utilizacion_reciente": "Saldo positivo de septiembre dividido por cupo",
    "utilizacion_media": "Utilización promedio de seis cortes",
    "utilizacion_max": "Pico de utilización en seis cortes",
    "utilizacion_volatilidad": "Desviación típica de la utilización",
    "cambio_utilizacion": "Utilización reciente menos la de abril",
    "tendencia_utilizacion": "Pendiente mensual estimada de utilización",
    "meses_sobre_cupo": "Meses con saldo reportado superior al cupo",
    "importe_saldo_reciente_log": "Logaritmo del saldo positivo de septiembre",
    "pago_reciente_log": "Logaritmo del pago positivo de septiembre",
    "ratio_pago_saldo_agregado": "Suma de pagos sobre suma de saldos positivos",
    "ratio_pago_saldo_reciente": "Pago reciente sobre saldo positivo reciente",
    "ratio_pago_saldo_mediano": "Mediana de ratios mensuales pago/saldo",
    "meses_sin_pago": "Meses sin pago reportado",
    "volatilidad_pago": "Volatilidad de pagos normalizada por cupo",
    "tendencia_pago": "Pendiente mensual de pagos normalizados por cupo",
    "pago_reciente_vs_historico": "Pago reciente sobre pago medio de meses anteriores",
}


def download_data(cache_dir: Path) -> pd.DataFrame:
    """Descarga el ZIP oficial UCI y lo cachea; no redistribuye datos en GitHub."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / "uci-350.zip"
    if not archive.exists():
        print(f"Descargando dataset UCI: {SOURCE_URL}")
        req = Request(SOURCE_URL, headers={"User-Agent": "QuorumRisk-public-reproducible-demo/1.0"})
        with urlopen(req, timeout=90) as response:
            data = response.read()
        with zipfile.ZipFile(io.BytesIO(data)) as test_zip:
            if not any(name.endswith(".xls") for name in test_zip.namelist()):
                raise ValueError("El archivo UCI no contiene el XLS esperado")
        archive.write_bytes(data)
    with zipfile.ZipFile(archive) as source:
        filename = next((x for x in source.namelist() if x.endswith(".xls")), None)
        if filename is None:
            raise ValueError("No se encontró el archivo XLS en la descarga UCI")
        df = pd.read_excel(io.BytesIO(source.read(filename)), header=1, engine="xlrd")
    if set(REQUIRED).difference(df.columns):
        raise ValueError(f"Columnas faltantes: {sorted(set(REQUIRED).difference(df.columns))}")
    return df


def select_cohort(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Observación: abril-septiembre 2005. Outcome UCI: siguiente mes.

    Cohorte preventiva: no hay código POSITIVO de atraso en septiembre.
    No se interpreta PAY_0 <= 0 como 'cliente al día' sin matices:
    los códigos -2/0 no están plenamente definidos por la ficha oficial.
    """
    n_total = len(df)
    valid_target = df[TARGET].isin([0, 1])
    valid_limit = df["LIMIT_BAL"].notna() & (df["LIMIT_BAL"] > 0)
    valid_history = df[[*STATUS_COLS, *BILL_COLS, *PAYMENT_COLS]].notna().all(axis=1)
    valid = valid_target & valid_limit & valid_history
    current_delay = df["PAY_0"].gt(0)
    cohort = df.loc[valid & ~current_delay].copy()
    if cohort.empty or cohort[TARGET].nunique() != 2:
        raise ValueError("La cohorte preventiva no permite clasificación binaria")
    audit = {
        "total_filas_fuente": n_total,
        "filas_target_indeterminado_o_invalido": int((~valid_target).sum()),
        "filas_cupo_o_historial_incompleto": int((valid_target & (~valid_limit | ~valid_history)).sum()),
        "filas_atraso_positivo_actual_excluidas": int((valid & current_delay).sum()),
        "filas_cohorte_preventiva": len(cohort),
        "eventos_cohorte": int(cohort[TARGET].sum()),
        "tasa_evento_cohorte": float(cohort[TARGET].mean()),
        "nota": "PAY_0 <= 0 significa sin atraso POSITIVO registrado; no garantiza ausencia de riesgos ni pagos puntuales.",
    }
    return cohort, audit


def row_slope(values: np.ndarray) -> np.ndarray:
    """Pendiente lineal desde abril (x=0) hasta septiembre (x=5)."""
    x = np.arange(values.shape[1], dtype=float)
    centered = x - x.mean()
    return (values * centered).sum(axis=1) / (centered**2).sum()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea SOLO variables transformadas; sin ID, demografía ni columnas crudas."""
    # Los nombres de UCI están ordenados del mes más reciente al más antiguo.
    codes = df[STATUS_COLS].to_numpy(dtype=float)
    arrears = (codes > 0).astype(float)
    severity = np.maximum(codes, 0)
    balances = df[BILL_COLS].to_numpy(dtype=float)
    payments = df[PAYMENT_COLS].to_numpy(dtype=float)
    positive_balances = np.maximum(balances, 0)
    positive_payments = np.maximum(payments, 0)
    limits = df["LIMIT_BAL"].to_numpy(dtype=float)

    # Ratios acotados: los saldos pueden ser negativos o superar el cupo.
    util = np.clip(positive_balances / limits[:, None], 0, 5)
    pay_by_limit = np.clip(positive_payments / limits[:, None], 0, 5)
    payments_to_balances = positive_payments / np.maximum(positive_balances, 1)
    payments_to_balances = np.clip(payments_to_balances, 0, 5)

    recent_to_old = np.where(arrears > 0, np.arange(6)[None, :], 7)
    recency = recent_to_old.min(axis=1)
    streak = np.zeros(len(df))
    for idx in range(6):
        streak += (arrears[:, : idx + 1].sum(axis=1) == idx + 1).astype(float)

    out = pd.DataFrame({
        "max_atraso_6m": severity.max(axis=1),
        "atrasos_6m": arrears.sum(axis=1),
        "atrasos_3m": arrears[:, :3].sum(axis=1),
        "recencia_ultimo_atraso": recency,
        "tendencia_atrasos": arrears[:, :3].mean(axis=1) - arrears[:, 3:].mean(axis=1),
        "severidad_reciente": severity[:, :3].max(axis=1),
        "persistencia_atraso": streak,
        "meses_estado_no_positivo": (codes <= 0).sum(axis=1),
        "limite_log": np.log1p(limits),
        "utilizacion_reciente": util[:, 0],
        "utilizacion_media": util.mean(axis=1),
        "utilizacion_max": util.max(axis=1),
        "utilizacion_volatilidad": util.std(axis=1),
        "cambio_utilizacion": util[:, 0] - util[:, -1],
        "tendencia_utilizacion": row_slope(util[:, ::-1]),
        "meses_sobre_cupo": (balances > limits[:, None]).sum(axis=1),
        "importe_saldo_reciente_log": np.log1p(positive_balances[:, 0]),
        "pago_reciente_log": np.log1p(positive_payments[:, 0]),
        "ratio_pago_saldo_agregado": positive_payments.sum(axis=1) / np.maximum(positive_balances.sum(axis=1), 1),
        "ratio_pago_saldo_reciente": payments_to_balances[:, 0],
        "ratio_pago_saldo_mediano": np.median(payments_to_balances, axis=1),
        "meses_sin_pago": (positive_payments == 0).sum(axis=1),
        "volatilidad_pago": pay_by_limit.std(axis=1),
        "tendencia_pago": row_slope(pay_by_limit[:, ::-1]),
        "pago_reciente_vs_historico": positive_payments[:, 0] / np.maximum(positive_payments[:, 1:].mean(axis=1), 1),
    }, index=df.index)
    out["ratio_pago_saldo_agregado"] = out["ratio_pago_saldo_agregado"].clip(upper=5)
    out["pago_reciente_vs_historico"] = out["pago_reciente_vs_historico"].clip(upper=10)
    if out.isna().any().any() or not np.isfinite(out.to_numpy()).all():
        raise ValueError("El feature engineering produjo valores inválidos")
    if set(out.columns) != set(DESCRIPTIONS):
        raise ValueError("Las variables no coinciden con su diccionario")
    return out


def ks_statistic(y: np.ndarray, probabilities: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y, probabilities)
    return float(np.max(tpr - fpr))


def choose_threshold(y: np.ndarray, probabilities: np.ndarray, objective: str) -> dict:
    """Selección EXCLUSIVAMENTE sobre predicciones OOF de desarrollo."""
    base_rate = float(np.mean(y))
    # Grid de cuantiles y un umbral mínimo: evita elegir umbrales imposibles.
    grid = np.unique(np.quantile(probabilities, np.linspace(0.03, 0.98, 191)))
    options = []
    for threshold in grid:
        pred = probabilities >= threshold
        rate = float(pred.mean())
        if not pred.any():
            continue
        prec = precision_score(y, pred, zero_division=0)
        rec = recall_score(y, pred, zero_division=0)
        f1 = f1_score(y, pred, zero_division=0)
        options.append({"threshold": float(threshold), "precision": float(prec), "recall": float(rec), "f1": float(f1), "contact_rate": rate})

    if objective == "precision":
        eligible = [x for x in options if x["contact_rate"] >= 0.05 and x["recall"] >= 0.12]
        score = lambda x: (x["precision"], x["recall"])
    elif objective == "recall":
        eligible = [x for x in options if x["precision"] >= base_rate + 0.08 and x["contact_rate"] <= 0.45]
        score = lambda x: (x["recall"], x["precision"])
    elif objective == "f1":
        eligible = [x for x in options if x["contact_rate"] <= 0.40]
        score = lambda x: (x["f1"], x["precision"])
    else:
        raise ValueError(objective)
    if not eligible:
        raise ValueError(f"No hay umbral viable para {objective}; no se inventa uno")
    return max(eligible, key=score)


def gain_table(y: np.ndarray, probabilities: np.ndarray, model: str) -> pd.DataFrame:
    indices = np.argsort(-probabilities, kind="stable")
    y_sorted = np.asarray(y)[indices]
    total_events = y_sorted.sum()
    n = len(y_sorted)
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(n), 10), start=1):
        events = int(y_sorted[idx].sum())
        rows.append({
            "modelo": model,
            "decil": i,
            "clientes": len(idx),
            "eventos": events,
            "tasa_evento": float(events / len(idx)),
            "lift": float((events / len(idx)) / (total_events / n)),
            "captura_acumulada": float(y_sorted[: idx[-1] + 1].sum() / total_events),
        })
    return pd.DataFrame(rows)


def model_metrics(y: np.ndarray, probabilities: np.ndarray, threshold: float, model: str) -> dict:
    predicted = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "modelo": model,
        "umbral": round(float(threshold), 5),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 5),
        "pr_auc_average_precision": round(float(average_precision_score(y, probabilities)), 5),
        "ks": round(ks_statistic(y, probabilities), 5),
        "precision": round(float(precision_score(y, predicted, zero_division=0)), 5),
        "recall": round(float(recall_score(y, predicted, zero_division=0)), 5),
        "f1": round(float(f1_score(y, predicted, zero_division=0)), 5),
        "brier": round(float(brier_score_loss(y, probabilities)), 5),
        "contact_rate": round(float(predicted.mean()), 5),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def calibration_bins(y: np.ndarray, p: np.ndarray) -> tuple[list[float], list[float]]:
    bins = pd.DataFrame({"y": y, "p": p})
    bins["bin"] = pd.qcut(bins["p"], q=10, duplicates="drop")
    aggregated = bins.groupby("bin", observed=True).agg(observed=("y", "mean"), predicted=("p", "mean"))
    return aggregated["predicted"].tolist(), aggregated["observed"].tolist()


def make_plots(y: np.ndarray, predictions: dict, gain: pd.DataFrame, importance: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white"})

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name, p in predictions.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(recall, precision, color=PALETTE[name], lw=2, label=f"{name.title()} · AP={ap:.3f}")
    ax.axhline(np.mean(y), color="#919d94", ls="--", label="Tasa base")
    ax.set(xlabel="Recall", ylabel="Precisión", title="Trade-off: precisión vs. recall")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(alpha=.14); ax.legend()
    fig.tight_layout(); fig.savefig(folder / "precision_recall.png", dpi=155); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot([0, 1], [0, 1], ls="--", color="#97a69b", label="Calibración ideal")
    for name, p in predictions.items():
        x, z = calibration_bins(y, p)
        ax.plot(x, z, "o-", ms=4, lw=2, color=PALETTE[name], label=name.title())
    ax.set(xlabel="Probabilidad estimada", ylabel="Frecuencia observada", title="Calibración de riesgo (deciles, zoom 0–0,45)")
    ax.set_xlim(0, .45); ax.set_ylim(0, .45); ax.grid(alpha=.14); ax.legend()
    fig.tight_layout(); fig.savefig(folder / "calibracion.png", dpi=155); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.2))
    for name in predictions:
        data = gain.loc[gain["modelo"] == name]
        ax.plot(data["decil"], data["captura_acumulada"] * 100, marker="o", color=PALETTE[name], lw=2, label=name.title())
    ax.plot([1, 10], [10, 100], "--", color="#97a69b", label="Selección aleatoria")
    ax.set(xlabel="Deciles contactados (1 = mayor riesgo)", ylabel="Eventos capturados acumulados (%)", title="Curva de ganancias · cohorte de prueba")
    ax.set_xticks(range(1, 11)); ax.set_ylim(0, 105); ax.grid(alpha=.14); ax.legend()
    fig.tight_layout(); fig.savefig(folder / "ganancias.png", dpi=155); plt.close(fig)

    top = importance.sort_values("importancia_ap", ascending=False).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.9))
    ax.barh(top["variable"], top["importancia_ap"], color=[GREEN if x >= 0 else GOLD for x in top["importancia_ap"]])
    ax.set(xlabel="Caída de AP al permutar (prueba)", title="Importancia por permutación · modelo de precisión")
    ax.grid(axis="x", alpha=.14)
    fig.tight_layout(); fig.savefig(folder / "variables.png", dpi=155); plt.close(fig)


def write_summary(output: Path, audit: dict, metrics: list[dict], winner: dict, best_lift: dict) -> None:
    header = "| Estrategia | ROC AUC | PR AUC (AP) | KS | Precisión | Recall | F1 | Brier | Top decil lift | % contactado |\n"
    header += "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    for m in metrics:
        header += f"| {m['modelo']} | {m['roc_auc']:.3f} | {m['pr_auc_average_precision']:.3f} | {m['ks']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['brier']:.3f} | {best_lift[m['modelo']]:.2f}× | {100*m['contact_rate']:.1f}% |\n"
    content = f"""# Resultados reproducibles · cohorte preventiva

> ⚠️ **Datos públicos de UCI, no de clientes de Quórum Risk.** Estos números son de un ejercicio académico/demostrativo; no representan resultados comerciales.

Fuente: I-Cheng Yeh (2009), *Default of Credit Card Clients*, UCI Machine Learning Repository, {SOURCE_DOI}; **CC BY 4.0**.

## Población y etiqueta

- Fuente total: **{audit['total_filas_fuente']:,}** filas.
- Sin atraso positivo registrado en septiembre de 2005: **{audit['filas_cohorte_preventiva']:,}**.
- Excluidas por atraso positivo en septiembre: **{audit['filas_atraso_positivo_actual_excluidas']:,}** (gestión correctiva, fuera de este demo).
- Resultado UCI para el mes siguiente: **{audit['eventos_cohorte']:,}** eventos (**{audit['tasa_evento_cohorte']:.2%}**).
- Observación abril–septiembre de 2005; desempeño en el **mes siguiente** según etiqueta original. Se trata de *default reportado*, no de una definición propia de 30/60/90 días de mora.
- El dataset tiene un único corte: **no permite prueba out-of-time**. Se utilizó separación estratificada y validación cruzada estratificada en desarrollo.

## Comparación final (holdout, no usada para ajustar umbrales)

{header}
    **Mayor PR AUC observada: `{winner['modelo']}`**, AP={winner['pr_auc_average_precision']:.3f}. La diferencia con el siguiente método es **{winner['pr_auc_average_precision'] - sorted((m['pr_auc_average_precision'] for m in metrics), reverse=True)[1]:.4f} puntos de AP**: demasiado pequeña para proclamar superioridad robusta sin intervalos de incertidumbre ni validación temporal. Tampoco implica que sea mejor en recall, precisión a un umbral o valor económico.

Los umbrales se seleccionaron usando predicciones *out-of-fold* del conjunto de desarrollo, con restricciones de cobertura/capacidad. La combinación es un promedio 50/50, evaluado como tercera hipótesis y **no se presupone mejor**.

## Visualizaciones

![Trade-off de precisión y recall](figures/precision_recall.png)
![Calibración](figures/calibracion.png)
![Ganancias acumuladas](figures/ganancias.png)
![Importancia de variables](figures/variables.png)

## Advertencias de interpretación

1. **No predice un nuevo atraso puro**: la etiqueta es default del mes siguiente tal como aparece en UCI. No se observan decisiones, contacto, pagos futuros intermedios ni fecha exacta de evento.
2. **La cohorte preventiva es una aproximación**: `PAY_0 <= 0` significa ausencia de código positivo de atraso, y los estados 0/-2 tienen ambigüedad semántica documentada.
3. Los ratios entre pagos y saldos son indicadores analíticos, **no conciliaciones contables**; pueden involucrar ciclos de facturación distintos.
4. La importancia por permutación muestra asociación predictiva, **no causalidad**. Las variables correlacionadas pueden repartirse la importancia.
5. Las probabilidades requieren examinar calibración, representatividad y efectos por grupos antes de cualquier uso real.
6. Una implementación real exige definición de target y exclusiones aprobada por negocio/riesgos; revisión de leakage; prueba temporal posterior; costos/capacidad; PSI/drift; monitoreo; recalibración; y gobernanza.

Reproducir: `pip install -r requirements.txt && python run_demo.py`.
"""
    (output / "resumen.md").write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo Quórum Risk: alerta temprana con UCI #350")
    parser.add_argument("--cache-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = download_data(args.cache_dir)
    cohort, audit = select_cohort(df)
    features = engineer_features(cohort)
    labels = cohort[TARGET].to_numpy(dtype=int)
    x_dev, x_test, y_dev, y_test = train_test_split(
        features, labels, test_size=0.25, random_state=RANDOM_STATE, stratify=labels
    )

    models = {
        "precision": HistGradientBoostingClassifier(
            max_iter=145, learning_rate=0.055, max_leaf_nodes=13, min_samples_leaf=75,
            l2_regularization=5, random_state=RANDOM_STATE, early_stopping=False
        ),
        "recall": make_pipeline(StandardScaler(), LogisticRegression(max_iter=950, C=0.35, random_state=RANDOM_STATE)),
    }
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    oof = {}
    holdout = {}
    fit_models = {}
    for name, estimator in models.items():
        print(f"CV estratificada · {name}")
        oof[name] = cross_val_predict(estimator, x_dev, y_dev, cv=folds, method="predict_proba", n_jobs=1)[:, 1]
        fitted = clone(estimator).fit(x_dev, y_dev)
        fit_models[name] = fitted
        holdout[name] = fitted.predict_proba(x_test)[:, 1]
    oof["combinacion"] = (oof["precision"] + oof["recall"]) / 2
    holdout["combinacion"] = (holdout["precision"] + holdout["recall"]) / 2

    strategies = {"precision": "precision", "recall": "recall", "combinacion": "f1"}
    thresholds = {name: choose_threshold(y_dev, oof[name], objective) for name, objective in strategies.items()}
    results = [model_metrics(y_test, holdout[name], thresholds[name]["threshold"], name) for name in holdout]
    gain = pd.concat([gain_table(y_test, p, name) for name, p in holdout.items()], ignore_index=True)
    gain.to_csv(args.output_dir / "tabla_lift.csv", index=False, float_format="%.6f")

    imp = permutation_importance(fit_models["precision"], x_test, y_test, scoring="average_precision", n_repeats=3, random_state=RANDOM_STATE, n_jobs=1)
    importance = pd.DataFrame({"variable": features.columns, "descripcion": [DESCRIPTIONS[x] for x in features.columns], "importancia_ap": imp.importances_mean, "desviacion": imp.importances_std})
    importance.sort_values("importancia_ap", ascending=False).to_csv(args.output_dir / "importancia_variables.csv", index=False, float_format="%.6f")

    winner = max(results, key=lambda x: x["pr_auc_average_precision"])
    best_lift = {name: float(gain.loc[(gain.modelo == name) & (gain.decil == 1), "lift"].iloc[0]) for name in holdout}
    make_plots(y_test, holdout, gain, importance, args.output_dir / "figures")
    write_summary(args.output_dir, audit, results, winner, best_lift)
    report = {
        "dataset": {"nombre": "Default of Credit Card Clients", "doi": SOURCE_DOI, "licencia": "CC BY 4.0", "fuente": SOURCE_URL},
        "definicion_target": {"observacion": "abril-septiembre 2005", "desempeno": "siguiente mes (etiqueta UCI)", "evento": "default payment next month = 1", "no_evento": "default payment next month = 0", "indeterminados": "etiquetas ausentes/no binarias excluidas; ninguno en este dataset", "exclusion_cartera_correctiva": "PAY_0 > 0", "semantica_ambigua": "PAY_0 0/-2 no prueba que esté al día"},
        "cohorte": audit,
        "validacion": {"particion": "75/25 estratificada, random_state=42", "desarrollo_cv": "4-fold StratifiedKFold; umbrales a partir de probabilidades out-of-fold", "motivo_no_out_of_time": "único corte temporal para todos los clientes", "n_desarrollo": len(y_dev), "n_prueba": len(y_test)},
        "umbrales_oof": thresholds,
        "modelos": results,
        "ganador_segun_pr_auc": winner["modelo"],
        "aviso": "Demostración académica. No son resultados de clientes ni prueba causal o de impacto económico.",
    }
    (args.output_dir / "metricas.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nMétricas de prueba (holdout):")
    print(pd.DataFrame(results)[["modelo", "roc_auc", "pr_auc_average_precision", "ks", "precision", "recall", "f1", "brier", "contact_rate"]].to_string(index=False))
    print(f"\nGanador por PR AUC: {winner['modelo']} · {winner['pr_auc_average_precision']:.4f}")
    print(f"Reportes y gráficos: {args.output_dir}")


if __name__ == "__main__":
    main()
