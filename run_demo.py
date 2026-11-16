"""Ejecuta de punta a punta el caso UCI: riesgo transversal del mes siguiente.

Uso: python run_demo.py
Test: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from src.data.carga import SOURCE_DOI, SOURCE_URL, SOURCE_ZIP_SHA256, TARGET, descargar_uci
from src.data.preparacion import seleccionar_cohorte
from src.evaluacion.graficos import guardar_figuras
from src.evaluacion.informes import escribir_resumen
from src.evaluacion.metricas import (
    captura_top_20_eventos, elegir_umbral, metricas_modelo, reglas_sin_ml,
    tabla_ganancias,
)
from src.features.construccion import DESCRIPCIONES, construir_variables
from src.models.combinacion.modelo import combinar
from src.models.precision.modelo import crear_modelo as crear_boosting
from src.models.recall.modelo import crear_modelo as crear_logistica

RAIZ = Path(__file__).resolve().parent
SEMILLA = 42
DEPENDENCIAS = ("numpy", "pandas", "scikit-learn", "matplotlib", "xlrd", "PyYAML")
# Revisión publicada del código modular que se audita; el commit final
# de informes es otro porque un JSON no puede contener su propio hash.
COMMIT_CODIGO_FUENTE = "95f895c7"


def validar_entorno() -> None:
    """Falla antes de descargar/ajustar si el entorno difiere del congelado."""
    for linea in (RAIZ / "requirements.txt").read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        nombre, separador, esperada = linea.partition("==")
        if separador != "==":
            raise ValueError(f"Dependencia no fijada exactamente: {linea}")
        instalada = version(nombre)
        if instalada != esperada:
            raise RuntimeError(
                f"Versión incompatible de {nombre}: {instalada} instalada, "
                f"{esperada} requerida. Ejecuta pip install -r requirements.txt."
            )
    if platform.python_version() != "3.12.14":
        raise RuntimeError(
            f"Python {platform.python_version()} no coincide con la versión "
            "de referencia 3.12.14. Lee docs/ENTORNO.md."
        )


def entorno() -> dict:
    """Metadatos estables; sin timestamp para facilitar comparación de hashes."""
    return {
        "python": platform.python_version(),
        "sistema": platform.system(),
        "arquitectura": platform.machine(),
        "librerias": {p: version(p) for p in DEPENDENCIAS},
        "commit_codigo_fuente": COMMIT_CODIGO_FUENTE,
        "nota_commit": "Revisión fuente del pipeline; no es el commit que incorpora este JSON (referencia circular).",
    }


def ejecutar(cache_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    validar_entorno()
    cache_dir = cache_dir or RAIZ / "data"
    output_dir = output_dir or RAIZ / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = descargar_uci(cache_dir)
    cohorte, auditoria = seleccionar_cohorte(df)
    x = construir_variables(cohorte)
    y = cohorte[TARGET].to_numpy(dtype=int)
    x_des, x_prueba, y_des, y_prueba = train_test_split(
        x, y, test_size=0.25, random_state=SEMILLA, stratify=y
    )

    candidatos = {"precision": crear_boosting(), "recall": crear_logistica()}
    folds = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEMILLA)
    oof, prueba, ajustados = {}, {}, {}
    for nombre, estimador in candidatos.items():
        print(f"CV estratificada · {nombre}")
        oof[nombre] = cross_val_predict(
            estimador, x_des, y_des, cv=folds, method="predict_proba", n_jobs=1
        )[:, 1]
        modelo = clone(estimador).fit(x_des, y_des)
        ajustados[nombre] = modelo
        prueba[nombre] = modelo.predict_proba(x_prueba)[:, 1]
    oof["combinacion"] = combinar(oof["precision"], oof["recall"])
    prueba["combinacion"] = combinar(prueba["precision"], prueba["recall"])

    objetivos = {"precision": "precision", "recall": "recall", "combinacion": "f1"}
    umbrales = {n: elegir_umbral(y_des, oof[n], objetivo) for n, objetivo in objetivos.items()}
    resultados = [
        metricas_modelo(y_prueba, p, umbrales[n]["threshold"], n)
        for n, p in prueba.items()
    ]
    ganancias = pd.concat(
        [tabla_ganancias(y_prueba, p, n) for n, p in prueba.items()], ignore_index=True
    )
    ganancias.to_csv(output_dir / "tabla_lift.csv", index=False, float_format="%.6f")

    importancia = permutation_importance(
        ajustados["precision"], x_prueba, y_prueba,
        scoring="average_precision", n_repeats=3, random_state=SEMILLA, n_jobs=1,
    )
    tabla_importancia = pd.DataFrame({
        "variable": x.columns,
        "descripcion": [DESCRIPCIONES[v] for v in x.columns],
        "importancia_ap": importancia.importances_mean,
        "desviacion": importancia.importances_std,
    })
    tabla_importancia.sort_values("importancia_ap", ascending=False).to_csv(
        output_dir / "importancia_variables.csv", index=False, float_format="%.6f"
    )

    reglas = reglas_sin_ml(x_prueba)
    baselines = {
        n: {"captura_eventos_top_20_pct": captura_top_20_eventos(y_prueba, orden)}
        for n, orden in reglas.items()
    }
    mejor_regla = max(baselines, key=lambda n: baselines[n]["captura_eventos_top_20_pct"])
    referencia = baselines[mejor_regla]["captura_eventos_top_20_pct"]
    for r in resultados:
        r["captura_eventos_top_20_pct"] = captura_top_20_eventos(y_prueba, prueba[r["modelo"]])
        r["diferencia_vs_mejor_regla_pp"] = round(r["captura_eventos_top_20_pct"] - referencia, 6)

    ganador_ap = max(resultados, key=lambda r: r["pr_auc_average_precision"])["modelo"]
    ganador_captura = max(
        resultados, key=lambda r: (r["captura_eventos_top_20_pct"], -r["brier"])
    )["modelo"]
    lifts = {
        n: float(ganancias.loc[ganancias["modelo"].eq(n) & ganancias["decil"].eq(1), "lift"].iloc[0])
        for n in prueba
    }
    informe = {
        "aviso": "Datos públicos de UCI, no de clientes. Resultados offline sin inferencia causal ni impacto económico.",
        "entorno": entorno(),
        "dataset": {"nombre": "Default of Credit Card Clients", "doi": SOURCE_DOI, "licencia": "CC BY 4.0", "fuente": SOURCE_URL, "sha256_zip": SOURCE_ZIP_SHA256},
        "definicion_target": {
            "observacion": "abril-septiembre 2005", "desempeno": "mes siguiente (etiqueta UCI)",
            "evento": "default payment next month = 1", "no_evento": "default payment next month = 0",
            "indeterminados": "etiquetas ausentes/no binarias excluidas",
            "exclusion_cartera_correctiva": "PAY_0 > 0",
            "semantica_ambigua": "PAY_0 0/-2 no garantiza estar al día",
        },
        "cohorte": auditoria,
        "validacion": {
            "particion": "75/25 estratificada, random_state=42",
            "desarrollo_cv": "4-fold StratifiedKFold, umbrales según probabilidades OOF",
            "motivo_no_out_of_time": "único corte transversal; no hay cohortes de observación en fechas distintas",
            "n_desarrollo": len(y_des), "n_prueba": len(y_prueba),
        },
        "umbrales_oof": umbrales,
        "baselines": baselines,
        "mejor_baseline": mejor_regla,
        "modelos": resultados,
        "ganador_segun_pr_auc": ganador_ap,
        "mejor_modelo_captura_top_20": ganador_captura,
        "nota_uplift": "Diferencia descriptiva de captura de eventos; NO uplift causal, ahorro o ROI.",
    }
    (output_dir / "metricas.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    guardar_figuras(y_prueba, prueba, ganancias, tabla_importancia, output_dir / "figures", reglas)
    escribir_resumen(output_dir, informe, lifts)
    print(pd.DataFrame(resultados)[[
        "modelo", "roc_auc", "pr_auc_average_precision", "ks",
        "captura_eventos_top_20_pct", "diferencia_vs_mejor_regla_pp",
    ]].to_string(index=False))
    print(f"Mejor regla simple: {mejor_regla} ({referencia:.2f}%)")
    return informe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo transversal de riesgo UCI, no validación temporal")
    parser.add_argument("--cache-dir", type=Path, default=RAIZ / "data")
    parser.add_argument("--output-dir", type=Path, default=RAIZ / "reports")
    args = parser.parse_args()
    ejecutar(args.cache_dir, args.output_dir)
