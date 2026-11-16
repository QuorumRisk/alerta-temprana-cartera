"""Pruebas sin red: python -m unittest discover -s tests -v."""

from __future__ import annotations

import json
import platform
import tempfile
import unittest
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_demo import COMMIT_CODIGO_FUENTE, validar_entorno
from src.data.carga import REQUIRED, STATUS_COLS, TARGET, descargar_uci
from src.data.preparacion import seleccionar_cohorte
from src.evaluacion.metricas import (
    captura_top_20_eventos, elegir_umbral, reglas_sin_ml, tabla_ganancias,
)
from src.features.construccion import DESCRIPCIONES, construir_variables
from src.models.combinacion.modelo import combinar
from src.models.precision.modelo import crear_modelo as boosting
from src.models.recall.modelo import crear_modelo as logistica


def filas_de_prueba() -> pd.DataFrame:
    filas = []
    estados = [[-1, 0, 2, -1, -1, -1], [2, 1, 0, -1, -1, -2], [0] * 6]
    for indice, serie in enumerate(estados):
        fila = {campo: 0 for campo in REQUIRED}
        fila.update({"ID": indice + 1, "LIMIT_BAL": 100000, TARGET: int(indice == 0)})
        for campo, valor in zip(STATUS_COLS, serie):
            fila[campo] = valor
        for mes in range(1, 7):
            fila[f"BILL_AMT{mes}"] = 14000 + (6 - mes) * 1000
            fila[f"PAY_AMT{mes}"] = 1200 + (6 - mes) * 50
        filas.append(fila)
    return pd.DataFrame(filas)


class PruebasDemo(unittest.TestCase):
    def test_cohorte_excluye_mora_positiva_actual(self):
        cohorte, auditoria = seleccionar_cohorte(filas_de_prueba())
        self.assertEqual(len(cohorte), 2)
        self.assertEqual(auditoria["filas_atraso_positivo_actual_excluidas"], 1)
        self.assertTrue(cohorte["PAY_0"].le(0).all())

    def test_solo_variables_transformadas_no_identificadores(self):
        x = construir_variables(filas_de_prueba())
        self.assertEqual(set(x.columns), set(DESCRIPCIONES))
        self.assertTrue(set(x.columns).isdisjoint(REQUIRED))
        self.assertTrue(np.isfinite(x.to_numpy()).all())

    def test_recencia_hasta_septiembre(self):
        x = construir_variables(filas_de_prueba())
        self.assertEqual(x.loc[0, "recencia_ultimo_atraso"], 2)
        self.assertEqual(x.loc[2, "recencia_ultimo_atraso"], 7)

    def test_lift_decil_y_captura(self):
        y = np.array([1] * 10 + [0] * 90)
        prob = np.linspace(1.0, 0.0, 100)
        lift = tabla_ganancias(y, prob, "precision")
        self.assertAlmostEqual(lift.iloc[0]["lift"], 10.0)
        self.assertAlmostEqual(lift.iloc[-1]["captura_acumulada"], 1.0)
        self.assertAlmostEqual(captura_top_20_eventos(y, prob), 100.0)

    def test_umbral_oof_tiene_restricciones_explicitas(self):
        y = np.array([1] * 20 + [0] * 80)
        prob = np.array([0.8] * 20 + [0.1] * 80)
        for objetivo in ("precision", "recall", "f1"):
            elegido = elegir_umbral(y, prob, objetivo)
            self.assertGreater(elegido["contact_rate"], 0)
            self.assertLess(elegido["threshold"], 1)

    def test_reglas_sin_ml_son_independientes_de_label(self):
        x = pd.DataFrame({
            "utilizacion_reciente": [0.3, 0.9, 0.1, 0.7, 0.8],
            "max_atraso_6m": [2, 0, 1, 2, 0],
        })
        reglas = reglas_sin_ml(x)
        self.assertEqual(np.argsort(-reglas["utilizacion_descendente"]).tolist(), [1, 4, 3, 0, 2])
        self.assertEqual(np.argsort(-reglas["peor_atraso_6m_desempate_utilizacion"]).tolist(), [3, 0, 2, 1, 4])
        y = np.array([1, 0, 0, 1, 0])
        self.assertAlmostEqual(captura_top_20_eventos(y, reglas["utilizacion_descendente"]), 0.0)
        self.assertAlmostEqual(captura_top_20_eventos(y, reglas["peor_atraso_6m_desempate_utilizacion"]), 50.0)

    def test_configuraciones_de_modelos_y_combinacion(self):
        self.assertEqual(boosting().max_iter, 145)
        self.assertEqual(logistica().named_steps["logisticregression"].C, 0.35)
        np.testing.assert_allclose(combinar(np.array([0.2, 0.8]), np.array([0.4, 0.6])), [0.3, 0.7])

    def test_requirements_exactos_y_entorno_reportado(self):
        validar_entorno()
        raiz = Path(__file__).resolve().parents[1]
        reporte = json.loads((raiz / "reports" / "metricas.json").read_text(encoding="utf-8"))
        e = reporte["entorno"]
        self.assertEqual(e["python"], platform.python_version())
        self.assertEqual(e["commit_codigo_fuente"], COMMIT_CODIGO_FUENTE)
        for linea in (raiz / "requirements.txt").read_text(encoding="utf-8").splitlines():
            nombre, esperada = linea.split("==")
            self.assertEqual(e["librerias"][nombre], esperada)
            self.assertEqual(version(nombre), esperada)

    def test_rechaza_entorno_con_libreria_distinta(self):
        with patch("run_demo.version", return_value="0.0.0"):
            with self.assertRaisesRegex(RuntimeError, "Versión incompatible"):
                validar_entorno()

    def test_rechaza_zip_publico_modificado(self):
        with tempfile.TemporaryDirectory() as temporal:
            (Path(temporal) / "uci-350.zip").write_bytes(b"contenido artificial alterado")
            with self.assertRaisesRegex(ValueError, "SHA-256 observado"):
                descargar_uci(Path(temporal))

    def test_resumen_y_json_comparten_capturas_y_uplift(self):
        raiz = Path(__file__).resolve().parents[1] / "reports"
        reporte = json.loads((raiz / "metricas.json").read_text(encoding="utf-8"))
        resumen = (raiz / "resumen.md").read_text(encoding="utf-8")
        mejor = reporte["baselines"][reporte["mejor_baseline"]]["captura_eventos_top_20_pct"]
        for fila in reporte["modelos"]:
            captura, diferencia = fila["captura_eventos_top_20_pct"], fila["diferencia_vs_mejor_regla_pp"]
            self.assertAlmostEqual(captura - mejor, diferencia, places=5)
            self.assertIn(f"{captura:.2f}%", resumen)
            self.assertIn(f"{diferencia:+.2f} pp", resumen)


if __name__ == "__main__":
    unittest.main()
