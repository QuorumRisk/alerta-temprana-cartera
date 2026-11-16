"""Indicadores originales desde abril–septiembre 2005, sin campos crudos en X."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.carga import BILL_COLS, PAYMENT_COLS, STATUS_COLS

DESCRIPCIONES = {
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


def _pendiente_mensual(valores: np.ndarray) -> np.ndarray:
    eje = np.arange(valores.shape[1], dtype=float)
    centrado = eje - eje.mean()
    return (valores * centrado).sum(axis=1) / (centrado**2).sum()


def construir_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Construye 25 predictores; nunca arrastra target, ID ni demografía."""
    codigos = df[STATUS_COLS].to_numpy(dtype=float)
    atrasos = (codigos > 0).astype(float)
    severidad = np.maximum(codigos, 0)
    saldos = df[BILL_COLS].to_numpy(dtype=float)
    pagos = df[PAYMENT_COLS].to_numpy(dtype=float)
    saldos_pos = np.maximum(saldos, 0)
    pagos_pos = np.maximum(pagos, 0)
    cupos = df["LIMIT_BAL"].to_numpy(dtype=float)

    uso = np.clip(saldos_pos / cupos[:, None], 0, 5)
    pago_sobre_cupo = np.clip(pagos_pos / cupos[:, None], 0, 5)
    pago_sobre_saldo = np.clip(pagos_pos / np.maximum(saldos_pos, 1), 0, 5)
    recencia = np.where(atrasos > 0, np.arange(6)[None, :], 7).min(axis=1)
    racha = np.zeros(len(df))
    for k in range(6):
        racha += (atrasos[:, : k + 1].sum(axis=1) == k + 1).astype(float)

    variables = pd.DataFrame({
        "max_atraso_6m": severidad.max(axis=1),
        "atrasos_6m": atrasos.sum(axis=1),
        "atrasos_3m": atrasos[:, :3].sum(axis=1),
        "recencia_ultimo_atraso": recencia,
        "tendencia_atrasos": atrasos[:, :3].mean(axis=1) - atrasos[:, 3:].mean(axis=1),
        "severidad_reciente": severidad[:, :3].max(axis=1),
        "persistencia_atraso": racha,
        "meses_estado_no_positivo": (codigos <= 0).sum(axis=1),
        "limite_log": np.log1p(cupos),
        "utilizacion_reciente": uso[:, 0],
        "utilizacion_media": uso.mean(axis=1),
        "utilizacion_max": uso.max(axis=1),
        "utilizacion_volatilidad": uso.std(axis=1),
        "cambio_utilizacion": uso[:, 0] - uso[:, -1],
        "tendencia_utilizacion": _pendiente_mensual(uso[:, ::-1]),
        "meses_sobre_cupo": (saldos > cupos[:, None]).sum(axis=1),
        "importe_saldo_reciente_log": np.log1p(saldos_pos[:, 0]),
        "pago_reciente_log": np.log1p(pagos_pos[:, 0]),
        "ratio_pago_saldo_agregado": pagos_pos.sum(axis=1) / np.maximum(saldos_pos.sum(axis=1), 1),
        "ratio_pago_saldo_reciente": pago_sobre_saldo[:, 0],
        "ratio_pago_saldo_mediano": np.median(pago_sobre_saldo, axis=1),
        "meses_sin_pago": (pagos_pos == 0).sum(axis=1),
        "volatilidad_pago": pago_sobre_cupo.std(axis=1),
        "tendencia_pago": _pendiente_mensual(pago_sobre_cupo[:, ::-1]),
        "pago_reciente_vs_historico": pagos_pos[:, 0] / np.maximum(pagos_pos[:, 1:].mean(axis=1), 1),
    }, index=df.index)
    variables["ratio_pago_saldo_agregado"] = variables["ratio_pago_saldo_agregado"].clip(upper=5)
    variables["pago_reciente_vs_historico"] = variables["pago_reciente_vs_historico"].clip(upper=10)
    if variables.isna().any().any() or not np.isfinite(variables.to_numpy()).all():
        raise ValueError("Los predictores contienen NaN o infinitos")
    if set(variables.columns) != set(DESCRIPCIONES):
        raise ValueError("Las columnas no coinciden con el diccionario")
    return variables
