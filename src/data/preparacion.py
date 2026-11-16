"""Universo transversal: sin código positivo de atraso en septiembre 2005."""

from __future__ import annotations

import pandas as pd

from src.data.carga import BILL_COLS, PAYMENT_COLS, STATUS_COLS, TARGET


def seleccionar_cohorte(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """No equiparar PAY_0 <= 0 con cliente contractualmente al día."""
    target_valido = df[TARGET].isin([0, 1])
    limite_valido = df["LIMIT_BAL"].notna() & df["LIMIT_BAL"].gt(0)
    historia_valida = df[[*STATUS_COLS, *BILL_COLS, *PAYMENT_COLS]].notna().all(axis=1)
    valido = target_valido & limite_valido & historia_valida
    mora_positiva = df["PAY_0"].gt(0)
    cohorte = df.loc[valido & ~mora_positiva].copy()
    if cohorte.empty or cohorte[TARGET].nunique() != 2:
        raise ValueError("La cohorte no permite clasificación binaria")
    auditoria = {
        "total_filas_fuente": len(df),
        "filas_target_indeterminado_o_invalido": int((~target_valido).sum()),
        "filas_cupo_o_historial_incompleto": int((target_valido & (~limite_valido | ~historia_valida)).sum()),
        "filas_atraso_positivo_actual_excluidas": int((valido & mora_positiva).sum()),
        "filas_cohorte_preventiva": len(cohorte),
        "eventos_cohorte": int(cohorte[TARGET].sum()),
        "tasa_evento_cohorte": float(cohorte[TARGET].mean()),
        "nota": "PAY_0 <= 0 solo indica ausencia de atraso positivo registrado; 0/-2 son ambiguos.",
    }
    return cohorte, auditoria
