"""Resumen Markdown ligado a los números de metricas.json (una sola fuente)."""

from __future__ import annotations

from pathlib import Path

from src.data.carga import SOURCE_DOI


def escribir_resumen(destino: Path, reporte: dict, lifts: dict[str, float]) -> None:
    metricas = reporte["modelos"]
    filas = []
    for m in metricas:
        filas.append(
            f"| {m['modelo']} | {m['roc_auc']:.3f} | {m['pr_auc_average_precision']:.3f} | "
            f"{m['ks']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['brier']:.3f} | {lifts[m['modelo']]:.2f}× | "
            f"{m['captura_eventos_top_20_pct']:.2f}% | {m['diferencia_vs_mejor_regla_pp']:+.2f} pp |"
        )
    reglas = [
        f"| `{nombre}` | {datos['captura_eventos_top_20_pct']:.2f}% |"
        for nombre, datos in reporte["baselines"].items()
    ]
    mejor = reporte["mejor_baseline"]
    captura_regla = reporte["baselines"][mejor]["captura_eventos_top_20_pct"]
    candidato = reporte["mejor_modelo_captura_top_20"]
    cap_modelo = next(m["captura_eventos_top_20_pct"] for m in metricas if m["modelo"] == candidato)
    diferencia = cap_modelo - captura_regla
    if diferencia <= 0:
        conclusion = (
            "Ningún candidato supera la mejor regla sencilla en esta métrica. "
            "No se justifica añadir complejidad ML en este experimento."
        )
    elif diferencia < 1:
        conclusion = (
            "La mejora es menor de un punto porcentual; sin bandas de incertidumbre, "
            "costo de intervención y validación temporal, no justifica por sí sola ML."
        )
    else:
        conclusion = (
            "El margen de captura merece investigación adicional, pero no demuestra "
            "estabilidad, causalidad ni valor neto. No se autoriza desplegarlo con esta prueba."
        )
    ganador_ap = reporte["ganador_segun_pr_auc"]
    mejor_ap = next(m["pr_auc_average_precision"] for m in metricas if m["modelo"] == ganador_ap)
    segundo_ap = sorted((m["pr_auc_average_precision"] for m in metricas), reverse=True)[1]
    auditoria = reporte["cohorte"]
    entorno = reporte["entorno"]

    texto = f"""# Evaluación transversal reproducible · UCI #350

> ⚠️ **Datos públicos, no de clientes.** Este caso no mide reducción de mora,
> ganancias monetarias ni impacto comercial de Quórum Risk.

## Conclusión ejecutiva: el modelo debe ganarse su lugar

La mejor regla sin ML es **`{mejor}`** con captura **{captura_regla:.2f}%**
de eventos futuros al revisar el 20% de clientes. El mejor candidato ML
para esa misma métrica es **`{candidato}`**, con **{cap_modelo:.2f}%**:
**{diferencia:+.2f} puntos porcentuales** de diferencia descriptiva.

**{conclusion}** No se conoce la respuesta causal a contactos, costos ni ROI.

## Fuente, universo y target

- I-Cheng Yeh (2009), *Default of Credit Card Clients*, UCI, {SOURCE_DOI}.
  Licencia de datos: **CC BY 4.0**; no se redistribuyen datos originales.
- Archivo original: **{auditoria['total_filas_fuente']:,}** registros.
- Cohorte sin atraso positivo en septiembre de 2005: **{auditoria['filas_cohorte_preventiva']:,}**.
- Excluidos por código positivo de atraso actual: **{auditoria['filas_atraso_positivo_actual_excluidas']:,}**.
- Evento: etiqueta UCI `default payment next month = 1`.
- Eventos en cohorte: **{auditoria['eventos_cohorte']:,}** (**{auditoria['tasa_evento_cohorte']:.2%}**).
- `PAY_0 <= 0` no prueba que se esté al día: códigos 0/-2 ambiguos.

## Validación honesta

- **Un solo corte transversal**: observación abril–septiembre 2005, etiqueta del mes siguiente.
- **75/25 estratificado** (semilla 42); 4 folds estratificados en desarrollo.
- Umbrales escogidos con probabilidades **out-of-fold**, nunca con test.
- **No hay validación out-of-time** ni evidencia de estabilidad entre fechas.
- Combinación **50/50** fija; se evalúa, no se asume superior.

## Métricas del holdout (25 % reservado)

| Estrategia | ROC AUC | PR AUC (AP) | KS | Precisión | Recall | Brier | Lift decil 1 | Captura top 20 % | Δ vs. mejor regla |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(filas)}

**AP más alta observada:** `{ganador_ap}` ({mejor_ap:.3f}); margen frente al
siguiente: **{mejor_ap - segundo_ap:.4f}**. No es evidencia de superioridad
robusta ni equivale al criterio de captura operativa del 20 %.

## Benchmark de políticas sin ML

| Regla de priorización (definida sin mirar etiquetas) | Captura de eventos top 20 % |
| --- | ---: |
{chr(10).join(reglas)}

Ambas reglas usan **solo información de septiembre y meses previos**. La
captura es el porcentaje de etiquetas positivas en el 20 % de cuentas con
mayor prioridad; **no** una reducción causal de defaults ni retorno financiero.

![Comparación contra reglas](figures/valor_incremental.png)

## Otras visualizaciones

![Precisión y recall](figures/precision_recall.png)
![Calibración](figures/calibracion.png)
![Ganancias](figures/ganancias.png)
![Importancia de variables](figures/variables.png)

## Entorno de generación

- Python **{entorno['python']}**, **{entorno['sistema']}** ({entorno['arquitectura']}).
- Commit de código fuente **`{entorno['commit_codigo_fuente']}`**.
- Dependencias exactas en `requirements.txt`; detalles en `docs/ENTORNO.md`.

## Límites para producción

Se requieren definición contractual del evento con negocio, verificación de
latencia y leakage, validación temporal posterior, calibración por segmentos,
análisis de sesgos, simulación de costos y capacidad, gobierno de modelos,
revisión de PSI/drift y evaluación de incremento causal.
"""
    (destino / "resumen.md").write_text(texto, encoding="utf-8")
