# Evaluación transversal reproducible · UCI #350

> ⚠️ **Datos públicos, no de clientes.** Este caso no mide reducción de mora,
> ganancias monetarias ni impacto comercial de Quórum Risk.

## Conclusión ejecutiva: el modelo debe ganarse su lugar

La mejor regla sin ML es **`peor_atraso_6m_desempate_utilizacion`** con captura **35.66%**
de eventos futuros al revisar el 20% de clientes. El mejor candidato ML
para esa misma métrica es **`combinacion`**, con **38.78%**:
**+3.12 puntos porcentuales** de diferencia descriptiva.

**El margen de captura merece investigación adicional, pero no demuestra estabilidad, causalidad ni valor neto. No se autoriza desplegarlo con esta prueba.** No se conoce la respuesta causal a contactos, costos ni ROI.

## Fuente, universo y target

- I-Cheng Yeh (2009), *Default of Credit Card Clients*, UCI, https://doi.org/10.24432/C55S3H.
  Licencia de datos: **CC BY 4.0**; no se redistribuyen datos originales.
- Archivo original: **30,000** registros.
- Cohorte sin atraso positivo en septiembre de 2005: **23,182**.
- Excluidos por código positivo de atraso actual: **6,818**.
- Evento: etiqueta UCI `default payment next month = 1`.
- Eventos en cohorte: **3,207** (**13.83%**).
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
| precision | 0.687 | 0.274 | 0.270 | 0.392 | 0.140 | 0.112 | 2.48× | 38.53% | +2.87 pp |
| recall | 0.667 | 0.262 | 0.243 | 0.213 | 0.570 | 0.113 | 2.33× | 37.03% | +1.37 pp |
| combinacion | 0.685 | 0.277 | 0.259 | 0.221 | 0.534 | 0.112 | 2.44× | 38.78% | +3.12 pp |

**AP más alta observada:** `combinacion` (0.277); margen frente al
siguiente: **0.0023**. No es evidencia de superioridad
robusta ni equivale al criterio de captura operativa del 20 %.

## Benchmark de políticas sin ML

| Regla de priorización (definida sin mirar etiquetas) | Captura de eventos top 20 % |
| --- | ---: |
| `utilizacion_descendente` | 23.32% |
| `peor_atraso_6m_desempate_utilizacion` | 35.66% |

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

- Python **3.12.14**, **Linux** (x86_64).
- Commit de código fuente **`671b6465`**.
- Dependencias exactas en `requirements.txt`; detalles en `docs/ENTORNO.md`.

## Límites para producción

Se requieren definición contractual del evento con negocio, verificación de
latencia y leakage, validación temporal posterior, calibración por segmentos,
análisis de sesgos, simulación de costos y capacidad, gobierno de modelos,
revisión de PSI/drift y evaluación de incremento causal.
