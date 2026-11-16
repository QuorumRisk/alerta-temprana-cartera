# Validación transversal, sin reclamar out-of-time

## Lo que el dataset permite demostrar

Los 30.000 registros UCI comparten **una única fecha de observación**:
historia abril–septiembre de 2005 y etiqueta del mes siguiente. **No hay
cohortes de decisión independientes en años/meses posteriores**. Ordenar por
`ID` sería metodológicamente incorrecto: no convierte el test en temporal.

1. El conjunto elegible se divide **75 % desarrollo / 25 % test** mediante
   `train_test_split(stratify=y, random_state=42)`.
2. En desarrollo se ejecuta **4-fold StratifiedKFold** con semilla 42.
3. Los candidatos generan probabilidades **out-of-fold**. Sobre ellas, y
   no sobre test, se eligen los umbrales de precisión, recall y F1.
4. Los modelos se entrenan en el desarrollo completo y se predice el 25 %
   reservado **una sola vez**, sin retocar pesos ni umbrales después.
5. Los hiperparámetros se conservan en YAML y **no** se eligen usando test.

## Candidatos y criterios

| Candidato | Modelo | Umbral de desarrollo |
| --- | --- | --- |
| Precisión | HistGradientBoosting | Maximiza precisión; contacto ≥5 %, recall ≥12 % |
| Recall | Regresión logística escalada | Maximiza recall; precisión ≥tasa base+8 pp, contacto ≤45 % |
| Combinación | Promedio fijo 50/50 | Maximiza F1; contacto ≤40 % |

Los nombres de las estrategias reflejan **los umbrales**, no una garantía
intrínseca de que cada algoritmo sea siempre el mejor en esa métrica.
La combinación se evalúa y no se considera superior por definición.

## Métricas y comparación con reglas simples

- AUC ROC, average precision (PR-AUC/AP), KS, Brier.
- Precisión, recall y F1 al umbral OOF; porcentaje de cuentas señaladas.
- Lift por decil, ganancias acumuladas y calibración por cuantiles.
- **Benchmark sin ML**, fijado antes de conocer las etiquetas de test:
  - Mayor `utilizacion_reciente`.
  - Mayor `max_atraso_6m`, desempate por utilización reciente.
- **Misma métrica para todos:** proporción de eventos del mes siguiente
  capturada al priorizar el **20 % de clientes** por cada estrategia.
- El diferencial **modelo − mejor regla simple** aparece en JSON, resumen
  y `figures/valor_incremental.png`. No es uplift causal ni dinero recuperado.

El mejor candidato por **AP** y el mejor candidato por **captura top 20 %**
se identifican separadamente. Un ganador retrospectivo en test no se usa
para reoptimizar los modelos: la elección real debería confirmarse en un
período posterior y evaluarse contra política vigente.

## No apto para despliegue

Faltan out-of-time, estimación de incertidumbre/intervalos, auditoría de
disponibilidad por fecha, seguridad y sesgos, costes y capacidad operativa,
validación de calibración por segmentos, PSI/drift y seguimiento.
Un score de riesgo tampoco mide que contactar a un cliente **evite** su
evento: para ello hacen falta diseños experimentales o causales.
