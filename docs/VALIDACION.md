# Cómo evaluamos el ranking de riesgo

Los 30.000 registros de UCI pertenecen a un solo corte: abril a septiembre de 2005 como historia y la etiqueta oficial del mes siguiente como desenlace. No tenemos cosechas de decisión de otros meses o años. Ordenar los registros por `ID` no produciría una prueba out-of-time, aunque el gráfico resultara convincente.

## Separación de datos

Primero apartamos un 25 por ciento para test mediante `train_test_split(stratify=y, random_state=42)`. El 75 por ciento restante es desarrollo y se trabaja con `StratifiedKFold` de cuatro folds y semilla 42. Generamos predicciones out-of-fold para elegir umbrales sin mirar el resultado del conjunto reservado.

Con esos umbrales fijados, entrenamos cada candidato en todo el conjunto de desarrollo y evaluamos una vez en test. Los hiperparámetros residen en YAML y tampoco se ajustan usando test.

## Las tres estrategias

La primera combina `HistGradientBoosting` con una política de umbral que busca precisión. Debe seleccionar al menos el 5 por ciento de las cuentas y alcanzar un recall mínimo del 12 por ciento.

La segunda usa regresión logística escalada. Su umbral intenta maximizar recall, pero exige precisión no inferior a la tasa base de eventos más 8 puntos porcentuales y limita los contactos al 45 por ciento.

La tercera promedia 50/50 las probabilidades de ambos modelos. Elegimos su umbral para F1 en desarrollo, con un máximo del 40 por ciento de cuentas señaladas. Que dos predicciones se combinen no significa que el resultado sea mejor. Los nombres «precisión» y «recall» describen la política de corte, no una propiedad garantizada del algoritmo.

## Qué medimos

Reportamos AUC ROC, average precision (PR-AUC/AP), KS y Brier. Con los umbrales fijados calculamos precisión, recall, F1 y porcentaje de cuentas señaladas. También publicamos ganancias y lift por decil, además de calibración por cuantiles.

La prueba más cercana a una decisión operativa consiste en ordenar las cuentas y mirar qué porcentaje de eventos del mes siguiente concentra el 20 por ciento priorizado. Aplicamos exactamente ese criterio a todos los candidatos y a dos reglas que no requieren ML: ordenar por `utilizacion_reciente`, o por `max_atraso_6m` con desempate por utilización.

El informe publica la diferencia en puntos porcentuales entre cada modelo y la mejor regla sencilla. Aparece en `metricas.json`, `reports/resumen.md` y `reports/figures/valor_incremental.png`. No es uplift causal, ahorro ni dinero recuperado.

Identificamos por separado el candidato con mayor AP y el de mayor captura top 20 por ciento. Ganar retrospectivamente una métrica en test no autoriza a volver a optimizarlo con ese test. Una elección operativa necesitaría confirmación en otro período.

## Lo que no podemos concluir

Aquí no hay prueba temporal, intervalos de incertidumbre, verificación completa de cuándo estaban disponibles los datos ni evaluación de costes y capacidad del equipo de gestión. También faltan revisión de sesgos y seguridad, calibración por segmentos, seguimiento de PSI y drift, y validación de políticas de contacto.

Una predicción de default identifica correlaciones del historial con un evento futuro. No demuestra que llamar al cliente prevenga ese evento; para sostener esa afirmación habría que medir la intervención con un diseño experimental o causal.
