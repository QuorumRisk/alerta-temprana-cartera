# Estrategia orientada a recall

Aquí usamos una regresión logística con escalado. `StandardScaler` está dentro del pipeline, de modo que cada fold de desarrollo aprende su propio escalado y no utiliza información del fold que se evalúa. Los hiperparámetros quedan en YAML.

El corte se define con las predicciones out-of-fold. Buscamos cubrir la mayor cantidad de eventos, pero exigimos una precisión equivalente, como mínimo, a la tasa de eventos más 8 puntos porcentuales. No podemos señalar más del 45 por ciento de las cuentas.

Esta restricción es ilustrativa: un equipo real decidiría cuánto puede contactar y a qué costo. Tampoco significa que la regresión logística siempre supere a los demás modelos en recall.
