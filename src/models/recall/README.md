# Candidato «recall»: regresión logística

Escalado aprendido **dentro del pipeline**, para que cada fold de desarrollo
ajuste su propio `StandardScaler`. Hiperparámetros en YAML.

El umbral se elige con predicciones out-of-fold maximizando recall, sujeto
a precisión **al menos tasa de eventos + 8 puntos** y hasta **45 %** de
contactos. No implica superioridad general ni costos operativos conocidos.
