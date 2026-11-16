# Combinación de probabilidades

Probamos una mezcla sencilla: 50 por ciento de la probabilidad estimada por boosting y 50 por ciento de la regresión logística. Los pesos están en `hiperparametros.yaml` y quedan fijados antes del test. No hay un tercer algoritmo entrenado ni una búsqueda de pesos a partir de los resultados reservados.

Elegimos el umbral con la validación cruzada estratificada del conjunto de desarrollo. El criterio es F1 y no permite señalar más del 40 por ciento de las cuentas.

La combinación compite con los modelos individuales y con las reglas sin ML. Una ventaja pequeña en AP no demuestra que sea estable ni que genere un beneficio económico. Si no gana con claridad, no tiene por qué usarse.
