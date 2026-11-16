# Candidato «precisión»: HistGradientBoosting

Se ajusta sobre el 75 % de desarrollo (variables transformadas). La selección
del umbral **no forma parte de la función de entrenamiento**: se efectúa con
probabilidades out-of-fold de cuatro folds estratificados, maximizando
precisión y exigiendo **al menos 5 % de contactos y 12 % de recall**.

`hiperparametros.yaml` fija las opciones del estimador, incluida la semilla.
El nombre «precisión» describe la **política de umbral**, no garantiza que
este algoritmo sea siempre el más preciso. No se ajusta en el 25 % de test.
