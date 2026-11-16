# Estrategia orientada a precisión

Este candidato utiliza `HistGradientBoosting` sobre las variables construidas. Se entrena con el 75 por ciento destinado a desarrollo. Sus opciones, incluida la semilla, quedan registradas en `hiperparametros.yaml`.

El nombre de la carpeta describe la decisión sobre el umbral. Con las probabilidades out-of-fold de cuatro particiones estratificadas buscamos la máxima precisión posible, siempre que se señale al menos el 5 por ciento de las cuentas y se alcance un recall del 12 por ciento. El umbral no es parte del ajuste del estimador.

El 25 por ciento de test se usa únicamente para comprobar el resultado. No afirmamos que boosting sea siempre el algoritmo de mayor precisión: esa prioridad depende de los datos y de cuánto del universo puede gestionar el negocio.
