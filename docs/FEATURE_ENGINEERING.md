# Variables construidas a partir del historial

El archivo UCI reúne estados de pago, saldos y pagos de abril a septiembre de 2005. Tenemos una sola foto histórica de ese período, no una serie de cortes de decisión independientes. El trabajo consiste en resumir el comportamiento de esos seis meses sin entregar las columnas crudas al modelo.

## Qué información usamos

Partimos de `PAY_0` y `PAY_2...PAY_6` para el historial de estados, de `BILL_AMT1...6` para saldos y de `PAY_AMT1...6` para pagos. El target del mes siguiente está separado y solo se usa para entrenar y evaluar. Ni `ID` ni las características demográficas pasan a la matriz de entrenamiento.

La condición `PAY_0 <= 0` define el universo del ejercicio, pero no equivale a probar que la cuenta está al día. Los códigos 0 y -2 no están descritos con precisión contractual suficiente en la ficha.

## Los 25 indicadores

Para entender si hubo atrasos anteriores, construimos `max_atraso_6m`, `severidad_reciente`, `atrasos_6m`, `atrasos_3m` y `persistencia_atraso`. Miramos tanto la mayor severidad reportada como la repetición de señales, sin atribuirles días de mora que no conocemos.

La cercanía de esos episodios y su evolución se resumen con `recencia_ultimo_atraso`, `tendencia_atrasos` y `meses_estado_no_positivo`. Así distinguimos una señal antigua de una más reciente y observamos si el historial mejora o empeora.

El bloque de cupo contiene `limite_log`, `utilizacion_reciente`, `utilizacion_media`, `utilizacion_max`, `utilizacion_volatilidad`, `cambio_utilizacion`, `tendencia_utilizacion` y `meses_sobre_cupo`. No basta con saber el saldo de septiembre: interesa si el uso del límite fue sostenido, variable o creciente. Añadimos `importe_saldo_reciente_log` para representar la magnitud del saldo observado sin dejar que las escalas más grandes dominen por sí solas.

En pagos calculamos `pago_reciente_log`, `ratio_pago_saldo_agregado`, `ratio_pago_saldo_reciente`, `ratio_pago_saldo_mediano`, `meses_sin_pago`, `volatilidad_pago`, `tendencia_pago` y `pago_reciente_vs_historico`. Son señales aproximadas de hábito y esfuerzo de pago, no el resultado de conciliar movimientos con un sistema contable.

Son 25 variables en total. Su construcción y sus descripciones están en `src/features/construccion.py`, bajo `DESCRIPCIONES`. El proceso rechaza valores NaN e infinitos en la matriz.

## Precauciones

UCI contiene algunos saldos negativos y pagos superiores al saldo de un ciclo. Por eso el código aplica clipping y transformaciones logarítmicas explícitas. `max_atraso_6m` es el mayor código positivo del historial, no una medición verificada de DPD contractual.

Aunque respetamos el corte disponible en el archivo, no podemos auditar cuándo habría llegado realmente cada campo a un sistema de decisión. Tampoco interpretamos la importancia por permutación sobre test como causa: una variable correlacionada con otras puede parecer más o menos relevante sin que eso diga qué sucedería al intervenir.

La protección contra fuga usa solo los 25 predictores admitidos. Quedan fuera `ID`, sexo, edad, educación, estado civil, la etiqueta y cualquier campo no autorizado. En un proyecto contratado revisaríamos marcas temporales reales, revisiones retroactivas, preprocesamiento dentro de cada fold y pruebas en fechas posteriores antes de confiar en ese corte.
