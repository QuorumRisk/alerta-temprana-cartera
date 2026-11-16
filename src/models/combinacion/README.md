# Combinación de probabilidades

Promedio fijo **50/50** entre boosting y logística, con pesos explícitos en
`hiperparametros.yaml`. No entrena un tercer algoritmo ni ajusta sus pesos
en el test. Su umbral se elige con CV estratificada de desarrollo buscando
F1, con un máximo de **40 %** de contactos.

**La combinación se evalúa, no se presupone ganadora.** Un incremento de AP
pequeño no evidencia superioridad estable ni beneficio económico.
