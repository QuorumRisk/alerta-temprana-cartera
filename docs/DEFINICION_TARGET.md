# Definición del evento y de la población

Este caso usa los datos públicos UCI #350, publicados bajo licencia CC BY 4.0. No contienen información de clientes de Quórum Risk. Hay una sola fecha de corte y, por tanto, la definición no sustituye la que tendríamos que acordar con un negocio real.

## Qué cuenta observamos

La unidad de análisis es una tarjeta o cliente del archivo UCI. Vemos seis meses de estados de pago, saldos y pagos efectuados, entre abril y septiembre de 2005. La decisión se sitúa al cierre de septiembre, según la convención de la fuente.

Dejamos fuera las cuentas con `PAY_0 > 0`, porque ya reportan un código positivo de retraso en septiembre. Este ejemplo se concentra en las cuentas con `PAY_0 <= 0`. No afirmamos que todas estén al día: la documentación de UCI no aclara por completo el significado de los códigos 0 y -2.

## Qué queremos predecir

La etiqueta oficial es `default payment next month`. Si es 1, contamos un evento positivo en el mes siguiente; si es 0, un no evento. El dataset no informa la fecha exacta del incumplimiento, así que no convertimos esa etiqueta en una transición inventada de 0 a 30, 60 o 90 DPD.

Cuando faltan la etiqueta, los campos necesarios de historial o un límite de crédito válido, se excluye la observación. También se descartan etiquetas distintas de 0 y 1. En la descarga utilizada no aparecieron casos indeterminados de ese tipo.

Es un ranking transversal de probabilidad de default del mes siguiente entre cuentas sin código positivo de atraso actual. No prueba cuándo comenzó un deterioro. Tampoco convierte la partición de clientes de septiembre en una validación temporal.

## Información que entra

Construimos indicadores de frecuencia, intensidad, recencia y tendencia de los códigos históricos de atraso; saldos y utilización de cupo; y estabilidad y proporciones de pagos. Los cocientes entre pagos y saldos son aproximaciones, no conciliaciones contables de ciclos que la fuente no documenta en detalle.

El modelo no recibe `ID`, la etiqueta del mes siguiente ni ninguna variable de octubre. Tampoco incorpora sexo, edad, educación o estado civil, porque aquí queremos examinar comportamiento sin introducir dependencias demográficas que no hemos justificado.

## Cómo juzgamos el resultado

Con un solo corte, reservamos el 25 por ciento de las cuentas de forma estratificada y trabajamos sobre el 75 por ciento restante con cuatro folds también estratificados. Los umbrales salen únicamente de predicciones out-of-fold del conjunto de desarrollo.

La estrategia de precisión busca la mayor precisión posible, pero exige señalar al menos el 5 por ciento de las cuentas y alcanzar un recall del 12 por ciento. La de recall busca cobertura con una precisión mínima de tasa base más 8 puntos porcentuales y un máximo del 45 por ciento de cuentas señaladas. Para la combinación, promediamos 50/50 las probabilidades y elegimos por F1, sin superar el 40 por ciento de contactos. Estas son restricciones ilustrativas, no decisiones de una operación comercial.

El 25 por ciento de test queda reservado para evaluar, no para corregir umbrales. También contrastamos la captura de eventos en el 20 por ciento priorizado frente a reglas sencillas basadas en utilización y en el peor código histórico de pago. El procedimiento está en [VALIDACION.md](VALIDACION.md).

## Qué falta para usarlo en un negocio

Antes de implantar algo así habría que acordar con Riesgo o Cobranza el evento real, sus días de mora, horizonte y exclusiones; auditar fechas de disponibilidad y cambios retroactivos; y comprobar varias cohortes futuras mediante backtesting out-of-time. También necesitaríamos evaluar sesgos, cumplimiento y aprobaciones, capacidad de contacto, costes de gestión, calibración, drift, PSI y posibles necesidades de reentrenamiento.

Un resultado offline no demuestra que contactar evite impagos, reduzca mora o recupere dinero. Eso requiere evaluar intervenciones reales.
