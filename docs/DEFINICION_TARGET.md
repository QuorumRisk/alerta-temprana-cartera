# Definición de target · priorización transversal de riesgo

> ⚠️ **Datos públicos UCI #350 (CC BY 4.0); no son datos de clientes
> de Quórum Risk.** Este ejercicio tiene una sola fecha de corte y no
> sustituye el acuerdo de definición de target con un negocio real.

## 1. Unidad de análisis y corte

- **Unidad:** cliente/tarjeta observada en el dataset UCI #350.
- **Ventana de observación:** abril a septiembre de 2005 (seis estados mensuales de pago, saldo y pago efectuado).
- **Corte de decisión:** cierre de septiembre de 2005, según la convención temporal publicada para las variables UCI.
- **Ventana de desempeño:** *mes siguiente*, representado por la etiqueta oficial `default payment next month`; no se conoce la fecha exacta de incumplimiento.
- **Evento positivo (`Y = 1`):** *default* reportado para el mes siguiente por UCI. No se redefine unilateralmente como 30, 60 o 90 DPD.
- **No evento (`Y = 0`):** ausencia de ese *default* reportado en la misma ventana.
- **Casos indeterminados:** target ausente, fuera de {0,1}, historial necesario incompleto o cupo inválido. El pipeline los aparta; en la fuente descargada no se detectaron casos de este tipo.

## 2. Definición de la cohorte preventiva

Se excluyen registros con `PAY_0 > 0`, pues presentan un **código positivo de retraso en septiembre**. Esos clientes requerirían una gestión correctiva o un caso distinto de clasificación.

Quedan los casos con `PAY_0 <= 0`, descritos **únicamente** como *sin atraso positivo registrado en septiembre*. **No debe decirse que todos están al día**: la ficha UCI no define completamente los códigos 0 y -2. Estos estados se conservan sin asignarles una interpretación causal/contable inventada.

El evento del mes siguiente puede estar influido por antecedentes de mora anteriores al corte; por tanto, el análisis es **ranking transversal de probabilidad de default del siguiente mes en una cohorte sin atraso positivo actual**, no una prueba rigurosa de *primer deterioro*. La predicción prospectiva de la etiqueta **no** convierte en *out-of-time* la validación entre filas de un mismo corte.

## 3. Ingeniería de variables sin fuga temporal intencional

- Estados de pago: intensidad, recencia, tendencia, persistencia y frecuencia de códigos positivos.
- Saldo y cupo: ratios de utilización, nivel reciente, evolución y volatilidad.
- Pagos: frecuencia de pagos reportados, pagos relativos a saldos, tendencia e inestabilidad.
- Los ratios de pagos/saldos son **proxies**, pues los ciclos exactos y la conciliación mensual no están detallados.
- Ni `ID` ni la etiqueta ni variables de octubre entran al entrenamiento. Se excluyen sexo, edad, educación y estado civil para mantener un ejemplo enfocado en comportamiento y evitar dependencias demográficas no justificadas.

## 4. Validación y elección de estrategia

No existe una dimensión de **fecha de corte distinta por cliente**. Un split ordenado por ID no es OOT. Se emplea una muestra final reservada 25% estratificada y validación cruzada estratificada de cuatro folds sobre el 75% de desarrollo.

Los umbrales se ajustan exclusivamente con probabilidades *out-of-fold* de desarrollo:

- **Precisión:** maximiza precisión bajo un mínimo de 5% de clientes señalados y recall de 12%.
- **Recall:** maximiza recall sujeto a una precisión mínima de tasa base + 8 puntos porcentuales y un máximo de 45% de clientes señalados.
- **Combinación:** promedio simple 50/50, umbral de mejor F1 con un máximo de 40% de clientes señalados. **No es una técnica superior por definición.**

Estas restricciones ilustran objetivos operativos alternativos, **no un mandato de negocio ni una optimización económica real**. El 25% reservado sirve para evaluación final, sin retocar esos umbrales tras conocer las métricas. La captura top 20 % se compara con reglas sin ML basadas en utilización o peor estado de pago histórico (véase `docs/VALIDACION.md`).

## 5. Qué exige un proyecto real

1. Definición conjunta del target, DPD, eventos y horizontes (por ejemplo, transición 0→30 DPD), fechas de disponibilidad real y exclusiones.
2. Contratos de datos, reconciliación de flujos, identificadores y auditoría de posibles fugas de información.
3. Backtesting *out-of-time* con múltiples cohortes históricas y, si procede, separación por entidad/persona.
4. Capacidad comercial de contacto, costes, intervenciones y medición causal del impacto.
5. Validación de sesgos, cumplimiento, calidad, explicabilidad y aprobaciones pertinentes.
6. Seguimiento de drift y PSI, desempeño, calibración y procedimientos de reentrenamiento.

**Regla del proyecto:** ningún resultado offline demuestra por sí solo recuperación de dinero, reducción de mora o retorno económico.
