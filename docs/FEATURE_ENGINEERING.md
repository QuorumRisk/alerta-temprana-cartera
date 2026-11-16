# Ingeniería de variables · un corte transversal

> **Datos públicos UCI #350 (CC BY 4.0); no datos de clientes.** Se hace
> ingeniería sobre la única foto histórica de 2005. No es un pipeline
> *point-in-time* evaluado contra múltiples cohortes temporales.

## Corte, datos y exclusiones

- Observación histórica reportada: **abril a septiembre de 2005**.
- Los estados `PAY_0` (septiembre), `PAY_2...PAY_6`, `BILL_AMT1...6`,
  `PAY_AMT1...6` son materias primas de los predictores, no se insertan
  directamente en la matriz.
- El **target del mes siguiente** solo se utiliza para entrenamiento y
  evaluación. Ni `ID` ni variables demográficas entran al modelo.
- `PAY_0 <= 0` no implica automáticamente ausencia de mora contractual;
  los códigos 0/-2 tienen interpretación incompleta en la ficha.

## Bloques de variables y racional de negocio

| Bloque | Variables transformadas | Pregunta |
| --- | --- | --- |
| Severidad e historial | `max_atraso_6m`, `severidad_reciente`, `atrasos_6m`, `atrasos_3m`, `persistencia_atraso` | ¿Hubo deterioros recurrentes aunque en septiembre no haya código positivo? |
| Recencia y cambio | `recencia_ultimo_atraso`, `tendencia_atrasos`, `meses_estado_no_positivo` | ¿Qué tan reciente fue la señal y hacia dónde evoluciona? |
| Cupo y utilización | `limite_log`, `utilizacion_reciente`, `utilizacion_media`, `utilizacion_max`, `utilizacion_volatilidad`, `cambio_utilizacion`, `tendencia_utilizacion`, `meses_sobre_cupo` | ¿Cuánta presión existe sobre la línea? |
| Saldo | `importe_saldo_reciente_log` | ¿Qué magnitud relativa presenta la deuda observada? |
| Hábitos de pago | `pago_reciente_log`, `ratio_pago_saldo_agregado`, `ratio_pago_saldo_reciente`, `ratio_pago_saldo_mediano`, `meses_sin_pago`, `volatilidad_pago`, `tendencia_pago`, `pago_reciente_vs_historico` | ¿Qué persistencia o volatilidad tienen los pagos? |

**Total: 25 variables.** Todas se construyen en
`src/features/construccion.py` y cada una tiene descripción concreta en
`DESCRIPCIONES`. La función falla ante predictores NaN o infinitos.

## Limitaciones específicas del dato

- Los importes de saldos pueden ser negativos, los pagos pueden exceder el
  saldo de un ciclo y los ratios no son conciliaciones contables. Se usan
  clipping y transformaciones logarítmicas **declaradas en código**.
- `max_atraso_6m` es el máximo **código positivo** del historial: no
  necesariamente días de mora verificados contractualmente.
- La selección de cohorte usa el estado de septiembre y el target está
  registrado para el mes siguiente, pero **no se puede auditar la latencia
  real de los campos** con la ficha UCI.
- La importancia por permutación en test identifica asociación predictiva,
  no causalidad, y se ve afectada por variables correlacionadas.

## Política anti-fuga

Se excluyen `ID`, sexo, educación, estado civil, edad, target y cualquier
campo no admitido en los 25 predictores. La etiqueta futura jamás entra en
los bloques de variables. Para un encargo real se exigirá auditoría de
disponibilidad de cada dato en fecha de decisión, cambios retroactivos,
preprocesamiento dentro de folds y backtesting temporal posterior.
