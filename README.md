# Quórum Risk · Priorización transversal de riesgo de crédito

> ⚠️ **DATOS PÚBLICOS UCI, NO DE CLIENTES.** Demostración académica con
> *Default of Credit Card Clients* (Taiwán, 2005). Las métricas **no**
> representan resultados comerciales, eficacia en producción ni reducción
> causal del impago. El repositorio conserva por continuidad el nombre
> `alerta-temprana-cartera`, pero **NO demuestra una alerta temprana validada
> temporalmente**: su evaluación es estrictamente transversal.

## La conclusión antes del algoritmo

Para identificar a quién revisar entre los clientes **sin código positivo
de atraso registrado en septiembre**, una regla sencilla —priorizar por
**peor atraso de los seis meses anteriores**, con desempate por utilización—
captura **35,66 % de los defaults del mes siguiente** entre el 20 % de
cuentas prioritarias. La combinación de modelos captura **38,78 %**:
**+3,12 puntos porcentuales** en el test reservado.

La mejora observada **no valida su persistencia temporal** ni demuestra
que gestionar esas cuentas evite impagos. Sin un out-of-time real, intervalos
de incertidumbre y costos de gestión, **no recomendamos poner el ML en
producción** basándonos solo en esta prueba. El modelo tiene que ganarse
su lugar frente a una regla de negocio; si no demuestra valor neto, no entra.

**Pruebas:** [métricas completas](reports/metricas.json) ·
[comparación visual contra reglas simples](reports/figures/valor_incremental.png) ·
[informe del holdout](reports/resumen.md).

![Eventos futuros capturados en el 20% priorizado](reports/figures/valor_incremental.png)

## 1. Qué predice realmente

| Decisión | Definición auditable |
| --- | --- |
| Unidad | Cuenta/tarjeta de crédito del archivo académico UCI |
| Datos de observación | Códigos de pago, saldos, cuotas/pagos reportados **abril–septiembre de 2005** |
| Corte | Septiembre de 2005 (una sola fotografía común a todos) |
| Universo | `PAY_0 <= 0`: sin **código positivo** de retraso en septiembre; 0/-2 son ambiguos |
| Etiqueta positiva | `default payment next month = 1` según UCI, para el mes siguiente |
| Etiqueta negativa | `default payment next month = 0` |
| Exclusiones | `PAY_0 > 0`; target no binario, historia incompleta o límite inválido |

De **30.000** registros, **23.182** cumplen el universo; **3.207** tienen
evento positivo (**13,83 %**). No se interpreta el target como transición
contractual 0→30 DPD, ni como primer deterioro, ni como evento cuya fecha
exacta se haya observado. La definición está en
[docs/DEFINICION_TARGET.md](docs/DEFINICION_TARGET.md).

## 2. El trabajo antes del modelo

Construimos **25 variables originales**, solo con el historial disponible
hasta el corte de septiembre:

- **Mora histórica:** severidad, frecuencia, recencia, persistencia y tendencia.
- **Utilización:** saldo sobre cupo, nivel, máximo y volatilidad.
- **Pagos:** ratios frente a saldos, meses sin pago, tendencia y volatilidad.
- **Magnitudes relativas:** transformaciones logarítmicas de cupo y saldo.

Se excluyen de los predictores **ID, etiqueta, variables demográficas y
columnas crudas**. Los ratios no son conciliaciones contables; las
importancias no son relaciones causales. Detalle y lista positiva en
[docs/FEATURE_ENGINEERING.md](docs/FEATURE_ENGINEERING.md).

![Importancia por permutación](reports/figures/variables.png)

## 3. Tres estrategias y dos reglas sin ML

| Estrategia | Técnica | Selección operativa |
| --- | --- | --- |
| Precisión | HistGradientBoosting | Umbral de precisión, con cobertura mínima |
| Recall | Regresión logística escalada | Umbral de recall, con restricción de precisión |
| Combinación | Promedio fijo 50/50 | Umbral de F1; **no se presume superior** |
| Regla 1 | Mayor utilización reciente | Ranking determinista, sin ML |
| Regla 2 | Peor código positivo de atraso previo; desempate por utilización | Ranking determinista, sin ML |

Los tres candidatos compiten sobre el mismo **25 % de test reservado**:

| Estrategia | AUC ROC | PR-AUC (AP) | KS | Eventos capturados en top 20 % | Δ vs. mejor regla |
| --- | ---: | ---: | ---: | ---: | ---: |
| Precisión | 0,687 | 0,274 | 0,270 | 38,53 % | +2,87 pp |
| Recall | 0,667 | 0,262 | 0,243 | 37,03 % | +1,37 pp |
| Combinación | 0,685 | 0,277 | 0,259 | **38,78 %** | **+3,12 pp** |

La **regla sencilla ganadora captura 35,66 %**. La diferencia frente al ML
es **descriptiva**, no uplift causal ni ahorro: un cliente podría haber
incumplido incluso si se le hubiera contactado. La AP de la combinación
solo mejora ~0,0023 sobre el mejor candidato individual; también es una
diferencia pequeña sin evidencia de estabilidad.

![Precisión y recall](reports/figures/precision_recall.png)

## 4. Validación honesta: NO es out-of-time

Todos los registros comparten **un mismo corte de observación**. Por ello:

1. Se reserva **25 %** mediante split estratificado (`random_state=42`).
2. En el **75 %** restante se usa **CV estratificada de 4 folds**.
3. Los umbrales se seleccionan con probabilidades **out-of-fold**, sin
   consultar etiquetas del test.
4. Se informa AUC, AP, KS, precisión, recall, F1, Brier, calibración, lift
   y captura top 20 % contra las dos reglas simples.

**No se dispone de validación temporal**, porque no hay varias cohortes de
decisión. Ningún split por `ID` arregla esa limitación. Más detalle en
[docs/VALIDACION.md](docs/VALIDACION.md).

## 5. Estructura canónica Quórum Risk

```text
repo/
├── README.md
├── requirements.txt           # dependencias con ==
├── .gitignore                 # excluye data/, .env y modelos serializados
├── data/                      # solo local; ZIP original ignorado
├── docs/
│   ├── DEFINICION_TARGET.md
│   ├── FEATURE_ENGINEERING.md
│   ├── VALIDACION.md
│   └── ENTORNO.md
├── src/
│   ├── data/                  # descarga UCI y cohorte
│   ├── features/              # 25 variables documentadas
│   ├── models/
│   │   ├── precision/         # modelo.py, hiperparametros.yaml, README.md
│   │   ├── recall/            # modelo.py, hiperparametros.yaml, README.md
│   │   └── combinacion/       # modelo.py, hiperparametros.yaml, README.md
│   └── evaluacion/            # métricas, figuras e informes
├── reports/
│   ├── metricas.json
│   ├── tabla_lift.csv
│   ├── importancia_variables.csv
│   ├── resumen.md
│   └── figures/
├── tests/
└── run_demo.py                 # orquestador
```

## 6. Reproducción y trazabilidad

Entorno probado: **Python 3.12.14, Linux x86_64**. Las versiones **exactas**
de NumPy, pandas, scikit-learn, Matplotlib, xlrd y PyYAML están congeladas
en `requirements.txt`. Si difieren, el programa **se niega a generar**.
`reports/metricas.json` registra las versiones efectivas y el commit fuente.
Consulta [docs/ENTORNO.md](docs/ENTORNO.md).

```bash
git clone https://github.com/QuorumRisk/alerta-temprana-cartera.git
cd alerta-temprana-cartera
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

La primera ejecución **requiere internet** para descargar el archivo
original desde UCI. El ZIP se almacena en `data/` y **no se sube a GitHub**.
El cargador valida su **SHA-256**, evitando cambios silenciosos de fuente.
Los archivos de `reports/` contienen únicamente agregados y gráficos.
Reproducibilidad de métricas no significa validez temporal ni comercial.

## 7. Datos, crédito y licencia

**Fuente:** I-Cheng Yeh (2009). *Default of Credit Card Clients*. UCI Machine
Learning Repository. DOI: [10.24432/C55S3H](https://doi.org/10.24432/C55S3H).
[Ficha oficial UCI #350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients).

**Licencia de los datos:** [Creative Commons Attribution 4.0 International
(CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Se cita la
fuente y **no se redistribuye el dataset**. La licencia citada corresponde
a los datos UCI y no constituye una afirmación sobre licencia de código.

**Contacto:** [contacto@quorumrisk.co](mailto:contacto@quorumrisk.co) ·
[quorumrisk.co](https://quorumrisk.co)
