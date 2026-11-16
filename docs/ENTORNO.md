# Entorno de referencia reproducible

> Este ejercicio utiliza **datos públicos UCI**, con licencia **CC BY 4.0**;
> no datos de clientes ni evidencia de eficacia comercial.

## Versiones exactas utilizadas

| Componente | Versión |
| --- | --- |
| Python | **3.12.14** |
| SO del experimento | **Linux 6.18.57, x86_64, glibc 2.41** |
| NumPy | **2.3.5** |
| pandas | **2.2.3** |
| scikit-learn | **1.8.0** |
| Matplotlib | **3.10.8** |
| xlrd (lectura de XLS) | **2.0.1** |
| PyYAML (configuraciones) | **6.0.3** |

`requirements.txt` fija estas seis dependencias directas con `==`. El
pipeline verifica automáticamente las versiones, incluida la versión
de Python, **antes de descargar datos o entrenar**. `metricas.json` guarda
un bloque `entorno` con versiones efectivamente instaladas y hash corto
de la revisión fuente del código modular.

`HistGradientBoostingClassifier` puede variar numéricamente entre versiones
de scikit-learn; por eso se congela su versión. También afectan potencialmente
las dependencias indirectas de NumPy/SciPy, BLAS/OpenMP, las fuentes y el
backend de Matplotlib. Fijar bibliotecas directas **no equivale** a tener
una imagen reproducible de todo el sistema operativo.

## Procedencia del hash

El atributo `commit_codigo_fuente` se refiere al **commit auditado con el
código predictivo** que sirve como origen del reporte, **no al commit que
contiene ese mismo JSON**. Un archivo no puede incluir el SHA de su propio
commit sin circularidad: editar el archivo cambia el SHA. El commit final
de publicación se proporciona por separado.

## Reproducción

```bash
git clone https://github.com/QuorumRisk/alerta-temprana-cartera.git
cd alerta-temprana-cartera
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

`run_demo.py` descarga **el ZIP oficial** desde UCI la primera vez y lo
almacena en `data/` (ignorado en Git). Por tanto, además del entorno fijado,
la reproducción depende de la **disponibilidad y contenido de esa fuente**.
No se versionan ni se redistribuyen filas originales del dataset.
El cargador exige el **SHA-256 exacto** del ZIP utilizado:
`56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`.
Si UCI cambia el archivo, el pipeline falla expresamente en lugar de
mezclar resultados procedentes de dos versiones distintas.

Se debe verificar la igualdad de `reports/metricas.json`, CSV y figuras
después de regenerar, mediante `sha256sum`. La igualdad de PNG byte a byte
solo se puede afirmar en el entorno donde se haya comprobado; en otros
sistemas las fuentes/renderizado pueden cambiar. La reproducibilidad de
métricas no constituye validación temporal o productiva.
