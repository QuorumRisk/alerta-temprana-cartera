# Entorno usado para reproducir el caso

Trabajamos con datos públicos de UCI bajo licencia CC BY 4.0. Las cifras no proceden de clientes ni son evidencia comercial.

## Versiones del experimento

El entorno de referencia fue Python 3.12.14 sobre Linux 6.18.57, arquitectura x86_64, con glibc 2.41. Las dependencias directas instaladas fueron:

- NumPy 2.3.5 y pandas 2.2.3.
- scikit-learn 1.8.0.
- Matplotlib 3.10.8.
- xlrd 2.0.1, para leer el XLS de origen.
- PyYAML 6.0.3, para las configuraciones.

`requirements.txt` las fija con `==`. Antes de descargar o entrenar, `run_demo.py` comprueba que coincidan la versión de Python y las librerías directas. `reports/metricas.json` registra las versiones efectivas y el hash corto de la revisión fuente del código modular.

Es especialmente relevante fijar scikit-learn: `HistGradientBoostingClassifier` puede cambiar sus resultados entre versiones. Aun así, un `requirements.txt` con dependencias directas no congela SciPy, BLAS/OpenMP, fuentes, backend gráfico ni todo el sistema operativo. En otros entornos hay que volver a comprobar los resultados y, especialmente, los archivos PNG.

## Qué significa el hash

El campo `commit_codigo_fuente` identifica el commit del código predictivo auditado que generó el informe. No apunta al commit que contiene ese mismo JSON: incorporarlo de forma autorreferencial no sería posible, porque escribir el hash cambiaría el propio commit. La revisión fuente de esta entrega es `95f895c7`. El commit de publicación se consulta por separado.

## Cómo ejecutar

```bash
git clone https://github.com/QuorumRisk/alerta-temprana-cartera.git
cd alerta-temprana-cartera
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_demo.py
python -m unittest discover -s tests -v
```

En el primer arranque el programa descarga el ZIP oficial de UCI y lo conserva en `data/`, carpeta excluida de Git. No se publican las filas originales. La descarga se verifica contra el SHA-256 `56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602`. Si la fuente cambia, preferimos que la ejecución falle a mezclar versiones de los datos.

Después de ejecutar, se puede comparar con `sha256sum` el `metricas.json`, los CSV y las figuras versionadas. Una coincidencia byte a byte de imágenes solo puede garantizarse en el entorno en que se haya comprobado. Y obtener las mismas métricas no convierte esta evaluación transversal en una validación temporal o productiva.
