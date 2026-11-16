"""Carga verificable del ZIP oficial UCI #350; sin redistribución de datos."""

from __future__ import annotations

import io
import hashlib
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip"
SOURCE_DOI = "https://doi.org/10.24432/C55S3H"
# Hash del ZIP público usado para generar los resultados. Si UCI publica
# otra versión, no se recalculan inadvertidamente métricas distintas.
SOURCE_ZIP_SHA256 = "56c885f84457f6680f8438f02bfcdac9579323d8a94465ee5f26e32baa727602"
TARGET = "default payment next month"
STATUS_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
REQUIRED = ["ID", "LIMIT_BAL", TARGET, *STATUS_COLS, *BILL_COLS, *PAYMENT_COLS]


def descargar_uci(carpeta: Path) -> pd.DataFrame:
    """Lee el XLS original, almacenado solamente en el directorio ignorado data/."""
    carpeta.mkdir(parents=True, exist_ok=True)
    archivo = carpeta / "uci-350.zip"
    if not archivo.exists():
        print(f"Descargando fuente oficial UCI: {SOURCE_URL}")
        peticion = Request(SOURCE_URL, headers={"User-Agent": "QuorumRisk-public-demo/2.0"})
        with urlopen(peticion, timeout=90) as respuesta:
            contenido = respuesta.read()
        with zipfile.ZipFile(io.BytesIO(contenido)) as paquete:
            if not any(nombre.endswith(".xls") for nombre in paquete.namelist()):
                raise ValueError("El ZIP oficial no contiene el archivo XLS esperado")
        archivo.write_bytes(contenido)
    observado = hashlib.sha256(archivo.read_bytes()).hexdigest()
    if observado != SOURCE_ZIP_SHA256:
        raise ValueError(
            f"El ZIP UCI no coincide con la versión documentada: "
            f"SHA-256 observado {observado}, esperado {SOURCE_ZIP_SHA256}. "
            "No se publicarán métricas calculadas sobre otra versión del dataset."
        )
    with zipfile.ZipFile(archivo) as paquete:
        nombre = next((x for x in paquete.namelist() if x.endswith(".xls")), None)
        if nombre is None:
            raise ValueError("No se halló el XLS dentro del ZIP oficial UCI")
        df = pd.read_excel(io.BytesIO(paquete.read(nombre)), header=1, engine="xlrd")
    ausentes = set(REQUIRED).difference(df.columns)
    if ausentes:
        raise ValueError(f"Columnas requeridas ausentes: {sorted(ausentes)}")
    return df
