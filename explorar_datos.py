import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# Configuración para manejar datasets grandes
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", 50)


def explorar_estructura_inicial():
    """
    Explora la estructura inicial del dataset de morbilidad
    """
    print("=" * 60)
    print("ANÁLISIS INICIAL DEL DATASET DE MORBILIDAD")
    print("=" * 60)

    # Cargar solo las primeras filas para entender la estructura
    print("Cargando muestra inicial...")
    try:
        # Leer primeras 1000 filas para análisis inicial
        df_sample = pd.read_csv("data/morbilidad_2024.csv", nrows=1000, sep=';', encoding='utf-8')
        print(
            f"Muestra cargada: {df_sample.shape[0]} filas, {df_sample.shape[1]} columnas"
        )

        print("\n" + "=" * 40)
        print("INFORMACIÓN GENERAL")
        print("=" * 40)
        print(df_sample.info(memory_usage="deep"))

        print("\n" + "=" * 40)
        print("COLUMNAS DEL DATASET")
        print("=" * 40)
        for i, col in enumerate(df_sample.columns, 1):
            print(f"{i:2d}. {col}")

        print("\n" + "=" * 40)
        print("PRIMERAS 5 FILAS")
        print("=" * 40)
        print(df_sample.head())

        print("\n" + "=" * 40)
        print("TIPOS DE DATOS")
        print("=" * 40)
        print(df_sample.dtypes)

        print("\n" + "=" * 40)
        print("VALORES ÚNICOS POR COLUMNA")
        print("=" * 40)
        for col in df_sample.columns:
            n_unique = df_sample[col].nunique()
            print(f"{col}: {n_unique} valores únicos")

        print("\n" + "=" * 40)
        print("VALORES NULOS")
        print("=" * 40)
        missing_data = df_sample.isnull().sum()
        missing_percent = (missing_data / len(df_sample)) * 100
        missing_df = pd.DataFrame(
            {
                "Columna": missing_data.index,
                "Valores_Nulos": missing_data.values,
                "Porcentaje": missing_percent.values,
            }
        )
        print(missing_df[missing_df["Valores_Nulos"] > 0])

        # Intentar determinar el tamaño real del archivo
        print("\n" + "=" * 40)
        print("ESTIMACIÓN DEL TAMAÑO TOTAL")
        print("=" * 40)

        # Contar líneas totales del archivo (más rápido que cargar todo)
        with open("data/morbilidad_2024.csv", "r", encoding="utf-8") as f:
            total_lines = sum(1 for line in f)

        total_rows = total_lines - 1  # Restar header
        print(f"Filas totales estimadas: {total_rows:,}")
        print(f"Columnas: {df_sample.shape[1]}")
        print(f"Dataset estimado: {total_rows:,} x {df_sample.shape[1]}")

        return df_sample, total_rows

    except Exception as e:
        print(f"Error al cargar el archivo: {e}")
        return None, 0


def analizar_columnas_clave(df_sample):
    """
    Analiza las columnas más importantes para identificar variables objetivo
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE COLUMNAS CLAVE PARA ML")
    print("=" * 60)

    # Buscar columnas que podrían ser variables objetivo
    posibles_targets = []

    for col in df_sample.columns:
        col_lower = col.lower()
        if any(
            keyword in col_lower
            for keyword in ["diagnostico", "enfermedad", "morbi", "codigo", "cie"]
        ):
            posibles_targets.append(col)
            print(f"Posible variable objetivo: {col}")
            print(f"  - Valores únicos: {df_sample[col].nunique()}")
            print(f"  - Primeros valores: {df_sample[col].unique()[:5]}")
            print()

    # Buscar columnas demográficas
    demograficas = []
    for col in df_sample.columns:
        col_lower = col.lower()
        if any(
            keyword in col_lower
            for keyword in [
                "edad",
                "sexo",
                "genero",
                "distrito",
                "provincia",
                "departamento",
            ]
        ):
            demograficas.append(col)

    print("Columnas demográficas encontradas:")
    for col in demograficas:
        print(f"  - {col}: {df_sample[col].nunique()} valores únicos")

    return posibles_targets, demograficas


if __name__ == "__main__":
    # Explorar estructura inicial
    df_sample, total_rows = explorar_estructura_inicial()

    if df_sample is not None:
        # Analizar columnas clave
        targets, demograficas = analizar_columnas_clave(df_sample)

        # Guardar muestra para análisis posterior
        df_sample.to_csv("data/muestra_datos.csv", index=False)
        print(f"\nMuestra guardada en 'muestra_datos.csv' para análisis posterior")

        print("\n" + "=" * 60)
        print("RESUMEN PARA MODELO PREDICTIVO")
        print("=" * 60)
        print(f"- Dataset total: ~{total_rows:,} filas")
        print(f"- Columnas disponibles: {len(df_sample.columns)}")
        print(f"- Posibles variables objetivo: {len(targets)}")
        print(f"- Variables demográficas: {len(demograficas)}")
        print("\nListo para continuar con el análisis exploratorio completo.")
