"""
SISTEMA MULTIMODAL PARA MÚLTIPLES DATASETS
=========================================
Pipeline escalable para procesar diferentes tipos de datasets de salud
"""

import pandas as pd
import numpy as np
import os
from modelo_predictivo_morbilidad import ModeloMorbilidad
import pickle
from datetime import datetime


class GestorMultiplesDatasets:
    def __init__(self, directorio_datos="data/"):
        self.directorio_datos = directorio_datos
        self.modelos_entrenados = {}
        self.resultados = {}

    def detectar_datasets(self):
        """
        Detecta automáticamente todos los datasets CSV en el directorio
        """
        datasets = []
        for archivo in os.listdir(self.directorio_datos):
            if archivo.endswith(".csv"):
                datasets.append(archivo)

        print(f"📁 Datasets encontrados: {len(datasets)}")
        for i, dataset in enumerate(datasets, 1):
            print(f"  {i}. {dataset}")

        return datasets

    def analizar_estructura_dataset(self, archivo_dataset):
        """
        Analiza la estructura de un nuevo dataset para adaptarlo
        """
        print(f"\n🔍 ANALIZANDO: {archivo_dataset}")
        print("=" * 50)

        try:
            # Cargar muestra para análisis
            df_sample = pd.read_csv(
                os.path.join(self.directorio_datos, archivo_dataset), nrows=1000
            )

            print(f"📊 Forma: {df_sample.shape}")
            print(f"📝 Columnas encontradas:")

            # Clasificar tipos de columnas
            columnas_demograficas = []
            columnas_geograficas = []
            columnas_temporales = []
            columnas_medicas = []
            columnas_numericas = []

            for col in df_sample.columns:
                col_lower = col.lower()

                # Demográficas
                if any(
                    term in col_lower for term in ["edad", "genero", "sexo", "etnia"]
                ):
                    columnas_demograficas.append(col)

                # Geográficas
                elif any(
                    term in col_lower
                    for term in [
                        "provincia",
                        "distrito",
                        "departamento",
                        "lat",
                        "lon",
                        "coord",
                    ]
                ):
                    columnas_geograficas.append(col)

                # Temporales
                elif any(
                    term in col_lower
                    for term in ["fecha", "mes", "año", "anio", "tiempo"]
                ):
                    columnas_temporales.append(col)

                # Médicas
                elif any(
                    term in col_lower
                    for term in [
                        "diagnostico",
                        "enfermedad",
                        "codigo",
                        "cie",
                        "morbi",
                        "sintoma",
                    ]
                ):
                    columnas_medicas.append(col)

                # Numéricas
                elif df_sample[col].dtype in ["int64", "float64"]:
                    columnas_numericas.append(col)

            estructura = {
                "demograficas": columnas_demograficas,
                "geograficas": columnas_geograficas,
                "temporales": columnas_temporales,
                "medicas": columnas_medicas,
                "numericas": columnas_numericas,
            }

            print(
                f"  👥 Demográficas ({len(columnas_demograficas)}): {columnas_demograficas}"
            )
            print(
                f"  🗺️  Geográficas ({len(columnas_geograficas)}): {columnas_geograficas}"
            )
            print(
                f"  📅 Temporales ({len(columnas_temporales)}): {columnas_temporales}"
            )
            print(f"  🏥 Médicas ({len(columnas_medicas)}): {columnas_medicas}")
            print(f"  🔢 Numéricas ({len(columnas_numericas)}): {columnas_numericas}")

            return estructura, df_sample

        except Exception as e:
            print(f"❌ Error analizando {archivo_dataset}: {e}")
            return None, None

    def entrenar_modelo_adaptativo(self, archivo_dataset, tipo_modelo="random_forest"):
        """
        Entrena un modelo adaptado a la estructura específica del dataset
        """
        print(f"\n🚀 ENTRENANDO MODELO PARA: {archivo_dataset}")
        print("=" * 50)

        # analisis de estructura
        estructura, df_sample = self.analizar_estructura_dataset(archivo_dataset)
        if estructura is None:
            return False

        # creación de modelo
        ruta_completa = os.path.join(self.directorio_datos, archivo_dataset)
        modelo = ModeloMorbilidad(ruta_completa)

        # Ejecutar pipeline
        exito = modelo.ejecutar_pipeline_completo(modelo_tipo=tipo_modelo)

        if exito:
            # guardar info al modelo
            nombre_modelo = archivo_dataset.replace(".csv", "").replace("-", "_")
            self.modelos_entrenados[nombre_modelo] = {
                "modelo": modelo,
                "archivo_fuente": archivo_dataset,
                "estructura": estructura,
                "fecha_entrenamiento": datetime.now().isoformat(),
                "tipo_modelo": tipo_modelo,
            }

            print(f"✅ Modelo para {archivo_dataset} entrenado exitosamente!")
            return True
        else:
            print(f"❌ Error entrenando modelo para {archivo_dataset}")
            return False

    def procesar_todos_los_datasets(self):
        """
        Procesa automáticamente todos los datasets encontrados
        """
        print("🔄 PROCESANDO TODOS LOS DATASETS")
        print("=" * 60)

        datasets = self.detectar_datasets()

        if not datasets:
            print("❌ No se encontraron datasets CSV")
            return

        resultados_procesamiento = []

        for dataset in datasets:
            print(f"\n⏳ Procesando {dataset}...")

            try:
                exito = self.entrenar_modelo_adaptativo(dataset)

                resultado = {
                    "dataset": dataset,
                    "procesado_exitosamente": exito,
                    "timestamp": datetime.now().isoformat(),
                }

                if exito:
                    print(f"✅ {dataset} procesado correctamente")
                else:
                    print(f"❌ Error procesando {dataset}")

                resultados_procesamiento.append(resultado)

            except Exception as e:
                print(f"❌ Error crítico con {dataset}: {e}")
                resultados_procesamiento.append(
                    {
                        "dataset": dataset,
                        "procesado_exitosamente": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Guardar resumen
        self._generar_resumen_procesamiento(resultados_procesamiento)

        print(f"\n🎉 PROCESAMIENTO COMPLETADO!")
        print(
            f"✅ Exitosos: {sum(1 for r in resultados_procesamiento if r['procesado_exitosamente'])}"
        )
        print(
            f"❌ Fallidos: {sum(1 for r in resultados_procesamiento if not r['procesado_exitosamente'])}"
        )

    def _generar_resumen_procesamiento(self, resultados):
        """Genera un resumen del procesamiento de múltiples datasets"""

        with open("resumen_procesamiento_datasets.txt", "w", encoding="utf-8") as f:
            f.write("RESUMEN DE PROCESAMIENTO - MÚLTIPLES DATASETS\n")
            f.write("=" * 60 + "\n\n")
            f.write(
                f"Fecha de procesamiento: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"Total de datasets procesados: {len(resultados)}\n\n")

            exitosos = [r for r in resultados if r["procesado_exitosamente"]]
            fallidos = [r for r in resultados if not r["procesado_exitosamente"]]

            f.write(f"✅ PROCESADOS EXITOSAMENTE ({len(exitosos)}):\n")
            f.write("-" * 40 + "\n")
            for resultado in exitosos:
                f.write(f"  • {resultado['dataset']}\n")
            f.write("\n")

            if fallidos:
                f.write(f"❌ PROCESAMIENTO FALLIDO ({len(fallidos)}):\n")
                f.write("-" * 40 + "\n")
                for resultado in fallidos:
                    f.write(f"  • {resultado['dataset']}")
                    if "error" in resultado:
                        f.write(f" - Error: {resultado['error']}")
                    f.write("\n")
                f.write("\n")

            f.write("MODELOS ENTRENADOS:\n")
            f.write("-" * 40 + "\n")
            for nombre, info in self.modelos_entrenados.items():
                f.write(f"• Modelo: {nombre}\n")
                f.write(f"  - Archivo fuente: {info['archivo_fuente']}\n")
                f.write(f"  - Tipo: {info['tipo_modelo']}\n")
                f.write(f"  - Fecha: {info['fecha_entrenamiento']}\n")
                f.write(f"  - Variable objetivo: {info['modelo'].target_column}\n")
                f.write(f"  - Features: {len(info['modelo'].feature_names)}\n\n")

        print("📄 Resumen guardado en: resumen_procesamiento_datasets.txt")

    def cargar_modelo_entrenado(self, nombre_modelo):
        """
        Carga un modelo previamente entrenado
        """
        if nombre_modelo in self.modelos_entrenados:
            return self.modelos_entrenados[nombre_modelo]["modelo"]

        # Buscar archivo .pkl
        for archivo in os.listdir("."):
            if (
                archivo.startswith("modelo_")
                and nombre_modelo in archivo
                and archivo.endswith(".pkl")
            ):
                try:
                    with open(archivo, "rb") as f:
                        modelo_data = pickle.load(f)
                    print(f"✅ Modelo {nombre_modelo} cargado desde {archivo}")
                    return modelo_data
                except Exception as e:
                    print(f"❌ Error cargando {archivo}: {e}")

        print(f"❌ Modelo {nombre_modelo} no encontrado")
        return None

    def comparar_modelos(self):
        """
        Compara el rendimiento de todos los modelos entrenados
        """
        if not self.modelos_entrenados:
            print("❌ No hay modelos entrenados para comparar")
            return

        print("\n📊 COMPARACIÓN DE MODELOS")
        print("=" * 50)

        for nombre, info in self.modelos_entrenados.items():
            modelo = info["modelo"]
            print(f"\n🤖 {nombre}:")
            print(f"  📁 Dataset: {info['archivo_fuente']}")
            print(f"  🎯 Variable objetivo: {modelo.target_column}")
            print(f"  📝 Features: {len(modelo.feature_names)}")
            print(
                f"  📊 Registros procesados: {len(modelo.df_procesado) if modelo.df_procesado is not None else 'N/A'}"
            )
            print(f"  📅 Entrenado: {info['fecha_entrenamiento']}")


# Función principal para usar con múltiples datasets
def main():
    print("🎯 SISTEMA MULTIMODAL DE MACHINE LEARNING")
    print("=" * 60)

    gestor = GestorMultiplesDatasets()

    # Opción 1: Procesar todos los datasets automáticamente
    gestor.procesar_todos_los_datasets()

    # Opción 2: Comparar modelos (descomenta si quieres usar)
    gestor.comparar_modelos()


if __name__ == "__main__":
    main()
