"""
MODELO PREDICTIVO DE MORBILIDAD - SAN MARTÍN 2024
==================================================
Sistema completo de machine learning para predecir patrones de morbilidad
optimizado para manejar millones de registros de manera eficiente.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
import warnings
import gc
from datetime import datetime
import pickle
import os
from collections import Counter

warnings.filterwarnings("ignore")

# Configuración para datasets grandes
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


class ModeloMorbilidad:
    def __init__(self, archivo_datos):
        """
        Inicializa el modelo predictivo de morbilidad
        """
        self.archivo_datos = archivo_datos
        self.df = None
        self.df_procesado = None
        self.modelo = None
        self.encoder = None
        self.scaler = None
        self.pipeline = None
        self.feature_names = []
        self.target_column = None

    def cargar_datos_por_chunks(self, chunk_size=50000):
        """
        Carga los datos por chunks para manejar archivos grandes eficientemente
        """
        print("🔄 Cargando datos por chunks...")
        chunks = []

        try:
            # Contar filas totales primero
            total_lines = sum(
                1 for line in open(self.archivo_datos, "r", encoding="utf-8")
            )
            total_rows = total_lines - 1
            print(f"📊 Total de registros estimados: {total_rows:,}")

            # Cargar por chunks
            chunk_reader = pd.read_csv(self.archivo_datos, chunksize=chunk_size)

            for i, chunk in enumerate(chunk_reader):
                print(f"📥 Procesando chunk {i+1} - {len(chunk)} registros")
                chunks.append(chunk)

                # Limitar memoria - solo cargar una muestra representativa para desarrollo
                if len(chunks) >= 20:  # ~1M registros para pruebas
                    print("⚠️  Limitando a 1M registros para optimización de memoria")
                    break

            self.df = pd.concat(chunks, ignore_index=True)
            print(
                f"✅ Datos cargados: {len(self.df):,} registros, {len(self.df.columns)} columnas"
            )

            # Liberar memoria
            del chunks
            gc.collect()

            return True

        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return False

    def analisis_exploratorio(self):
        """
        Realiza análisis exploratorio optimizado para grandes datasets
        """
        print("\n" + "=" * 60)
        print("📈 ANÁLISIS EXPLORATORIO DE DATOS")
        print("=" * 60)

        if self.df is None:
            print("❌ No hay datos cargados")
            return

        # Información general
        print(f"📊 Forma del dataset: {self.df.shape}")
        print(
            f"💾 Uso de memoria: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        )

        # Información de columnas
        print("\n🔍 Información de columnas:")
        print(self.df.info(memory_usage="deep"))

        # Valores nulos
        print("\n❌ Valores nulos:")
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        missing_df = pd.DataFrame(
            {
                "Columna": missing_data.index,
                "Valores_Nulos": missing_data.values,
                "Porcentaje": missing_percent.values,
            }
        ).sort_values("Valores_Nulos", ascending=False)

        print(missing_df[missing_df["Valores_Nulos"] > 0])

        # Análisis de distribuciones clave
        self._analizar_distribuciones()

        # Identificar patrones temporales
        self._analizar_temporales()

        # Análisis geográfico
        self._analizar_geografico()

    def _analizar_distribuciones(self):
        """Analiza las distribuciones principales"""
        print("\n📊 DISTRIBUCIONES PRINCIPALES")
        print("-" * 40)

        # Análisis de casos
        if "CASOS" in self.df.columns:
            print("Estadísticas de CASOS:")
            print(self.df["CASOS"].describe())

        # Top diagnósticos
        if "DIAGNOSTICO" in self.df.columns:
            print("\n🔝 Top 10 diagnósticos más frecuentes:")
            top_diagnosticos = self.df["DIAGNOSTICO"].value_counts().head(10)
            for diag, count in top_diagnosticos.items():
                print(f"  {diag}: {count:,}")

        # Distribución por género
        if "GENERO" in self.df.columns:
            print(f"\n👥 Distribución por género:")
            print(self.df["GENERO"].value_counts())

        # Distribución por grupo de edad
        if "GRUPO_EDAD" in self.df.columns:
            print(f"\n🎂 Distribución por grupo de edad:")
            print(self.df["GRUPO_EDAD"].value_counts())

    def _analizar_temporales(self):
        """Analiza patrones temporales"""
        print("\n📅 ANÁLISIS TEMPORAL")
        print("-" * 40)

        if "MES" in self.df.columns:
            print("Distribución por mes:")
            casos_por_mes = (
                self.df.groupby("MES")["CASOS"].sum()
                if "CASOS" in self.df.columns
                else self.df["MES"].value_counts().sort_index()
            )
            print(casos_por_mes)

    def _analizar_geografico(self):
        """Analiza distribución geográfica"""
        print("\n🗺️  ANÁLISIS GEOGRÁFICO")
        print("-" * 40)

        if "PROVINCIA" in self.df.columns:
            print("Top 5 provincias con más casos:")
            casos_por_provincia = (
                self.df.groupby("PROVINCIA")["CASOS"].sum().sort_values(ascending=False)
                if "CASOS" in self.df.columns
                else self.df["PROVINCIA"].value_counts()
            )
            print(casos_por_provincia.head())

        if "DISTRITO" in self.df.columns:
            print(f"\nTotal de distritos: {self.df['DISTRITO'].nunique()}")

    def definir_problema_predictivo(self):
        """
        Define el problema predictivo basado en los datos disponibles
        """
        print("\n" + "=" * 60)
        print("🎯 DEFINICIÓN DEL PROBLEMA PREDICTIVO")
        print("=" * 60)

        # Opción 1: Predecir categoría de diagnóstico principal
        if "CAPITULO" in self.df.columns:
            print("🔵 OPCIÓN 1: Predicción de Capítulo CIE-10")
            capitulos = self.df["CAPITULO"].value_counts()
            print(f"Capítulos únicos: {len(capitulos)}")
            print("Top 5 capítulos:")
            print(capitulos.head())

            # Simplificar a top N capítulos más frecuentes
            top_n = 10
            top_capitulos = capitulos.head(top_n).index.tolist()
            self.df["CAPITULO_SIMPLIFIED"] = self.df["CAPITULO"].apply(
                lambda x: x if x in top_capitulos else "OTROS"
            )

            print(f"\n✅ Variable objetivo definida: CAPITULO_SIMPLIFIED")
            print(f"Clases a predecir: {self.df['CAPITULO_SIMPLIFIED'].nunique()}")
            self.target_column = "CAPITULO_SIMPLIFIED"

        # Opción 2: Predecir riesgo alto/bajo basado en casos
        elif "CASOS" in self.df.columns:
            print("🔵 OPCIÓN 2: Predicción de Riesgo por Volumen de Casos")
            casos_median = self.df["CASOS"].median()
            self.df["RIESGO_ALTO"] = (self.df["CASOS"] > casos_median).astype(int)
            print(f"Umbral de riesgo alto: > {casos_median} casos")
            print(f"Distribución:")
            print(self.df["RIESGO_ALTO"].value_counts())
            self.target_column = "RIESGO_ALTO"

        else:
            print("❌ No se pudo definir una variable objetivo clara")
            return False

        return True

    def preparar_features(self):
        """
        Prepara las features para el modelo de machine learning
        """
        print("\n" + "=" * 60)
        print("🔧 PREPARACIÓN DE FEATURES")
        print("=" * 60)

        if self.target_column is None:
            print("❌ No se ha definido variable objetivo")
            return False

        # Seleccionar features relevantes
        feature_columns = []

        # Features demográficas
        demographic_features = ["GRUPO_EDAD", "GENERO"]
        feature_columns.extend(
            [col for col in demographic_features if col in self.df.columns]
        )

        # Features geográficas
        geo_features = ["PROVINCIA", "DISTRITO", "CATEGORIA_EESS"]
        feature_columns.extend([col for col in geo_features if col in self.df.columns])

        # Features temporales
        time_features = ["MES"]
        feature_columns.extend([col for col in time_features if col in self.df.columns])

        # Features de servicio
        service_features = ["DESCRIPCION_UPS", "DESCRIPCION_FINANCIADOR"]
        feature_columns.extend(
            [col for col in service_features if col in self.df.columns]
        )

        # Features étnicas
        ethnic_features = ["DESCRIPCION_ETNIA"]
        feature_columns.extend(
            [col for col in ethnic_features if col in self.df.columns]
        )

        print(f"✅ Features seleccionadas: {len(feature_columns)}")
        for i, feature in enumerate(feature_columns, 1):
            print(f"  {i}. {feature}")

        # Crear dataset de features
        self.df_procesado = self.df[feature_columns + [self.target_column]].copy()

        # Limpiar datos
        print(f"\n🧹 Limpieza de datos...")
        print(f"Registros antes de limpieza: {len(self.df_procesado):,}")

        # Eliminar filas con valores nulos en target
        self.df_procesado = self.df_procesado.dropna(subset=[self.target_column])

        # Manejar valores nulos en features
        for col in feature_columns:
            if self.df_procesado[col].isnull().sum() > 0:
                if self.df_procesado[col].dtype == "object":
                    self.df_procesado[col].fillna("DESCONOCIDO", inplace=True)
                else:
                    self.df_procesado[col].fillna(
                        self.df_procesado[col].median(), inplace=True
                    )

        print(f"Registros después de limpieza: {len(self.df_procesado):,}")

        # Limitar categorías con pocas observaciones
        for col in feature_columns:
            if self.df_procesado[col].dtype == "object":
                value_counts = self.df_procesado[col].value_counts()
                # Mantener solo categorías con al menos 100 observaciones
                frequent_values = value_counts[value_counts >= 100].index.tolist()
                self.df_procesado[col] = self.df_procesado[col].apply(
                    lambda x: x if x in frequent_values else "OTROS"
                )

        self.feature_names = feature_columns
        print(f"✅ Dataset preparado: {self.df_procesado.shape}")

        return True

    def entrenar_modelo(
        self, modelo_tipo="random_forest", test_size=0.2, random_state=42
    ):
        """
        Entrena el modelo de machine learning
        """
        print("\n" + "=" * 60)
        print("🚀 ENTRENAMIENTO DEL MODELO")
        print("=" * 60)

        if self.df_procesado is None:
            print("❌ Datos no preparados")
            return False

        # Preparar X e y
        X = self.df_procesado[self.feature_names]
        y = self.df_procesado[self.target_column]

        print(f"📊 Forma de X: {X.shape}")
        print(f"📊 Forma de y: {y.shape}")
        print(f"📊 Distribución de clases:")
        print(y.value_counts())

        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y if len(y.unique()) < 100 else None,
        )

        print(f"🔄 Conjunto de entrenamiento: {X_train.shape[0]:,}")
        print(f"🔄 Conjunto de prueba: {X_test.shape[0]:,}")

        # Crear pipeline de preprocesamiento
        categorical_features = [
            col for col in self.feature_names if X[col].dtype == "object"
        ]
        numerical_features = [
            col for col in self.feature_names if X[col].dtype != "object"
        ]

        print(f"📝 Features categóricas: {len(categorical_features)}")
        print(f"📝 Features numéricas: {len(numerical_features)}")

        # Preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", max_categories=50),
                    categorical_features,
                ),
            ]
        )

        # Seleccionar modelo
        if modelo_tipo == "random_forest":
            modelo = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            )
        elif modelo_tipo == "xgboost":
            modelo = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
            )
        else:
            print("❌ Tipo de modelo no soportado")
            return False

        # Crear pipeline completo
        self.pipeline = Pipeline(
            [("preprocessor", preprocessor), ("classifier", modelo)]
        )

        print(f"🎯 Entrenando modelo: {modelo_tipo}")
        print("⏳ Esto puede tomar varios minutos...")

        # Entrenar
        start_time = datetime.now()
        self.pipeline.fit(X_train, y_train)
        end_time = datetime.now()

        print(f"✅ Modelo entrenado en {(end_time - start_time).seconds} segundos")

        # Evaluación
        print("\n📊 EVALUACIÓN DEL MODELO")
        print("-" * 40)

        # Predicciones
        y_pred = self.pipeline.predict(X_test)
        y_pred_proba = self.pipeline.predict_proba(X_test)

        # Métricas
        accuracy = accuracy_score(y_test, y_pred)
        print(f"🎯 Precisión: {accuracy:.4f}")

        # Reporte de clasificación
        print("\n📋 Reporte de clasificación:")
        print(classification_report(y_test, y_pred))

        # Matriz de confusión
        print("\n🔍 Matriz de confusión:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

        # Importancia de features (si es posible)
        if hasattr(self.pipeline.named_steps["classifier"], "feature_importances_"):
            self._mostrar_importancia_features(X_train)

        # Guardar modelo
        self._guardar_modelo()

        print(f"\n✅ Modelo entrenado y evaluado exitosamente!")
        return True

    def _mostrar_importancia_features(self, X_train):
        """Muestra la importancia de las features"""
        try:
            # Obtener nombres de features después del preprocessor
            feature_names = []

            # Features numéricas
            numerical_features = [
                col for col in self.feature_names if X_train[col].dtype != "object"
            ]
            feature_names.extend(numerical_features)

            # Features categóricas (one-hot encoded)
            categorical_features = [
                col for col in self.feature_names if X_train[col].dtype == "object"
            ]
            if categorical_features:
                ohe = self.pipeline.named_steps["preprocessor"].named_transformers_[
                    "cat"
                ]
                ohe_feature_names = ohe.get_feature_names_out(categorical_features)
                feature_names.extend(ohe_feature_names)

            # Importancias
            importances = self.pipeline.named_steps["classifier"].feature_importances_

            # Crear DataFrame
            importance_df = pd.DataFrame(
                {"feature": feature_names, "importance": importances}
            ).sort_values("importance", ascending=False)

            print("\n🔝 Top 10 Features más importantes:")
            print(importance_df.head(10))

        except Exception as e:
            print(f"⚠️  No se pudo calcular importancia de features: {e}")

    def _guardar_modelo(self):
        """Guarda el modelo entrenado"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"modelo_morbilidad_{timestamp}.pkl"

            modelo_data = {
                "pipeline": self.pipeline,
                "feature_names": self.feature_names,
                "target_column": self.target_column,
                "timestamp": timestamp,
            }

            with open(filename, "wb") as f:
                pickle.dump(modelo_data, f)

            print(f"💾 Modelo guardado: {filename}")

        except Exception as e:
            print(f"⚠️  Error al guardar modelo: {e}")

    def predecir_nuevos_casos(self, datos_nuevos):
        """
        Realiza predicciones sobre nuevos datos
        """
        if self.pipeline is None:
            print("❌ Modelo no entrenado")
            return None

        try:
            predicciones = self.pipeline.predict(datos_nuevos)
            probabilidades = self.pipeline.predict_proba(datos_nuevos)

            return predicciones, probabilidades

        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return None, None

    def ejecutar_pipeline_completo(self, modelo_tipo="random_forest"):
        """
        Ejecuta el pipeline completo de machine learning
        """
        print("🚀 INICIANDO PIPELINE COMPLETO DE MACHINE LEARNING")
        print("=" * 60)

        # Paso 1: Cargar datos
        if not self.cargar_datos_por_chunks():
            return False

        # Paso 2: Análisis exploratorio
        self.analisis_exploratorio()

        # Paso 3: Definir problema predictivo
        if not self.definir_problema_predictivo():
            return False

        # Paso 4: Preparar features
        if not self.preparar_features():
            return False

        # Paso 5: Entrenar modelo
        if not self.entrenar_modelo(modelo_tipo=modelo_tipo):
            return False

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETADO EXITOSAMENTE!")
        print("=" * 60)
        print(f"📊 Registros procesados: {len(self.df_procesado):,}")
        print(f"🎯 Variable objetivo: {self.target_column}")
        print(f"📝 Features utilizadas: {len(self.feature_names)}")
        print(f"🤖 Modelo: {modelo_tipo}")

        return True


def main():
    """
    Función principal para ejecutar el análisis
    """
    archivo_datos = "data/morbilidad_2024.csv"

    print("🏥 SISTEMA PREDICTIVO DE MORBILIDAD - SAN MARTÍN 2024")
    print("=" * 60)

    modelo = ModeloMorbilidad(archivo_datos)

    exito = modelo.ejecutar_pipeline_completo(modelo_tipo="random_forest")

    if exito:
        print("\n¡Análisis completado exitosamente!")
        print("Los resultados y modelos han sido guardados.")

        print("\n💡 El modelo está listo para hacer predicciones.")
        print("💾 Puedes cargar el modelo guardado para usarlo en producción.")

    else:
        print("\n❌ Error en el pipeline de machine learning")


if __name__ == "__main__":
    main()
