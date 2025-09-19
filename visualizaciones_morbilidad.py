"""
VISUALIZACIONES Y ANÁLISIS AVANZADO - MORBILIDAD SAN MARTÍN
===========================================================
Script complementario para generar visualizaciones y análisis
avanzado de los patrones de morbilidad.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from datetime import datetime
import folium
from folium import plugins

warnings.filterwarnings("ignore")

# visualización
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


class VisualizadorMorbilidad:
    def __init__(self, archivo_datos):
        self.archivo_datos = archivo_datos
        self.df = None
        self.df_sample = None

    def cargar_muestra_representativa(self, n_sample=100000):
        """
        Carga una muestra representativa para visualizaciones
        """
        print(f"📊 Cargando muestra representativa de {n_sample:,} registros...")

        try:
            # Parámetros robustos para CSV con diferentes formatos
            csv_params = {
                'sep': ';',  # El archivo usa punto y coma como separador
                'encoding': 'utf-8',
                'on_bad_lines': 'skip',  # Saltar líneas problemáticas
                'dtype': str,  # Cargar todo como string inicialmente
                'engine': 'python'  # Usar motor Python para mejor manejo de errores
            }

            # Cargar muestra aleatoria
            # Primero contar líneas totales
            total_lines = sum(
                1 for line in open(self.archivo_datos, "r", encoding="utf-8")
            )
            total_rows = total_lines - 1

            # Calcular skip rows para muestra aleatoria
            skip_rows = sorted(
                np.random.choice(
                    range(1, total_rows), size=total_rows - n_sample, replace=False
                )
            )

            self.df_sample = pd.read_csv(self.archivo_datos, skiprows=skip_rows, **csv_params)
            print(f"✅ Muestra cargada: {len(self.df_sample):,} registros")

            # Convertir columnas numéricas
            self._convertir_tipos_numericos()

            return True

        except Exception as e:
            print(f"❌ Error al cargar muestra: {e}")
            # Fallback: cargar primeros N registros
            try:
                csv_params['nrows'] = n_sample
                self.df_sample = pd.read_csv(self.archivo_datos, **csv_params)
                print(f"⚠️  Usando primeros {n_sample:,} registros como muestra")

                # Convertir columnas numéricas
                self._convertir_tipos_numericos()

                return True
            except Exception as e2:
                print(f"❌ Error en fallback: {e2}")

                # Último fallback: intentar con delimitador comma
                try:
                    csv_params['sep'] = ','
                    csv_params['nrows'] = min(10000, n_sample)  # Muestra más pequeña
                    self.df_sample = pd.read_csv(self.archivo_datos, **csv_params)
                    print(f"⚠️  Usando muestra reducida de {len(self.df_sample):,} registros con separador comma")

                    # Convertir columnas numéricas
                    self._convertir_tipos_numericos()

                    return True
                except Exception as e3:
                    print(f"❌ Error final: {e3}")
                    return False

    def _convertir_tipos_numericos(self):
        """
        Convierte las columnas apropiadas a tipos numéricos
        """
        if self.df_sample is None:
            return

        # Columnas que deberían ser numéricas
        columnas_numericas = ['CASOS', 'ANIO', 'MES', 'LATITUD', 'LONGITUD',
                             'CODIGO_RED', 'CODIGO_MICRORED', 'CODIGO_UNICO']

        for col in columnas_numericas:
            if col in self.df_sample.columns:
                try:
                    # Convertir a numérico, errores como NaN
                    self.df_sample[col] = pd.to_numeric(self.df_sample[col], errors='coerce')
                except Exception as e:
                    print(f"⚠️  No se pudo convertir {col} a numérico: {e}")

        print(f"✅ Tipos de datos ajustados para {len(self.df_sample.columns)} columnas")

    def generar_dashboard_exploratorio(self):
        """
        Genera un dashboard completo de análisis exploratorio
        """
        if self.df_sample is None:
            print("❌ No hay datos cargados")
            return

        print("\n📈 GENERANDO DASHBOARD EXPLORATORIO")
        print("=" * 50)

        # Crear figura con subplots
        fig = plt.figure(figsize=(20, 16))

        # 1. Distribución de casos por mes
        plt.subplot(3, 3, 1)
        if "MES" in self.df_sample.columns and "CASOS" in self.df_sample.columns:
            casos_mes = self.df_sample.groupby("MES")["CASOS"].sum()
            casos_mes.plot(kind="bar", color="skyblue")
            plt.title("Casos por Mes", fontsize=12, fontweight="bold")
            plt.xlabel("Mes")
            plt.ylabel("Número de Casos")
            plt.xticks(rotation=45)

        # 2. Top 10 diagnósticos
        plt.subplot(3, 3, 2)
        if "DIAGNOSTICO" in self.df_sample.columns:
            top_diag = self.df_sample["DIAGNOSTICO"].value_counts().head(10)
            top_diag.plot(kind="barh", color="lightcoral")
            plt.title("Top 10 Diagnósticos", fontsize=12, fontweight="bold")
            plt.xlabel("Frecuencia")

        # 3. Distribución por género
        plt.subplot(3, 3, 3)
        if "GENERO" in self.df_sample.columns:
            genero_dist = self.df_sample["GENERO"].value_counts()
            plt.pie(
                genero_dist.values,
                labels=genero_dist.index,
                autopct="%1.1f%%",
                colors=["lightblue", "pink"],
            )
            plt.title("Distribución por Género", fontsize=12, fontweight="bold")

        # 4. Casos por grupo de edad
        plt.subplot(3, 3, 4)
        if "GRUPO_EDAD" in self.df_sample.columns:
            edad_casos = (
                self.df_sample.groupby("GRUPO_EDAD")["CASOS"].sum()
                if "CASOS" in self.df_sample.columns
                else self.df_sample["GRUPO_EDAD"].value_counts()
            )
            edad_casos.plot(kind="bar", color="lightgreen")
            plt.title("Casos por Grupo de Edad", fontsize=12, fontweight="bold")
            plt.xlabel("Grupo de Edad")
            plt.ylabel("Número de Casos")
            plt.xticks(rotation=45)

        # 5. Top provincias
        plt.subplot(3, 3, 5)
        if "PROVINCIA" in self.df_sample.columns:
            top_provincias = self.df_sample["PROVINCIA"].value_counts().head(10)
            top_provincias.plot(kind="bar", color="orange")
            plt.title("Top 10 Provincias", fontsize=12, fontweight="bold")
            plt.xlabel("Provincia")
            plt.ylabel("Número de Registros")
            plt.xticks(rotation=45)

        # 6. Distribución por capítulo CIE-10
        plt.subplot(3, 3, 6)
        if "CAPITULO" in self.df_sample.columns:
            # Obtener solo el número romano del capítulo
            self.df_sample["CAPITULO_NUM"] = self.df_sample["CAPITULO"].str.extract(
                r"([IVX]+)"
            )
            capitulo_dist = self.df_sample["CAPITULO_NUM"].value_counts().head(15)
            capitulo_dist.plot(kind="bar", color="purple", alpha=0.7)
            plt.title("Top 15 Capítulos CIE-10", fontsize=12, fontweight="bold")
            plt.xlabel("Capítulo")
            plt.ylabel("Frecuencia")
            plt.xticks(rotation=45)

        # 7. Casos por tipo de financiador
        plt.subplot(3, 3, 7)
        if "DESCRIPCION_FINANCIADOR" in self.df_sample.columns:
            financiador_dist = (
                self.df_sample["DESCRIPCION_FINANCIADOR"].value_counts().head(8)
            )
            financiador_dist.plot(kind="pie", autopct="%1.1f%%")
            plt.title("Distribución por Financiador", fontsize=12, fontweight="bold")

        # 8. Casos por categoría de establecimiento
        plt.subplot(3, 3, 8)
        if "CATEGORIA_EESS" in self.df_sample.columns:
            categoria_dist = self.df_sample["CATEGORIA_EESS"].value_counts()
            categoria_dist.plot(kind="bar", color="brown", alpha=0.7)
            plt.title("Casos por Categoría EESS", fontsize=12, fontweight="bold")
            plt.xlabel("Categoría")
            plt.ylabel("Número de Registros")
            plt.xticks(rotation=45)

        # 9. Heatmap de correlaciones (solo variables numéricas)
        plt.subplot(3, 3, 9)
        numeric_cols = self.df_sample.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.df_sample[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", center=0, square=True)
            plt.title(
                "Correlaciones Variables Numéricas", fontsize=12, fontweight="bold"
            )
        else:
            plt.text(
                0.5,
                0.5,
                "No hay suficientes\nvariables numéricas",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Correlaciones", fontsize=12, fontweight="bold")

        plt.tight_layout()
        plt.savefig("dashboard_morbilidad.png", dpi=300, bbox_inches="tight")
        print("📊 Dashboard guardado como 'dashboard_morbilidad.png'")
        plt.show()

    def generar_analisis_temporal(self):
        """
        Genera análisis temporal detallado
        """
        print("\n📅 ANÁLISIS TEMPORAL DETALLADO")
        print("=" * 50)

        if "MES" not in self.df_sample.columns:
            print("❌ No hay datos temporales disponibles")
            return

        # Crear serie temporal
        if "CASOS" in self.df_sample.columns:
            serie_temporal = (
                self.df_sample.groupby("MES")["CASOS"]
                .agg(["sum", "mean", "std"])
                .reset_index()
            )
        else:
            serie_temporal = (
                self.df_sample.groupby("MES").size().reset_index(name="frecuencia")
            )

        # Gráfico temporal
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Casos totales por mes
        axes[0, 0].plot(
            serie_temporal["MES"],
            (
                serie_temporal["sum"]
                if "sum" in serie_temporal.columns
                else serie_temporal["frecuencia"]
            ),
            marker="o",
            linewidth=2,
            markersize=8,
        )
        axes[0, 0].set_title("Evolución Temporal de Casos", fontweight="bold")
        axes[0, 0].set_xlabel("Mes")
        axes[0, 0].set_ylabel("Número de Casos")
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Casos por mes y género
        if "GENERO" in self.df_sample.columns:
            casos_genero = (
                self.df_sample.groupby(["MES", "GENERO"])["CASOS"].sum().unstack()
                if "CASOS" in self.df_sample.columns
                else self.df_sample.groupby(["MES", "GENERO"]).size().unstack()
            )
            casos_genero.plot(kind="bar", ax=axes[0, 1], stacked=True)
            axes[0, 1].set_title("Casos por Mes y Género", fontweight="bold")
            axes[0, 1].set_xlabel("Mes")
            axes[0, 1].set_ylabel("Número de Casos")
            axes[0, 1].legend(title="Género")

        # 3. Estacionalidad por grupo de edad
        if "GRUPO_EDAD" in self.df_sample.columns:
            # Seleccionar grupos de edad principales
            top_grupos_edad = self.df_sample["GRUPO_EDAD"].value_counts().head(5).index
            df_edad_filtrado = self.df_sample[
                self.df_sample["GRUPO_EDAD"].isin(top_grupos_edad)
            ]

            estacionalidad = (
                df_edad_filtrado.groupby(["MES", "GRUPO_EDAD"])["CASOS"].sum().unstack()
                if "CASOS" in df_edad_filtrado.columns
                else df_edad_filtrado.groupby(["MES", "GRUPO_EDAD"]).size().unstack()
            )
            estacionalidad.plot(ax=axes[1, 0])
            axes[1, 0].set_title("Estacionalidad por Grupo de Edad", fontweight="bold")
            axes[1, 0].set_xlabel("Mes")
            axes[1, 0].set_ylabel("Número de Casos")
            axes[1, 0].legend(
                title="Grupo de Edad", bbox_to_anchor=(1.05, 1), loc="upper left"
            )

        # 4. Promedio móvil
        if "sum" in serie_temporal.columns:
            # Calcular promedio móvil de 3 meses
            serie_temporal["promedio_movil"] = (
                serie_temporal["sum"].rolling(window=3).mean()
            )

            axes[1, 1].plot(
                serie_temporal["MES"],
                serie_temporal["sum"],
                label="Casos Totales",
                marker="o",
            )
            axes[1, 1].plot(
                serie_temporal["MES"],
                serie_temporal["promedio_movil"],
                label="Promedio Móvil (3 meses)",
                linewidth=2,
            )
            axes[1, 1].set_title("Tendencia con Promedio Móvil", fontweight="bold")
            axes[1, 1].set_xlabel("Mes")
            axes[1, 1].set_ylabel("Número de Casos")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("analisis_temporal.png", dpi=300, bbox_inches="tight")
        print("📅 Análisis temporal guardado como 'analisis_temporal.png'")
        plt.show()

    def generar_mapa_interactivo(self):
        """
        Genera mapa interactivo de casos por ubicación
        """
        print("\n🗺️  GENERANDO MAPA INTERACTIVO")
        print("=" * 50)

        if not all(col in self.df_sample.columns for col in ["LATITUD", "LONGITUD"]):
            print("❌ No hay datos de coordenadas disponibles")
            return

        # Filtrar datos con coordenadas válidas
        df_geo = self.df_sample.dropna(subset=["LATITUD", "LONGITUD"])

        if len(df_geo) == 0:
            print("❌ No hay registros con coordenadas válidas")
            return

        # Centro del mapa (San Martín)
        centro_lat = df_geo["LATITUD"].mean()
        centro_lon = df_geo["LONGITUD"].mean()

        # Crear mapa base
        mapa = folium.Map(
            location=[centro_lat, centro_lon], zoom_start=8, tiles="OpenStreetMap"
        )

        # Agregar marcadores por establecimiento
        if "NOMBRE_ESTABLECIMIENTO" in df_geo.columns:
            # Agrupar por establecimiento
            establecimientos = (
                df_geo.groupby(["NOMBRE_ESTABLECIMIENTO", "LATITUD", "LONGITUD"])
                .agg({"CASOS": "sum" if "CASOS" in df_geo.columns else "size"})
                .reset_index()
            )

            # Añadir marcadores
            for idx, row in establecimientos.head(
                50
            ).iterrows():  # Limitar a 50 para rendimiento
                casos = row["CASOS"] if "CASOS" in row else 1

                # Tamaño del marcador basado en número de casos
                radio = min(max(casos / 10, 5), 20)

                folium.CircleMarker(
                    location=[row["LATITUD"], row["LONGITUD"]],
                    radius=radio,
                    popup=f"<b>{row['NOMBRE_ESTABLECIMIENTO']}</b><br>Casos: {casos}",
                    tooltip=f"{row['NOMBRE_ESTABLECIMIENTO']}: {casos} casos",
                    color="red",
                    fillColor="red",
                    fillOpacity=0.6,
                ).add_to(mapa)

        # Añadir mapa de calor
        if len(df_geo) > 10:
            # Crear datos para heatmap
            heat_data = []
            for idx, row in df_geo.iterrows():
                weight = (
                    row["CASOS"] if "CASOS" in row and pd.notna(row["CASOS"]) else 1
                )
                heat_data.append([row["LATITUD"], row["LONGITUD"], weight])

            # Añadir heatmap
            plugins.HeatMap(heat_data, radius=15).add_to(mapa)

        # Guardar mapa
        mapa.save("mapa_morbilidad_interactivo.html")
        print("🗺️  Mapa interactivo guardado como 'mapa_morbilidad_interactivo.html'")

        return mapa

    def generar_analisis_diagnosticos(self):
        """
        Genera análisis detallado de diagnósticos
        """
        print("\n🔬 ANÁLISIS DETALLADO DE DIAGNÓSTICOS")
        print("=" * 50)

        if "DIAGNOSTICO" not in self.df_sample.columns:
            print("❌ No hay datos de diagnósticos")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Top diagnósticos
        top_diagnosticos = self.df_sample["DIAGNOSTICO"].value_counts().head(15)
        top_diagnosticos.plot(kind="barh", ax=axes[0, 0], color="skyblue")
        axes[0, 0].set_title("Top 15 Diagnósticos Más Frecuentes", fontweight="bold")
        axes[0, 0].set_xlabel("Frecuencia")

        # 2. Diagnósticos por capítulo CIE-10
        if "CAPITULO" in self.df_sample.columns:
            # Extraer número de capítulo
            self.df_sample["CAPITULO_NUM"] = self.df_sample["CAPITULO"].str.extract(
                r"([IVX]+)"
            )
            capitulos = self.df_sample["CAPITULO_NUM"].value_counts().head(10)
            capitulos.plot(kind="bar", ax=axes[0, 1], color="lightcoral")
            axes[0, 1].set_title("Diagnósticos por Capítulo CIE-10", fontweight="bold")
            axes[0, 1].set_xlabel("Capítulo")
            axes[0, 1].set_ylabel("Frecuencia")
            axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Diagnósticos por género (top 10)
        if "GENERO" in self.df_sample.columns:
            top_10_diag = self.df_sample["DIAGNOSTICO"].value_counts().head(10).index
            df_top_diag = self.df_sample[
                self.df_sample["DIAGNOSTICO"].isin(top_10_diag)
            ]

            diag_genero = pd.crosstab(df_top_diag["DIAGNOSTICO"], df_top_diag["GENERO"])
            diag_genero.plot(kind="bar", ax=axes[1, 0], stacked=True)
            axes[1, 0].set_title("Top 10 Diagnósticos por Género", fontweight="bold")
            axes[1, 0].set_xlabel("Diagnóstico")
            axes[1, 0].set_ylabel("Frecuencia")
            axes[1, 0].legend(title="Género")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Diagnósticos por grupo de edad
        if "GRUPO_EDAD" in self.df_sample.columns:
            top_5_diag = self.df_sample["DIAGNOSTICO"].value_counts().head(5).index
            df_top_5_diag = self.df_sample[
                self.df_sample["DIAGNOSTICO"].isin(top_5_diag)
            ]

            diag_edad = pd.crosstab(
                df_top_5_diag["GRUPO_EDAD"], df_top_5_diag["DIAGNOSTICO"]
            )
            diag_edad.plot(kind="bar", ax=axes[1, 1], stacked=True)
            axes[1, 1].set_title(
                "Top 5 Diagnósticos por Grupo de Edad", fontweight="bold"
            )
            axes[1, 1].set_xlabel("Grupo de Edad")
            axes[1, 1].set_ylabel("Frecuencia")
            axes[1, 1].legend(
                title="Diagnóstico", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig("analisis_diagnosticos.png", dpi=300, bbox_inches="tight")
        print("🔬 Análisis de diagnósticos guardado como 'analisis_diagnosticos.png'")
        plt.show()

    def generar_reporte_completo(self):
        """
        Genera un reporte completo con todas las visualizaciones
        """
        print("\n📋 GENERANDO REPORTE COMPLETO DE VISUALIZACIONES")
        print("=" * 60)

        # Cargar muestra si no está cargada
        if self.df_sample is None:
            if not self.cargar_muestra_representativa():
                return

        # Generar todas las visualizaciones
        print("1/5 Generando dashboard exploratorio...")
        self.generar_dashboard_exploratorio()

        print("\n2/5 Generando análisis temporal...")
        self.generar_analisis_temporal()

        print("\n3/5 Generando análisis de diagnósticos...")
        self.generar_analisis_diagnosticos()

        print("\n4/5 Generando mapa interactivo...")
        self.generar_mapa_interactivo()

        print("\n5/5 Generando estadísticas resumidas...")
        self._generar_estadisticas_resumen()

        print("\n✅ REPORTE COMPLETO GENERADO!")
        print("📁 Archivos generados:")
        print("  - dashboard_morbilidad.png")
        print("  - analisis_temporal.png")
        print("  - analisis_diagnosticos.png")
        print("  - mapa_morbilidad_interactivo.html")
        print("  - estadisticas_resumen.txt")

    def _generar_estadisticas_resumen(self):
        """Genera un archivo de texto con estadísticas resumidas"""

        with open("estadisticas_resumen.txt", "w", encoding="utf-8") as f:
            f.write("ESTADÍSTICAS RESUMIDAS - MORBILIDAD SAN MARTÍN 2024\n")
            f.write("=" * 60 + "\n\n")

            # Información general
            f.write(f"Total de registros analizados: {len(self.df_sample):,}\n")
            f.write(f"Período: 2024\n")
            f.write(f"Región: San Martín\n\n")

            # Estadísticas por columnas principales
            if "DIAGNOSTICO" in self.df_sample.columns:
                f.write(
                    f"Total diagnósticos únicos: {self.df_sample['DIAGNOSTICO'].nunique():,}\n"
                )
                f.write(
                    f"Diagnóstico más común: {self.df_sample['DIAGNOSTICO'].value_counts().index[0]}\n"
                )
                f.write(
                    f"Frecuencia del más común: {self.df_sample['DIAGNOSTICO'].value_counts().iloc[0]:,}\n\n"
                )

            if "GENERO" in self.df_sample.columns:
                f.write("Distribución por género:\n")
                for genero, count in self.df_sample["GENERO"].value_counts().items():
                    f.write(
                        f"  {genero}: {count:,} ({count/len(self.df_sample)*100:.1f}%)\n"
                    )
                f.write("\n")

            if "GRUPO_EDAD" in self.df_sample.columns:
                f.write("Distribución por grupos de edad:\n")
                for edad, count in (
                    self.df_sample["GRUPO_EDAD"].value_counts().head(10).items()
                ):
                    f.write(f"  {edad}: {count:,}\n")
                f.write("\n")

            if "PROVINCIA" in self.df_sample.columns:
                f.write("Top 5 provincias:\n")
                for provincia, count in (
                    self.df_sample["PROVINCIA"].value_counts().head(5).items()
                ):
                    f.write(f"  {provincia}: {count:,}\n")
                f.write("\n")

            if "CASOS" in self.df_sample.columns:
                f.write("Estadísticas de casos:\n")
                f.write(f"  Total casos: {self.df_sample['CASOS'].sum():,}\n")
                f.write(
                    f"  Promedio por registro: {self.df_sample['CASOS'].mean():.2f}\n"
                )
                f.write(f"  Mediana: {self.df_sample['CASOS'].median():.2f}\n")
                f.write(f"  Máximo: {self.df_sample['CASOS'].max():,}\n")
                f.write(f"  Mínimo: {self.df_sample['CASOS'].min():,}\n\n")

            f.write(
                f"Reporte generado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )


def main():
    """Función principal"""
    archivo_datos = "data/morbilidad_2024.csv"

    print("🎨 GENERADOR DE VISUALIZACIONES - MORBILIDAD SAN MARTÍN")
    print("=" * 60)

    # Crear visualizador
    viz = VisualizadorMorbilidad(archivo_datos)

    # Generar reporte completo
    viz.generar_reporte_completo()


if __name__ == "__main__":
    main()
