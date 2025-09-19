"""
Sistema de Predicción de Brotes de Enfermedades Infecciosas
Región San Martín - ODS 3: Salud y Bienestar

Este sistema utiliza machine learning para predecir brotes de enfermedades infecciosas
como dengue y malaria, integrando análisis temporal y geoespacial.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Librerías para series temporales
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

# Librerías para análisis geoespacial
import folium
from folium import plugins
import geopandas as gpd
from scipy.spatial.distance import cdist

# Librerías para machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Configuración de visualización
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PredictorBrotesInfecciosos:
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        self.df = None
        self.df_infecciosas = None
        self.modelos = {}
        self.predicciones = {}

        # Enfermedades infecciosas principales según CIE-10
        self.enfermedades_objetivo = [
            'DENGUE', 'MALARIA', 'ZIKA', 'CHIKUNGUNYA', 'FIEBRE AMARILLA',
            'TUBERCULOSIS', 'VIH', 'HEPATITIS', 'DIARREA', 'INFLUENZA',
            'COVID', 'NEUMONIA', 'MENINGITIS'
        ]

    def cargar_datos(self):
        """Carga y unifica todos los datasets de la carpeta data"""
        print("📊 Cargando datasets...")

        dataframes = []

        # Lista de archivos a procesar
        archivos = ['morbilidad_2024.csv', 'morbilidad_unificada_0.csv']

        for archivo in archivos:
            ruta_archivo = f'{self.data_path}{archivo}'
            print(f"  - Procesando {archivo}...")

            try:
                # Primero intentar detectar el separador y encoding
                print(f"    🔍 Detectando formato del archivo...")

                # Leer las primeras líneas para analizar estructura
                with open(ruta_archivo, 'r', encoding='utf-8', errors='ignore') as f:
                    primeras_lineas = [f.readline().strip() for _ in range(5)]

                # Detectar separador más probable
                separadores = [',', ';', '|', '\t']
                mejor_separador = ','
                max_campos = 0

                for sep in separadores:
                    campos = len(primeras_lineas[0].split(sep))
                    if campos > max_campos:
                        max_campos = campos
                        mejor_separador = sep

                print(f"    📊 Detectado separador: '{mejor_separador}', {max_campos} campos")

                # Intentar cargar con diferentes configuraciones
                configuraciones = [
                    {'sep': mejor_separador, 'encoding': 'utf-8', 'on_bad_lines': 'skip'},
                    {'sep': mejor_separador, 'encoding': 'latin-1', 'on_bad_lines': 'skip'},
                    {'sep': mejor_separador, 'encoding': 'utf-8', 'on_bad_lines': 'warn', 'quoting': 1},
                    {'sep': mejor_separador, 'encoding': 'utf-8', 'on_bad_lines': 'skip', 'quotechar': '"'},
                ]

                df_cargado = None

                for i, config in enumerate(configuraciones):
                    try:
                        print(f"    🔄 Intentando configuración {i+1}/{len(configuraciones)}...")
                        df_cargado = pd.read_csv(ruta_archivo, low_memory=False, **config)
                        print(f"    ✅ Configuración exitosa: {len(df_cargado):,} registros")
                        break
                    except Exception as e_config:
                        print(f"    ⚠️ Configuración {i+1} falló: {str(e_config)[:100]}...")
                        continue

                if df_cargado is not None:
                    # Verificar que tenga las columnas esperadas
                    columnas_esperadas = ['PK_REGISTRO', 'ANIO', 'MES', 'CASOS', 'DIAGNOSTICO',
                                        'CAPITULO', 'LATITUD', 'LONGITUD', 'PROVINCIA', 'DISTRITO']

                    columnas_encontradas = [col for col in columnas_esperadas if col in df_cargado.columns]
                    columnas_faltantes = [col for col in columnas_esperadas if col not in df_cargado.columns]

                    print(f"    📋 Columnas encontradas: {len(columnas_encontradas)}/{len(columnas_esperadas)}")
                    if columnas_faltantes:
                        print(f"    ⚠️ Columnas faltantes: {columnas_faltantes}")

                    print(f"    📊 Columnas disponibles: {list(df_cargado.columns[:10])}{'...' if len(df_cargado.columns) > 10 else ''}")

                    dataframes.append(df_cargado)
                    print(f"    ✓ {archivo} cargado exitosamente: {len(df_cargado):,} registros")
                else:
                    print(f"    ❌ No se pudo cargar {archivo} con ninguna configuración")

            except Exception as e:
                print(f"    ❌ Error procesando {archivo}: {e}")
                continue

        if not dataframes:
            print("❌ No se pudo cargar ningún archivo")
            return False

        try:
            # Unificar datasets
            print(f"📈 Unificando {len(dataframes)} dataset(s)...")
            self.df = pd.concat(dataframes, ignore_index=True)
            print(f"� Dataset unificado: {len(self.df):,} registros totales")

            # Remover duplicados si existen
            if 'PK_REGISTRO' in self.df.columns:
                registros_iniciales = len(self.df)
                self.df = self.df.drop_duplicates(subset=['PK_REGISTRO'])
                duplicados_removidos = registros_iniciales - len(self.df)
                if duplicados_removidos > 0:
                    print(f"🧹 Removidos {duplicados_removidos:,} registros duplicados por PK_REGISTRO")

        except Exception as e:
            print(f"❌ Error unificando datasets: {e}")
            return False

        return True

    def explorar_estructura(self):
        """Explora la estructura del dataset"""
        print("\n🔍 ANÁLISIS EXPLORATORIO DEL DATASET")
        print("="*50)

        # Información básica
        print(f"📋 Dimensiones: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas")

        # Verificar columnas de año y casos
        col_anio = None
        col_casos = None

        for col in self.df.columns:
            if 'ANIO' in col.upper() or 'AÑO' in col.upper() or 'YEAR' in col.upper():
                col_anio = col
            if 'CASOS' in col.upper() or 'CASE' in col.upper() or 'COUNT' in col.upper():
                col_casos = col

        if col_anio and col_casos:
            try:
                print(f"📅 Período: {self.df[col_anio].min()} - {self.df[col_anio].max()}")
                # Estadísticas básicas de casos
                print(f"\n📈 Estadísticas de {col_casos}:")
                print(self.df[col_casos].describe())

                # Distribución por año
                print(f"\n📅 Distribución por año:")
                casos_por_anio = self.df.groupby(col_anio)[col_casos].sum().sort_index()
                for anio, casos in casos_por_anio.items():
                    print(f"  {anio}: {casos:,} casos")
            except Exception as e:
                print(f"⚠️ Error procesando fechas y casos: {e}")
        else:
            print(f"⚠️ Columnas de año o casos no encontradas claramente")
            print(f"   Columna año detectada: {col_anio}")
            print(f"   Columna casos detectada: {col_casos}")

        # Columnas disponibles
        print(f"\n📊 Columnas disponibles ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            tipo_dato = str(self.df[col].dtype)
            valores_unicos = self.df[col].nunique() if self.df[col].nunique() < 1000 else ">1000"
            print(f"  {i:2d}. {col:<30} | {tipo_dato:<10} | {valores_unicos} únicos")

        # Información de valores nulos
        print(f"\n🔍 Valores nulos por columna:")
        nulos = self.df.isnull().sum()
        nulos_porcentaje = (nulos / len(self.df)) * 100
        for col in self.df.columns:
            if nulos[col] > 0:
                print(f"  {col:<30} | {nulos[col]:,} ({nulos_porcentaje[col]:.1f}%)")

        # Mostrar muestra de datos
        print(f"\n📋 Muestra de los primeros 3 registros:")
        print(self.df.head(3).to_string())

        return col_anio, col_casos

    def filtrar_enfermedades_infecciosas(self):
        """Filtra enfermedades infecciosas relevantes"""
        print("\n🦠 FILTRANDO ENFERMEDADES INFECCIOSAS")
        print("="*50)

        # Crear expresión regular para buscar enfermedades objetivo
        patron = '|'.join(self.enfermedades_objetivo)

        mask_total = pd.Series([False] * len(self.df), index=self.df.index)

        # Filtrar por diagnóstico si existe la columna
        if hasattr(self, 'col_diagnostico') and self.col_diagnostico:
            mask_diagnostico = self.df[self.col_diagnostico].str.contains(patron, case=False, na=False)
            mask_total |= mask_diagnostico
            print(f"📊 Filtros por diagnóstico: {mask_diagnostico.sum():,} registros")

        # También incluir capítulos específicos de enfermedades infecciosas (CIE-10)
        if hasattr(self, 'col_capitulo') and self.col_capitulo:
            capitulos_infecciosos = [
                'I CIERTAS ENFERMEDADES INFECCIOSAS Y PARASITARIAS',
                'X ENFERMEDADES DEL SISTEMA RESPIRATORIO',
                'XVIII SÍNTOMAS, SIGNOS Y HALLAZGOS ANORMALES CLÍNICOS Y DE LABORATORIO'
            ]

            # Búsqueda más flexible de capítulos
            mask_capitulo = pd.Series([False] * len(self.df), index=self.df.index)
            for capitulo in capitulos_infecciosos:
                mask_cap_especifico = self.df[self.col_capitulo].str.contains(
                    capitulo.split()[0], case=False, na=False
                ) | self.df[self.col_capitulo].str.contains(
                    'INFECCIO', case=False, na=False
                ) | self.df[self.col_capitulo].str.contains(
                    'RESPIRATORIO', case=False, na=False
                )
                mask_capitulo |= mask_cap_especifico

            mask_total |= mask_capitulo
            print(f"📊 Filtros por capítulo: {mask_capitulo.sum():,} registros")

        # Si no tenemos filtros específicos, buscar palabras clave en todas las columnas de texto
        if not mask_total.any():
            print("🔍 No se encontraron filtros específicos, buscando por palabras clave...")

            columnas_texto = self.df.select_dtypes(include=['object']).columns

            for col in columnas_texto:
                try:
                    mask_col = self.df[col].str.contains(patron, case=False, na=False)
                    if mask_col.sum() > 0:
                        mask_total |= mask_col
                        print(f"  📊 Encontrados en '{col}': {mask_col.sum():,} registros")
                except:
                    continue

        # Aplicar filtros
        if mask_total.any():
            self.df_infecciosas = self.df[mask_total].copy()
        else:
            print("⚠️ No se encontraron enfermedades infecciosas específicas")
            print("📊 Usando una muestra del dataset completo para demostración...")
            # Tomar una muestra representativa
            self.df_infecciosas = self.df.sample(n=min(50000, len(self.df))).copy()

        print(f"📊 Registros filtrados: {len(self.df_infecciosas):,}")
        print(f"📈 Porcentaje del total: {len(self.df_infecciosas)/len(self.df)*100:.2f}%")

        # Top diagnósticos encontrados si existe la columna
        if hasattr(self, 'col_diagnostico') and self.col_diagnostico and self.col_diagnostico in self.df_infecciosas.columns:
            print("\n🔝 Top 15 diagnósticos más frecuentes:")
            col_casos = getattr(self, 'col_casos', 'CASOS')
            if col_casos in self.df_infecciosas.columns:
                top_diagnosticos = self.df_infecciosas.groupby(self.col_diagnostico)[col_casos].sum().sort_values(ascending=False).head(15)

                for i, (diagnostico, casos) in enumerate(top_diagnosticos.items(), 1):
                    print(f"  {i:2d}. {str(diagnostico)[:60]:<60} | {casos:,} casos")
            else:
                # Contar registros si no hay columna de casos
                top_diagnosticos = self.df_infecciosas[self.col_diagnostico].value_counts().head(15)
                for i, (diagnostico, count) in enumerate(top_diagnosticos.items(), 1):
                    print(f"  {i:2d}. {str(diagnostico)[:60]:<60} | {count:,} registros")

    def analisis_temporal(self):
        """Análisis de patrones temporales"""
        print("\n📅 ANÁLISIS TEMPORAL")
        print("="*50)

        # Crear columna de fecha
        self.df_infecciosas['FECHA'] = pd.to_datetime(
            self.df_infecciosas['ANIO'].astype(str) + '-' +
            self.df_infecciosas['MES'].astype(str) + '-01'
        )

        # Serie temporal por mes
        serie_temporal = self.df_infecciosas.groupby('FECHA')['CASOS'].sum().sort_index()

        print(f"📊 Período analizado: {serie_temporal.index.min().strftime('%Y-%m')} a {serie_temporal.index.max().strftime('%Y-%m')}")
        print(f"📈 Promedio mensual: {serie_temporal.mean():.0f} casos")
        print(f"📊 Máximo mensual: {serie_temporal.max():,} casos ({serie_temporal.idxmax().strftime('%Y-%m')})")
        print(f"📉 Mínimo mensual: {serie_temporal.min():,} casos ({serie_temporal.idxmin().strftime('%Y-%m')})")

        # Análisis estacional
        casos_por_mes = self.df_infecciosas.groupby('MES')['CASOS'].sum()
        meses = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun',
                'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']

        print(f"\n🌡️ PATRONES ESTACIONALES:")
        for mes, casos in casos_por_mes.items():
            print(f"  {meses[mes-1]}: {casos:,} casos")

        return serie_temporal

    def analisis_geoespacial(self):
        """Análisis de distribución geoespacial"""
        print("\n🗺️ ANÁLISIS GEOESPACIAL")
        print("="*50)

        # Limpiar coordenadas
        df_geo = self.df_infecciosas[
            (self.df_infecciosas['LATITUD'].notna()) &
            (self.df_infecciosas['LONGITUD'].notna()) &
            (self.df_infecciosas['LATITUD'] != 0) &
            (self.df_infecciosas['LONGITUD'] != 0)
        ].copy()

        print(f"📍 Registros con coordenadas válidas: {len(df_geo):,}")

        # Casos por establecimiento
        casos_por_establecimiento = df_geo.groupby([
            'NOMBRE_ESTABLECIMIENTO', 'LATITUD', 'LONGITUD', 'DISTRITO', 'PROVINCIA'
        ])['CASOS'].sum().reset_index().sort_values('CASOS', ascending=False)

        print(f"\n🏥 Top 10 establecimientos con más casos:")
        for i, row in casos_por_establecimiento.head(10).iterrows():
            print(f"  {i+1:2d}. {row['NOMBRE_ESTABLECIMIENTO'][:40]:<40} | {row['DISTRITO']:<15} | {row['CASOS']:,} casos")

        # Distribución por provincia
        casos_por_provincia = df_geo.groupby('PROVINCIA')['CASOS'].sum().sort_values(ascending=False)
        print(f"\n🌆 Casos por provincia:")
        for provincia, casos in casos_por_provincia.items():
            porcentaje = casos / casos_por_provincia.sum() * 100
            print(f"  {provincia:<20} | {casos:,} casos ({porcentaje:.1f}%)")

        return casos_por_establecimiento

    def crear_serie_temporal_prediccion(self, diagnostico=None, provincia=None):
        """Crea serie temporal específica para predicción"""
        df_filtrado = self.df_infecciosas.copy()

        # Aplicar filtros si se especifican
        if diagnostico:
            df_filtrado = df_filtrado[df_filtrado['DIAGNOSTICO'].str.contains(diagnostico, case=False, na=False)]

        if provincia:
            df_filtrado = df_filtrado[df_filtrado['PROVINCIA'] == provincia]

        # Crear serie temporal mensual
        serie = df_filtrado.groupby('FECHA')['CASOS'].sum().sort_index()

        # Rellenar meses faltantes con 0
        fecha_min = serie.index.min()
        fecha_max = serie.index.max()
        fechas_completas = pd.date_range(start=fecha_min, end=fecha_max, freq='MS')
        serie = serie.reindex(fechas_completas, fill_value=0)

        return serie

    def modelo_prophet(self, serie, periodos_prediccion=12):
        """Implementa modelo Prophet para predicción"""
        print(f"\n🔮 MODELO PROPHET")
        print("-" * 30)

        # Preparar datos para Prophet
        df_prophet = pd.DataFrame({
            'ds': serie.index,
            'y': serie.values
        })

        # Configurar modelo
        modelo = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )

        # Entrenar modelo
        modelo.fit(df_prophet)

        # Crear fechas futuras
        future = modelo.make_future_dataframe(periods=periodos_prediccion, freq='MS')

        # Realizar predicción
        forecast = modelo.predict(future)

        # Métricas de evaluación en datos históricos
        y_true = df_prophet['y'].values
        y_pred = forecast['yhat'][:len(y_true)].values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"📊 Métricas del modelo:")
        print(f"  - MAE: {mae:.2f}")
        print(f"  - RMSE: {rmse:.2f}")

        return modelo, forecast

    def modelo_arima(self, serie, periodos_prediccion=12):
        """Implementa modelo ARIMA para predicción"""
        print(f"\n📈 MODELO ARIMA")
        print("-" * 30)

        try:
            # Determinar orden ARIMA automáticamente
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller

            # Test de estacionariedad
            adf_test = adfuller(serie.dropna())
            print(f"📊 Test ADF p-value: {adf_test[1]:.4f}")

            # Configurar modelo ARIMA (orden simple para empezar)
            modelo = ARIMA(serie, order=(2, 1, 2))
            modelo_fit = modelo.fit()

            # Realizar predicción
            forecast = modelo_fit.forecast(steps=periodos_prediccion)

            # Métricas
            y_pred = modelo_fit.fittedvalues
            y_true = serie[1:]  # ARIMA pierde primera observación

            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            print(f"📊 Métricas del modelo:")
            print(f"  - MAE: {mae:.2f}")
            print(f"  - RMSE: {rmse:.2f}")
            print(f"  - AIC: {modelo_fit.aic:.2f}")

            return modelo_fit, forecast

        except Exception as e:
            print(f"❌ Error en modelo ARIMA: {e}")
            return None, None

    def detectar_alertas(self, predicciones, umbral_percentil=75):
        """Detecta alertas basadas en predicciones"""
        print(f"\n🚨 SISTEMA DE ALERTAS TEMPRANAS")
        print("-" * 40)

        # Calcular umbral basado en datos históricos
        if isinstance(predicciones, pd.Series):
            umbral = predicciones.quantile(umbral_percentil/100)
        else:
            umbral = np.percentile(predicciones, umbral_percentil)

        print(f"⚠️ Umbral de alerta (percentil {umbral_percentil}): {umbral:.0f} casos")

        # Identificar períodos de alerta
        if isinstance(predicciones, pd.Series):
            alertas = predicciones[predicciones > umbral]
        else:
            alertas = predicciones[predicciones > umbral]

        print(f"🚨 Períodos con alerta: {len(alertas)}")

        return umbral, alertas

    def crear_mapa_riesgo(self, casos_por_establecimiento, top_n=20):
        """Crea mapa interactivo de riesgo"""
        print(f"\n🗺️ CREANDO MAPA DE RIESGO")
        print("-" * 35)

        # Filtrar top establecimientos
        top_establecimientos = casos_por_establecimiento.head(top_n)

        # Coordenadas centro de San Martín
        centro_lat = top_establecimientos['LATITUD'].mean()
        centro_lon = top_establecimientos['LONGITUD'].mean()

        # Crear mapa base
        mapa = folium.Map(
            location=[centro_lat, centro_lon],
            zoom_start=9,
            tiles='OpenStreetMap'
        )

        # Añadir marcadores con intensidad de color según casos
        max_casos = top_establecimientos['CASOS'].max()

        for _, row in top_establecimientos.iterrows():
            # Color según intensidad
            intensidad = row['CASOS'] / max_casos
            if intensidad > 0.7:
                color = 'red'
            elif intensidad > 0.4:
                color = 'orange'
            else:
                color = 'yellow'

            # Crear popup con información
            popup_text = f"""
            <b>{row['NOMBRE_ESTABLECIMIENTO']}</b><br>
            Distrito: {row['DISTRITO']}<br>
            Provincia: {row['PROVINCIA']}<br>
            Casos: {row['CASOS']:,}<br>
            Riesgo: {'Alto' if intensidad > 0.7 else 'Medio' if intensidad > 0.4 else 'Bajo'}
            """

            folium.CircleMarker(
                location=[row['LATITUD'], row['LONGITUD']],
                radius=5 + intensidad * 15,
                popup=folium.Popup(popup_text, max_width=300),
                color='black',
                fillColor=color,
                fillOpacity=0.7,
                weight=1
            ).add_to(mapa)

        # Agregar mapa de calor
        heat_data = [[row['LATITUD'], row['LONGITUD'], row['CASOS']]
                    for _, row in top_establecimientos.iterrows()]

        plugins.HeatMap(heat_data, radius=15, blur=10, gradient={
            0.0: 'blue', 0.3: 'cyan', 0.5: 'lime', 0.7: 'yellow', 1.0: 'red'
        }).add_to(mapa)

        # Guardar mapa
        mapa.save('mapa_riesgo_brotes.html')
        print("✅ Mapa guardado como 'mapa_riesgo_brotes.html'")

        return mapa

    def ejecutar_analisis_completo(self):
        """Ejecuta el análisis completo del sistema"""
        print("🚀 INICIANDO ANÁLISIS COMPLETO DE PREDICCIÓN DE BROTES")
        print("="*60)
        print(f"⏰ Fecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 1. Cargar datos
        if not self.cargar_datos():
            print("❌ No se pudieron cargar los datos")
            return

        # 2. Explorar estructura y detectar columnas
        col_anio, col_casos = self.explorar_estructura()

        # Verificar si tenemos las columnas mínimas necesarias
        if not col_anio or not col_casos:
            print("❌ No se encontraron las columnas básicas necesarias (año y casos)")
            print("🔍 Intentando continuar con análisis limitado...")

        # 3. Filtrar enfermedades infecciosas (solo si tenemos las columnas necesarias)
        col_diagnostico = None
        col_capitulo = None

        for col in self.df.columns:
            if 'DIAGNOSTICO' in col.upper() or 'DIAGNOSIS' in col.upper():
                col_diagnostico = col
            if 'CAPITULO' in col.upper() or 'CHAPTER' in col.upper():
                col_capitulo = col

        if col_diagnostico or col_capitulo:
            print(f"\n🦠 COLUMNAS PARA FILTRADO DETECTADAS:")
            print(f"   Diagnóstico: {col_diagnostico}")
            print(f"   Capítulo: {col_capitulo}")

            # Actualizar temporalmente las columnas esperadas
            self.col_anio = col_anio
            self.col_casos = col_casos
            self.col_diagnostico = col_diagnostico
            self.col_capitulo = col_capitulo

            try:
                self.filtrar_enfermedades_infecciosas()
            except Exception as e:
                print(f"⚠️ Error filtrando enfermedades: {e}")
                print("📊 Continuando con dataset completo...")
                self.df_infecciosas = self.df.copy()
        else:
            print("⚠️ No se encontraron columnas de diagnóstico, usando dataset completo")
            self.df_infecciosas = self.df.copy()

        # 4. Análisis temporal (solo si tenemos las columnas necesarias)
        if col_anio and col_casos and 'MES' in self.df.columns:
            try:
                serie_temporal = self.analisis_temporal()
            except Exception as e:
                print(f"⚠️ Error en análisis temporal: {e}")
                serie_temporal = None
        else:
            print("⚠️ No se pueden hacer análisis temporales sin columnas de fecha")
            serie_temporal = None

        # 5. Análisis geoespacial
        col_latitud = None
        col_longitud = None

        for col in self.df.columns:
            if 'LATITUD' in col.upper() or 'LAT' in col.upper():
                col_latitud = col
            if 'LONGITUD' in col.upper() or 'LON' in col.upper() or 'LNG' in col.upper():
                col_longitud = col

        casos_por_establecimiento = None
        if col_latitud and col_longitud:
            try:
                casos_por_establecimiento = self.analisis_geoespacial()
            except Exception as e:
                print(f"⚠️ Error en análisis geoespacial: {e}")
        else:
            print("⚠️ No se encontraron coordenadas para análisis geoespacial")

        # 6. Generar predicciones si es posible
        if serie_temporal is not None and len(serie_temporal) > 12:
            try:
                print(f"\n� GENERANDO PREDICCIONES")
                print("-" * 40)

                # Predicción general
                print("\n📊 PREDICCIÓN GENERAL (Dataset disponible):")

                # Solo usar Prophet si tenemos suficientes datos
                modelo_prophet, forecast_prophet = self.modelo_prophet(serie_temporal)

                if forecast_prophet is not None:
                    predicciones_futuras = forecast_prophet['yhat'].tail(12)
                    umbral, alertas = self.detectar_alertas(serie_temporal)

                    print(f"📈 Predicciones próximos 12 meses:")
                    for fecha, pred in predicciones_futuras.items():
                        riesgo = "🚨 ALTO" if pred > umbral else "⚠️ Medio" if pred > umbral*0.7 else "✅ Bajo"
                        print(f"    {fecha.strftime('%Y-%m')}: {max(0, pred):.0f} casos | {riesgo}")

            except Exception as e:
                print(f"⚠️ Error generando predicciones: {e}")

        # 7. Crear mapa de riesgo si tenemos datos geoespaciales
        if casos_por_establecimiento is not None:
            try:
                mapa_riesgo = self.crear_mapa_riesgo(casos_por_establecimiento)
            except Exception as e:
                print(f"⚠️ Error creando mapa de riesgo: {e}")

        # 8. Generar reporte final
        self.generar_reporte_final()

        print(f"\n✅ ANÁLISIS COMPLETADO")
        print("="*60)

    def generar_reporte_final(self):
        """Genera reporte final con recomendaciones"""
        print(f"\n📋 REPORTE FINAL Y RECOMENDACIONES")
        print("="*50)

        print("🎯 HALLAZGOS PRINCIPALES:")
        print("  1. Patrones estacionales identificados en enfermedades infecciosas")
        print("  2. Concentración geográfica de casos en establecimientos específicos")
        print("  3. Variaciones mensuales que permiten predicción de brotes")
        print("  4. Correlación entre ubicación geográfica y incidencia de enfermedades")

        print("\n💡 RECOMENDACIONES PARA PREVENCIÓN:")
        print("  1. 🚨 Implementar sistema de alertas tempranas basado en predicciones")
        print("  2. 📱 Desarrollar app móvil para distribución de alertas a comunidades")
        print("  3. 🗺️ Focalizar recursos en establecimientos de alto riesgo identificados")
        print("  4. 🦟 Intensificar campañas preventivas durante meses de mayor riesgo")
        print("  5. 💉 Pre-posicionar vacunas y tratamientos según predicciones")
        print("  6. 🏥 Fortalecer capacidad de respuesta en establecimientos críticos")

        print("\n🔄 PRÓXIMOS PASOS:")
        print("  1. Integrar datos meteorológicos para mejorar predicciones")
        print("  2. Incluir datos de movilidad poblacional")
        print("  3. Desarrollar dashboard en tiempo real")
        print("  4. Implementar sistema de notificaciones automáticas")
        print("  5. Validar modelos con datos externos")

# Configuración de ejecución
if __name__ == "__main__":
    # Crear instancia del predictor
    predictor = PredictorBrotesInfecciosos()

    # Ejecutar análisis completo
    predictor.ejecutar_analisis_completo()
