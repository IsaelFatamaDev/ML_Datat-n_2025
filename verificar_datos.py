"""
Script de verificación para comprobar la carga de datos CSV
"""
import pandas as pd
import os

def verificar_carga_datos():
    """Verifica que los datos CSV se estén cargando correctamente"""

    data_path = 'data/'
    print("🔍 VERIFICANDO CARGA DE DATOS CSV")
    print("="*50)

    # Verificar que existe la carpeta
    if not os.path.exists(data_path):
        print(f"❌ La carpeta '{data_path}' no existe")
        return

    # Listar archivos en la carpeta
    archivos = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    print(f"📁 Archivos CSV encontrados: {archivos}")

    for archivo in archivos:
        ruta_completa = os.path.join(data_path, archivo)
        tamaño_mb = os.path.getsize(ruta_completa) / (1024*1024)

        print(f"\n📊 ANALIZANDO: {archivo}")
        print(f"   📦 Tamaño: {tamaño_mb:.1f} MB")

        try:
            # Intentar leer solo las primeras líneas
            print("   🔍 Leyendo primeras 5 líneas...")

            with open(ruta_completa, 'r', encoding='utf-8', errors='ignore') as f:
                for i in range(5):
                    linea = f.readline().strip()
                    print(f"   Línea {i+1}: {linea[:100]}{'...' if len(linea) > 100 else ''}")

            # Intentar detectar separador
            with open(ruta_completa, 'r', encoding='utf-8', errors='ignore') as f:
                primera_linea = f.readline().strip()

            separadores = [',', ';', '|', '\t']
            separador_detectado = ','
            max_campos = 0

            for sep in separadores:
                campos = len(primera_linea.split(sep))
                print(f"   Separador '{sep}': {campos} campos")
                if campos > max_campos:
                    max_campos = campos
                    separador_detectado = sep

            print(f"   ✅ Separador detectado: '{separador_detectado}' con {max_campos} campos")

            # Intentar cargar una muestra pequeña
            print("   📈 Intentando cargar muestra...")
            try:
                df_muestra = pd.read_csv(ruta_completa,
                                       sep=separador_detectado,
                                       nrows=100,
                                       encoding='utf-8',
                                       on_bad_lines='skip')

                print(f"   ✅ Muestra cargada: {len(df_muestra)} filas × {len(df_muestra.columns)} columnas")
                print(f"   📋 Columnas: {list(df_muestra.columns)}")

                # Verificar columnas importantes
                columnas_importantes = ['ANIO', 'MES', 'CASOS', 'DIAGNOSTICO', 'LATITUD', 'LONGITUD', 'PROVINCIA']
                columnas_encontradas = [col for col in columnas_importantes if col in df_muestra.columns]
                columnas_faltantes = [col for col in columnas_importantes if col not in df_muestra.columns]

                print(f"   ✅ Columnas importantes encontradas: {columnas_encontradas}")
                if columnas_faltantes:
                    print(f"   ⚠️ Columnas faltantes: {columnas_faltantes}")

                # Mostrar tipos de datos
                print(f"   📊 Tipos de datos:")
                for col in df_muestra.columns[:10]:  # Solo primeras 10 columnas
                    print(f"      {col}: {df_muestra[col].dtype}")

            except Exception as e:
                print(f"   ❌ Error cargando muestra: {e}")

        except Exception as e:
            print(f"   ❌ Error general: {e}")

if __name__ == "__main__":
    verificar_carga_datos()
