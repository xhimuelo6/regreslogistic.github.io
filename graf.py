import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split  # Importa esta función
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Visualizador de Archivos", layout="wide", page_icon="📊", menu_items={"About": "Creado por [Carlos Antonio Jimenez Apaza]"})

# Estilo de fondo
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #212121;
        color: white;
    }
    .sidebar .sidebar-content a {
        color: white;
    }
    .stButton > button {
        background-color: #1e90ff;
        color: white;
        border-radius: 8px;
        border: 1px solid #1e90ff;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #ff6347;
        border-color: #ff6347;
    }
    .stSelectbox, .stSlider {
        background-color: #fff;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Títulos de la página y descripción
st.title("📊 **Visualizador de Archivos - Análisis y Gráficos Interactivos**")
st.markdown("""
    **Bienvenido** a esta herramienta de análisis interactivo. Aquí podrás cargar tus archivos y generar gráficos para un análisis visual detallado. 
    Selecciona una opción del menú y empieza a explorar los datos.
""")

# Menú lateral con iconos y estilo
st.sidebar.title("Menú de Opciones")
menu = st.sidebar.radio(
    "Selecciona una opción:",
    [
        "📂 Cargar archivo",
        "📊 Histograma",
        "📈 Gráfico de Línea",
        "📉 Gráfico de Barras",
        "📌 Gráfico de Dispersión",
        "🎂 Gráfico de Torta",
        "🔥 Mapa de Calor",
        "🔥 Mapa de Calor de la Frontera de Decisión",
    ],
    help="Elige el tipo de gráfico o acción que deseas realizar con los datos cargados."
)

# Función para cargar archivos
def cargar_archivo():
    archivo = st.file_uploader("Sube tu archivo (CSV, Excel, JSON)", type=["csv", "xlsx", "xls", "json"], label_visibility="collapsed")
    if archivo is not None:
        if archivo.name.endswith('.csv'):
            return pd.read_csv(archivo)
        elif archivo.name.endswith('.xlsx') or archivo.name.endswith('.xls'):
            return pd.read_excel(archivo)
        elif archivo.name.endswith('.json'):
            return pd.read_json(archivo)
    else:
        st.warning("Por favor, sube un archivo para continuar.")
        return None

# Mostrar datos y gráficos según el menú
if menu == "📂 Cargar archivo":
    st.subheader("Cargar y ver tus datos")
    df = cargar_archivo()
    if df is not None:
        st.dataframe(df)

elif menu == "📊 Histograma":
    st.subheader("Generar Histograma")
    df = cargar_archivo()
    if df is not None:
        columna = st.selectbox("Selecciona una columna numérica", df.select_dtypes(include='number').columns)
        bins = st.slider("Número de intervalos", min_value=5, max_value=50, value=10)
        plt.figure(figsize=(10, 6))
        plt.hist(df[columna], bins=bins, color='#FF6347', edgecolor='black')
        plt.title(f"Histograma de {columna}", fontsize=14, fontweight='bold')
        plt.xlabel(columna, fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        st.pyplot(plt)

elif menu == "📈 Gráfico de Línea":
    st.subheader("Generar Gráfico de Línea")
    df = cargar_archivo()
    if df is not None:
        columna_x = st.selectbox("Selecciona la columna para el eje X", df.columns)
        columna_y = st.selectbox("Selecciona la columna para el eje Y", df.select_dtypes(include='number').columns)
        plt.figure(figsize=(10, 6))
        plt.plot(df[columna_x], df[columna_y], marker='o', color='#1E90FF', linestyle='-', linewidth=2, markersize=5)
        plt.title(f"Gráfico de Línea: {columna_y} vs {columna_x}", fontsize=14, fontweight='bold')
        plt.xlabel(columna_x, fontsize=12)
        plt.ylabel(columna_y, fontsize=12)
        st.pyplot(plt)

elif menu == "📉 Gráfico de Barras":
    st.subheader("Generar Gráfico de Barras")
    df = cargar_archivo()
    if df is not None:
        columna = st.selectbox("Selecciona una columna categórica", df.select_dtypes(include='object').columns)
        conteos = df[columna].value_counts()
        
        # Crear gráfico de barras
        plt.figure(figsize=(10, 6))
        plt.bar(conteos.index, conteos.values, color='#32CD32', edgecolor='black')
        plt.title(f"Gráfico de Barras: Frecuencia de {columna}", fontsize=14, fontweight='bold')
        plt.xlabel(columna, fontsize=12)
        plt.ylabel("Frecuencia", fontsize=12)
        st.pyplot(plt)
        
        # Estadísticas de las categorías
        with st.expander("Ver Estadísticas de las categorías"):
            st.write("Conteo de cada categoría:")
            st.write(conteos)
            st.write(f"Media de frecuencias: {conteos.mean():.2f}")
            st.write(f"Varianza de frecuencias: {conteos.var():.2f}")
            st.write(f"Desviación estándar: {conteos.std():.2f}")
            st.write(f"Categoría más frecuente: {conteos.idxmax()} con {conteos.max()} ocurrencias")
            st.write(f"Categoría menos frecuente: {conteos.idxmin()} con {conteos.min()} ocurrencias")

elif menu == "📌 Gráfico de Dispersión":
    st.subheader("Generar Gráfico de Dispersión")
    df = cargar_archivo()
    if df is not None:
        columna_x = st.selectbox("Selecciona la columna para el eje X", df.select_dtypes(include='number').columns)
        columna_y = st.selectbox("Selecciona la columna para el eje Y", df.select_dtypes(include='number').columns)
        plt.figure(figsize=(10, 6))
        plt.scatter(df[columna_x], df[columna_y], alpha=0.7, color='#FFD700', edgecolor='black')
        plt.title(f"Gráfico de Dispersión: {columna_y} vs {columna_x}", fontsize=14, fontweight='bold')
        plt.xlabel(columna_x, fontsize=12)
        plt.ylabel(columna_y, fontsize=12)
        st.pyplot(plt)

elif menu == "🎂 Gráfico de Torta":
    st.subheader("Generar Gráfico de Torta")
    df = cargar_archivo()
    if df is not None:
        columna = st.selectbox("Selecciona una columna categórica", df.select_dtypes(include='object').columns)
        conteos = df[columna].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(conteos.values, labels=conteos.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2", len(conteos)))
        plt.title(f"Distribución de {columna}", fontsize=14, fontweight='bold')
        st.pyplot(plt)

elif menu == "🔥 Mapa de Calor":
    st.subheader("Generar Mapa de Calor")
    df = cargar_archivo()
    if df is not None:
        # Seleccionamos las columnas numéricas para el mapa de calor
        columnas_numericas = df.select_dtypes(include='number')
        
        if not columnas_numericas.empty:
            # Calcular la matriz de correlación
            corr = columnas_numericas.corr()
            
            # Crear el mapa de calor
            plt.figure(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1, vmax=1)
            plt.title("Mapa de Calor: Matriz de Correlación", fontsize=14, fontweight='bold')
            st.pyplot(plt)
        else:
            st.warning("No hay columnas numéricas en el archivo para generar el mapa de calor.")

if menu == "🔥 Mapa de Calor de la Frontera de Decisión":
    st.subheader("Visualizar la Frontera de Decisión")
    df = cargar_archivo()
    
    if df is not None:
        # Supongamos que tenemos un problema binario y seleccionamos solo dos características
        # (en caso contrario, puedes hacer PCA o seleccionar dos columnas)
        if df.shape[1] < 3:
            st.warning("El conjunto de datos debe tener al menos tres columnas: dos características y una columna de clase.")
        else:
            X = df.iloc[:, :2].values  # Usamos solo las dos primeras características
            y = df.iloc[:, 2].values   # La tercera columna es la etiqueta de clase

            # Dividir el conjunto de datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Crear un clasificador SVM (Support Vector Machine)
            clf = SVC(kernel='linear', random_state=42)
            clf.fit(X_train, y_train)

            # Crear una malla de puntos para la visualización de la frontera de decisión
            h = 0.02  # Resolución del gráfico
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))

            # Predecir las etiquetas para cada punto en la malla
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            # Crear el gráfico de la frontera de decisión
            plt.figure(figsize=(10, 6))
            plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)  # Mapa de calor de la frontera de decisión
            plt.colorbar()

            # Graficar los puntos de entrenamiento
            plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=100, cmap=plt.cm.coolwarm, label="Entrenamiento")
            plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='^', s=100, cmap=plt.cm.coolwarm, label="Prueba")

            # Títulos y etiquetas
            plt.title("Frontera de Decisión del Clasificador SVM", fontsize=14, fontweight='bold')
            plt.xlabel('Característica 1', fontsize=12)
            plt.ylabel('Característica 2', fontsize=12)
            plt.legend()
            st.pyplot(plt)