import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu

# Configuración de la página
st.set_page_config(page_title="Análisis Visual y Regresión", layout="wide")

# Menú de navegación con íconos personalizados
with st.sidebar:
    selected = option_menu(
        menu_title="Menú Principal",
        options=["📂 Carga de Datos", "📊 Visualización", "📈 Regresión Logística"],
        icons=["folder-open", "bar-chart-line", "calculator"],  # Bootstrap icons
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#f4f4f4"},
            "icon": {"color": "#007BFF", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#007BFF", "color": "white"},
        },
    )

# Función para cargar datos
def cargar_datos():
    archivo = st.file_uploader("📤 Sube un archivo (CSV o Excel)", type=["csv", "xlsx"])
    if archivo is not None:
        if archivo.name.endswith(".csv"):
            df = pd.read_csv(archivo)
        elif archivo.name.endswith(".xlsx"):
            df = pd.read_excel(archivo)
        st.write("👀 **Vista previa de los datos:**")
        st.dataframe(df.head())
        # Guardar los datos en el estado de sesión
        st.session_state["data"] = df
        return df
    return None

# Carga de datos
if selected == "📂 Carga de Datos":
    st.header("📂 **Carga de Datos**")
    if "data" not in st.session_state:
        data = cargar_datos()
    else:
        st.write("📋 **Datos previamente cargados:**")
        st.dataframe(st.session_state["data"])

# Visualización
if selected == "📊 Visualización":
    st.header("📊 **Visualización de Datos**")
    if "data" in st.session_state:
        data = st.session_state["data"]

        # Menú de selección para tipos de gráficos
        tipo_grafico = st.selectbox(
            "🔍 **Selecciona el tipo de gráfico que quieres visualizar:**",
            [
                "📉 Gráfico de Dispersión",
                "📅 Series Temporales",
                "📊 Histograma",
                "🔥 Mapa de Calor",
                "🎯 Gráfico Vectorial",
            ],
        )

        if tipo_grafico == "📉 Gráfico de Dispersión":
            st.subheader("📉 **Gráfico de Dispersión**")
            col_x = st.selectbox("📌 Selecciona la variable independiente (X):", data.columns)
            col_y = st.selectbox("📌 Selecciona la variable dependiente (Y):", data.columns)
            if col_x and col_y:
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=data, x=col_x, y=col_y)
                plt.title(f"Relación entre {col_x} y {col_y}")
                st.pyplot(plt)
                
                # Agregar interpretación
                corr = data[[col_x, col_y]].corr().iloc[0, 1]
                st.write(f"📈 **Correlación entre {col_x} y {col_y}: {corr:.2f}**")
                if abs(corr) > 0.7:
                    st.write("💡 La correlación es fuerte, lo que sugiere que ambas variables están muy relacionadas.")
                elif abs(corr) > 0.4:
                    st.write("💡 La correlación es moderada, lo que sugiere cierta relación entre las variables.")
                else:
                    st.write("💡 La correlación es débil, lo que sugiere una relación poco significativa.")

        elif tipo_grafico == "📅 Series Temporales":
            st.subheader("📅 **Series Temporales**")
            col_time = st.selectbox("🕒 Selecciona la columna de tiempo:", data.columns)
            col_value = st.selectbox("📈 Selecciona la columna de valores:", data.columns)
            if col_time and col_value:
                plt.figure(figsize=(10, 6))
                plt.plot(data[col_time], data[col_value], marker="o")
                plt.title(f"Serie Temporal: {col_value} vs {col_time}")
                plt.xlabel(col_time)
                plt.ylabel(col_value)
                st.pyplot(plt)

        elif tipo_grafico == "📊 Histograma":
            st.subheader("📊 **Histograma**")
            col_hist = st.selectbox("📌 Selecciona la columna para el histograma:", data.columns)
            if col_hist:
                plt.figure(figsize=(8, 6))
                sns.histplot(data[col_hist], kde=True, bins=20, color="blue")
                plt.title(f"Distribución de {col_hist}")
                st.pyplot(plt)

                # Agregar estadísticas descriptivas
                st.write(f"📊 **Estadísticas de {col_hist}:**")
                st.write(data[col_hist].describe())
                st.write("💡 El histograma muestra la distribución de los datos. Observa si la distribución es normal, sesgada o tiene alguna forma específica.")

        elif tipo_grafico == "🔥 Mapa de Calor":
            st.subheader("🔥 **Mapa de Calor**")
            numeric_cols = data.select_dtypes(include=[np.number])
            if numeric_cols.shape[1] > 1:
                corr = numeric_cols.corr()
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
                plt.title("Mapa de Calor de Correlación")
                st.pyplot(plt)
                
                # Agregar interpretación
                st.write("💡 El mapa de calor muestra las correlaciones entre variables numéricas. Las correlaciones cercanas a +1 o -1 indican una relación fuerte.")
            else:
                st.warning("⚠️ Necesitas al menos dos columnas numéricas para generar un mapa de calor.")

        elif tipo_grafico == "🎯 Gráfico Vectorial":
            st.subheader("🎯 **Gráfico Vectorial**")
            col_x = st.selectbox("📌 Selecciona la componente X:", data.columns)
            col_y = st.selectbox("📌 Selecciona la componente Y:", data.columns)
            if col_x and col_y:
                X = data[col_x]
                Y = data[col_y]
                plt.figure(figsize=(8, 6))
                plt.quiver(
                    [0] * len(X), [0] * len(Y), X, Y, angles="xy", scale_units="xy", scale=1
                )
                plt.xlim(-1.5 * max(abs(X)), 1.5 * max(abs(X)))
                plt.ylim(-1.5 * max(abs(Y)), 1.5 * max(abs(Y)))
                plt.title("Gráfico Vectorial")
                plt.xlabel("X")
                plt.ylabel("Y")
                st.pyplot(plt)
# Regresión Logística
# Regresión Logística
if selected == "📈 Regresión Logística":
    st.header("📈 **Regresión Logística**")

    if "data" in st.session_state:
        data = st.session_state["data"]

        # Verifica que los datos sean suficientes para entrenar un modelo
        if data.shape[0] > 1 and data.shape[1] > 1:
            # Menú de selección para las columnas
            st.subheader("📊 **Selecciona las columnas para el modelo**")

            # Selección de la columna objetivo (y) y las características (X)
            objetivo = st.selectbox("📌 Selecciona la columna objetivo (y):", data.columns)
            características = st.multiselect("📌 Selecciona las columnas de características (X):", data.columns)

            # Asegurarse de que se han seleccionado al menos una columna de características y una de objetivo
            if objetivo and len(características) > 0:
                # Preprocesamiento de datos
                X = data[características]
                y = data[objetivo]

                # Convertir variables categóricas a numéricas si es necesario
                X = pd.get_dummies(X, drop_first=True)  # Convierte variables categóricas a numéricas

                # División de los datos en entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

                # Entrenamiento del modelo de regresión logística
                modelo = LogisticRegression(max_iter=1000)
                modelo.fit(X_train, y_train)

                # Predicciones
                y_pred = modelo.predict(X_test)

                # Mostrar las métricas de rendimiento
                st.subheader("📊 **Métricas del modelo**")
                st.write(f"**Precisión:** {accuracy_score(y_test, y_pred):.2f}")
                st.write("**Matriz de Confusión:**")
                cm = confusion_matrix(y_test, y_pred)
                st.write(cm)

                # Mostrar reporte de clasificación
                st.write("**Reporte de Clasificación:**")
                st.text(classification_report(y_test, y_pred))

                # Graficar la matriz de confusión
                st.subheader("📊 **Matriz de Confusión Visualizada**")
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=modelo.classes_, yticklabels=modelo.classes_)
                plt.title("Matriz de Confusión")
                plt.xlabel('Predicción')
                plt.ylabel('Real')
                st.pyplot(plt)

            else:
                st.warning("⚠️ Debes seleccionar al menos una columna para las características y una para el objetivo.")
        else:
            st.warning("⚠️ Los datos no son suficientes para entrenar un modelo.")
    else:
        st.warning("⚠️ No se han cargado datos. Por favor, carga los datos en la sección 'Carga de Datos'.")

