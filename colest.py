import streamlit as st
import pandas as pd
from scipy import stats

# Título de la aplicación
st.title("Cálculo del valor p en una prueba F (Distribución F de Snedecor)")

# Subida del archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Lectura del archivo CSV
    data = pd.read_csv(uploaded_file)
    
    # Mostrar el DataFrame cargado
    st.write("Datos cargados:")
    st.dataframe(data)
    
    # Seleccionar las columnas a comparar
    column1 = st.selectbox("Selecciona la primera columna", data.columns)
    column2 = st.selectbox("Selecciona la segunda columna", data.columns)
    
    if column1 != column2:
        # Calcular la prueba F
        group1 = data[column1].dropna()
        group2 = data[column2].dropna()

        f_stat, p_value = stats.f_oneway(group1, group2)

        # Mostrar resultados
        st.write(f"Estadístico F: {f_stat}")
        st.write(f"Valor p: {p_value}")

        # Interpretación del valor p
        if p_value < 0.05:
            st.write("Hay diferencias significativas entre las dos muestras (rechazamos H0).")
        else:
            st.write("No hay diferencias significativas entre las dos muestras (no se rechaza H0).")
    else:
        st.warning("Por favor, selecciona dos columnas diferentes para realizar la prueba F.")
