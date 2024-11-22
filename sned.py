import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Título de la aplicación
st.title("Prueba de Homogeneidad de Varianzas - F de Fisher (Cola Derecha e Izquierda)")

# Descripción del análisis
st.write("""
Esta aplicación realiza una prueba de homogeneidad de varianzas entre dos o más muestras utilizando la prueba F de Fisher. 
La fórmula utilizada es la siguiente:
""")
st.latex(r'F = \frac{s_1^2}{s_2^2}')
st.write("""
Donde \( s_1^2 \) y \( s_2^2 \) son las **varianzas muestrales** de las dos muestras que se comparan.
Se calculan los valores críticos tanto para la **cola derecha** (si la varianza de la primera muestra es significativamente mayor) 
como para la **cola izquierda** (si la varianza de la primera muestra es significativamente menor).
""")

# Subir archivo Excel
uploaded_file = st.file_uploader("Sube tu archivo Excel con las muestras", type=["xlsx"])

if uploaded_file is not None:
    # Leer el archivo Excel
    df = pd.read_excel(uploaded_file)
    
    # Mostrar los primeros registros del archivo cargado
    st.write("### Vista previa de los datos:")
    st.dataframe(df.head())
    
    # Selección de múltiples columnas
    column_options = df.columns.tolist()
    selected_columns = st.multiselect("Selecciona las columnas para analizar", column_options)
    
    if len(selected_columns) > 1:
        # Realizar análisis por pares de columnas seleccionadas
        st.write("### Resultados de la prueba F para las columnas seleccionadas (Cola Derecha e Izquierda):")
        
        # Iterar sobre los pares de columnas seleccionadas
        for i in range(len(selected_columns)):
            for j in range(i + 1, len(selected_columns)):
                col1 = selected_columns[i]
                col2 = selected_columns[j]

                # Extraer las columnas seleccionadas
                muestra_1 = df[col1].dropna().values
                muestra_2 = df[col2].dropna().values

                # Calcular las varianzas de las muestras
                var1 = np.var(muestra_1, ddof=1)  # Varianza muestral de la primera muestra
                var2 = np.var(muestra_2, ddof=1)  # Varianza muestral de la segunda muestra

                # Calcular el estadístico F como la relación de las varianzas
                F = var1 / var2

                # Grados de libertad
                df1 = len(muestra_1) - 1
                df2 = len(muestra_2) - 1

                # Nivel de significancia
                alpha = 0.05

                # Valores críticos para una prueba de dos colas
                F_crit_derecha = stats.f.ppf(1 - alpha, df1, df2)  # Valor crítico para la cola derecha
                F_crit_izquierda = stats.f.ppf(alpha, df1, df2)    # Valor crítico para la cola izquierda

                # Mostrar los resultados para el par de columnas
                st.write(f"**Comparación entre {col1} y {col2}:**")
                st.write(f"Varianza de {col1} (\(s_1^2\)): {var1:.9f}")
                st.write(f"Varianza de {col2} (\(s_2^2\)): {var2:.9f}")
                st.write(f"Estadístico F: {F:.9f}")
                st.write(f"Grados de libertad {col1} (\(df_1\)): {df1}")
                st.write(f"Grados de libertad {col2} (\(df_2\)): {df2}")
                st.write(f"Valor crítico de F (cola derecha, α=0.05): {F_crit_derecha:.5f}")
                st.write(f"Valor crítico de F (cola izquierda, α=0.05): {F_crit_izquierda:.5f}")

                # Interpretación
                if F > F_crit_derecha:
                    st.write(f"**Resultado (cola derecha):** Rechazamos la hipótesis nula. La varianza de **{col1}** es significativamente mayor que la de **{col2}**.")
                elif F < F_crit_izquierda:
                    st.write(f"**Resultado (cola izquierda):** Rechazamos la hipótesis nula. La varianza de **{col1}** es significativamente menor que la de **{col2}**.")
                else:
                    st.write("**Resultado:** No se rechaza la hipótesis nula. Las varianzas son homogéneas.")

                # Crear gráfico de la distribución F
                x_max = max(F_crit_derecha * 1.5, F_crit_izquierda * 0.5, F * 1.5)
                x = np.linspace(0, x_max, 500)
                f_dist = stats.f.pdf(x, df1, df2)

                plt.figure(figsize=(8, 6))
                plt.plot(x, f_dist, label=f'Distribución F con df1={df1}, df2={df2}')
                plt.axvline(F_crit_derecha, color='red', linestyle='--', label=f'Valor Crítico F (cola derecha): {F_crit_derecha:.2f}')
                plt.axvline(F_crit_izquierda, color='green', linestyle='--', label=f'Valor Crítico F (cola izquierda): {F_crit_izquierda:.2f}')
                plt.axvline(F, color='blue', linestyle='--', label=f'Estadístico F: {F:.2f}')
                plt.fill_between(x, 0, f_dist, where=(x > F_crit_derecha), color='red', alpha=0.3)
                plt.fill_between(x, 0, f_dist, where=(x < F_crit_izquierda), color='green', alpha=0.3)
                plt.title(f'Distribución F y el Estadístico F para {col1} y {col2}')
                plt.xlabel('Valor de F')
                plt.ylabel('Densidad de Probabilidad')
                plt.legend()

                # Mostrar el gráfico en Streamlit
                st.pyplot(plt)
                st.write("---")

    else:
        st.write("Por favor, sube un archivo Excel para continuar.")
