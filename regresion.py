import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from scipy import stats

def main():
    st.title("Análisis de Regresión Lineal Múltiple para Rendimiento Académico")
    st.write("Este análisis tiene como objetivo identificar los factores que influyen en el rendimiento académico de los estudiantes.")

    uploaded_file = st.file_uploader("Carga el archivo CSV, XLS o XLSX con los datos de los estudiantes", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Formato de archivo no soportado. Por favor, sube un archivo CSV, XLS o XLSX.")
                return
        except Exception as e:
            st.error(f"Error al cargar el archivo: {str(e)}")
            return

        st.subheader("Vista previa de los datos")
        st.write(df.head())

        st.subheader("Exploración de Datos")
        st.write("Dimensiones del dataset: ", df.shape)
        st.write("Resumen estadístico: ")
        st.write(df.describe())
        st.write("Visualización de la distribución de las variables")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, ax=ax)
        st.pyplot(fig)

        st.write("Valores faltantes en los datos: ")
        st.write(df.isna().sum())
        df = df.dropna()

        st.subheader("Evaluación de Multicolinealidad")
        numeric_columns = df.select_dtypes(include=['number'])
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        st.write("Nombres de las columnas en el dataset: ")
        st.write(df.columns)
        X = df[['Horas estudiadas', 'puntuaciones anteriores', 'actividades extracurriculares', 'horas de sueño', 'ejemplos de exámenes realizados']]
        y = df['índice de rendimiento']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        st.subheader("Ajuste del Modelo")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        st.subheader("Evaluación del Modelo")
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        st.write(f"R² del conjunto de entrenamiento: {r2_train:.4f}")
        st.write(f"R² del conjunto de prueba: {r2_test:.4f}")
        st.write(f"MSE del conjunto de entrenamiento: {mse_train:.4f}")
        st.write(f"MSE del conjunto de prueba: {mse_test:.4f}")

        st.subheader("Análisis de Varianza (ANOVA)")
        X_train_sm = sm.add_constant(X_train)
        model_sm = sm.OLS(y_train, X_train_sm).fit()
        anova_table = sm.stats.anova_lm(model_sm, typ=2)
        st.write(anova_table)

        st.subheader("Coeficientes del Modelo")
        coef_df = pd.DataFrame({'Variable': X.columns, 'Coeficiente': model.coef_})
        st.write(coef_df)
        for idx, coef in enumerate(model.coef_):
            st.write(f"Interpretación: Por cada unidad adicional de {X.columns[idx]}, el GPA cambia en promedio {coef:.4f} unidades, manteniendo las demás variables constantes.")

        st.subheader("Diagnósticos de Regresión")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred_train, y_train - y_pred_train)
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel("Valores Ajustados")
        ax.set_ylabel("Residuos")
        ax.set_title("Gráfico de Residuos vs Valores Ajustados")
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(y_train - y_pred_train, dist="norm", plot=ax)
        st.pyplot(fig)
        st.write("Interpretación: Cuando los puntos se agrupan cerca de la línea diagonal, indica que los residuos siguen una distribución normal.")

        st.subheader("Predicciones para Nuevos Estudiantes")
        input_data = []
        for col in X.columns:
            value = st.number_input(f"Ingrese un valor para {col}", value=float(X[col].mean()))
            input_data.append(value)
        prediction = model.predict([input_data])
        st.write(f"Predicción del Índice de Rendimiento: {prediction[0]:.4f}")

        st.subheader("Descripción del Modelo")
        st.write("Este modelo de regresión lineal múltiple evalúa cómo distintas variables independientes influyen en el desempeño académico.")
        st.write("Consideraciones: El modelo se basa en las variables disponibles y puede no reflejar todas las dinámicas posibles. Es recomendable explorar modelos más avanzados para mejorar la precisión de las predicciones.")

if __name__ == '__main__':
    main()