import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import io

# Función para cargar los datos desde un archivo
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            st.error("Formato de archivo no soportado. Por favor, sube un archivo CSV o Excel.")
            return None
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# Función para verificar los supuestos de la regresión
def check_assumptions(X, y, y_pred, residuals):
    st.subheader('Prueba de Normalidad de Residuos')
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Gráfico Q-Q de Normalidad de Residuos")
    st.pyplot(fig)

    st.subheader('Ajuste del Modelo de Regresión Ridge')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([min(y), max(y)], [min(y_pred), max(y_pred)], color='red', linewidth=2)
    ax.set_xlabel('Valores Observados (Y)')
    ax.set_ylabel('Valores Predichos (Ŷ)')
    ax.set_title('Gráfico de Linealidad: Observados vs Predichos')
    st.pyplot(fig)

    st.subheader('Evaluación del Modelo: Homocedasticidad de Residuos')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.set_xlabel('Valores Predichos (Ŷ)')
    ax.set_ylabel('Residuos')
    ax.set_title('Gráfico de Homocedasticidad de Residuos')
    ax.axhline(y=0, color='r', linestyle='--')
    st.pyplot(fig)

# Función para calcular la regresión Ridge
def calculate_ridge(X, y, alpha=1.0):
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return model, y_pred, residuals, r2, mse

# Función para mostrar el procedimiento de cálculo con fórmulas LaTeX
def show_procedure(X, y, model, r2, mse):
    st.subheader("Procedimiento de cálculo detallado")
    
    st.write("1. Ecuación del modelo de regresión Ridge:")
    coef_str = " + ".join([f"{coef:.4f} X{i+1}" for i, coef in enumerate(model.coef_)])
    st.latex(f"Y = {model.intercept_:.4f} + {coef_str}")
    
    st.write("2. Cálculo del coeficiente de determinación (R²):")
    st.latex(f"R^2 = 1 - \\frac{{\\sum(Y - \\hat{{Y}})^2}}{{\\sum(Y - \\bar{{Y}})^2}} = {r2:.4f}")
    
    st.write("3. Cálculo del Error Cuadrático Medio (MSE):")
    st.latex(f"MSE = \\frac{{\\sum(Y - \\hat{{Y}})^2}}{{n}} = {mse:.4f}")

# Función principal de la aplicación Streamlit
def main():
    st.title('Aplicación de Regresión Lineal Ridge')
    st.write("Esta aplicación realiza un análisis de regresión lineal múltiple utilizando Ridge, incluyendo visualizaciones e interpretaciones detalladas.")
    
    # Carga de datos
    uploaded_file = st.file_uploader("Elige un archivo CSV o Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.subheader("Vista previa de los datos")
            st.write(df.head())
            
            # Selección de variables
            columns = df.columns.tolist()
            x_columns = st.multiselect('Selecciona las variables independientes (X)', columns)
            y_column = st.selectbox('Selecciona la variable dependiente (Y)', columns)
            
            if len(x_columns) > 0 and y_column:
                X = df[x_columns].values
                y = df[y_column].values
                
                # Parámetro alpha de Ridge
                alpha = st.slider('Selecciona el valor de alpha para Ridge', min_value=0.01, max_value=10.0, value=1.0, step=0.01)
            
                # Cálculo de la regresión Ridge
                model, y_pred, residuals, r2, mse = calculate_ridge(X, y, alpha)
                
                # Resultados de la regresión
                st.subheader('Resultados de la Regresión Ridge')
                st.latex(f"Intercepto (b_0): {model.intercept_:.4f}")
                coef_str = ", ".join([f"b_{i+1} = {coef:.4f}" for i, coef in enumerate(model.coef_)])
                st.latex(f"Coeficientes: {coef_str}")
                st.latex(f"R^2: {r2:.4f}")
                st.latex(f"MSE: {mse:.4f}")
                
                # Gráficos de dispersión
                st.subheader('Gráficos de dispersión')
                for i, col in enumerate(x_columns):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(df[col], y, alpha=0.5)
                    ax.plot(df[col], y_pred, color='red', linewidth=2)
                    ax.set_xlabel(col)
                    ax.set_ylabel(y_column)
                    ax.set_title(f'Dispersión: {col} vs {y_column}')
                    st.pyplot(fig)
                
                # Verificación de supuestos
                st.subheader('Verificación de supuestos de la regresión Ridge')
                check_assumptions(X, y, y_pred, residuals)
                
                # Predicción
                st.subheader('Realización de Predicciones')
                x_pred = [st.number_input(f'Ingresa un valor de {col} para predecir Y:', value=float(df[col].mean())) for col in x_columns]
                prediction = model.predict([x_pred])
                st.latex(f"Predicción: Y = {prediction[0]:.4f} \\text{{ para }} X = {x_pred}")
                
                # Procedimiento de cálculo
                show_procedure(X, y, model, r2, mse)

if __name__ == '__main__':
    main()
