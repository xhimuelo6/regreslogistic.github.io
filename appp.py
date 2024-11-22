import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
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

# Función para verificar los supuestos de la regresión lineal
def check_assumptions(X, y, y_pred, residuals):
    # Linealidad
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X, y, alpha=0.5)
    ax.plot(X, y_pred, color='red', linewidth=2)
    ax.set_xlabel('Variable independiente')
    ax.set_ylabel('Variable dependiente')
    ax.set_title('Gráfico de Linealidad')
    st.pyplot(fig)
    st.write("Interpretación del gráfico de linealidad:")
    st.write("- Los puntos azules representan los datos observados.")
    st.write("- La línea roja representa la recta de regresión ajustada.")
    st.write("- Si los puntos se distribuyen aleatoriamente alrededor de la línea roja, el supuesto de linealidad se cumple.")
    st.write("- Si se observa un patrón curvo, podría ser necesario considerar una transformación de variables o un modelo no lineal.")
    
    # Normalidad de residuos
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Gráfico Q-Q de Normalidad de Residuos")
    st.pyplot(fig)
    st.write("Interpretación del gráfico Q-Q:")
    st.write("- Si los puntos se alinean cerca de la línea diagonal, los residuos siguen una distribución normal.")
    st.write("- Desviaciones significativas de la línea diagonal sugieren que los residuos no son normales.")
    st.write("- Curvatura en forma de S indica asimetría en los residuos.")
    
    # Homocedasticidad
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.set_xlabel('Valores predichos')
    ax.set_ylabel('Residuos')
    ax.set_title('Gráfico de Homocedasticidad')
    ax.axhline(y=0, color='r', linestyle='--')
    st.pyplot(fig)
    st.write("Interpretación del gráfico de homocedasticidad:")
    st.write("- Si los puntos se distribuyen uniformemente alrededor de la línea horizontal roja, se cumple el supuesto de homocedasticidad.")
    st.write("- Si se observa un patrón de embudo o cualquier otro patrón sistemático, podría indicar heterocedasticidad.")
    st.write("- La heterocedasticidad puede afectar la precisión de las predicciones y la validez de las pruebas de hipótesis.")

# Función para calcular la regresión lineal
def calculate_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    return model, y_pred, residuals, r2, mse

# Función para mostrar el procedimiento de cálculo con fórmulas LaTeX
def show_procedure(X, y, model, r2, mse):
    st.subheader("Procedimiento de cálculo detallado")
    
    x_mean = np.mean(X)
    y_mean = np.mean(y)
    
    st.write("1. Cálculo de medias:")
    st.latex(f"\\bar{{X}} = {x_mean:.4f}")
    st.latex(f"\\bar{{Y}} = {y_mean:.4f}")
    
    st.write("2. Cálculo de la pendiente (b1):")
    numerator = np.sum((X - x_mean) * (y - y_mean))
    denominator = np.sum((X - x_mean)**2)
    b1 = numerator / denominator
    st.latex(f"b_1 = \\frac{{\\sum(X - \\bar{{X}})(Y - \\bar{{Y}})}}{{\\sum(X - \\bar{{X}})^2}} = \\frac{{{numerator:.4f}}}{{{denominator:.4f}}} = {b1:.4f}")
    
    st.write("3. Cálculo del intercepto (b0):")
    b0 = y_mean - b1 * x_mean
    st.latex(f"b_0 = \\bar{{Y}} - b_1 \\cdot \\bar{{X}} = {y_mean:.4f} - {b1:.4f} \\cdot {x_mean:.4f} = {b0:.4f}")
    
    st.write("4. Ecuación de la recta de regresión:")
    st.latex(f"Y = {b0:.4f} + {b1:.4f}X")
    
    st.write("5. Cálculo del coeficiente de determinación (R²):")
    st.latex(f"R^2 = 1 - \\frac{{\\sum(Y - \\hat{{Y}})^2}}{{\\sum(Y - \\bar{{Y}})^2}} = {r2:.4f}")
    st.write("   Donde Ŷ son los valores predichos por el modelo.")
    
    st.write("6. Cálculo del Error Cuadrático Medio (MSE):")
    st.latex(f"MSE = \\frac{{\\sum(Y - \\hat{{Y}})^2}}{{n}} = {mse:.4f}")
    st.write("   Donde n es el número de observaciones.")

# Función principal de la aplicación Streamlit
def main():
    st.title('Aplicación Completa de Regresión Lineal Simple')
    st.write("Esta aplicación realiza un análisis de regresión lineal simple, incluyendo visualizaciones, interpretaciones y cálculos detallados.")
    
    # Carga de datos
    uploaded_file = st.file_uploader("Elige un archivo CSV o Excel", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
        if df is not None:
            st.subheader("Vista previa de los datos")
            st.write(df.head())
            
            # Selección de variables
            columns = df.columns.tolist()
            x_column = st.selectbox('Selecciona la variable independiente (X)', columns)
            y_column = st.selectbox('Selecciona la variable dependiente (Y)', columns)
            
            X = df[x_column].values.reshape(-1, 1)
            y = df[y_column].values
            
            # Cálculo de la regresión
            model, y_pred, residuals, r2, mse = calculate_regression(X, y)
            
            # Resultados de la regresión
            st.subheader('Resultados de la Regresión Lineal')
            st.latex(f"Intercepto (b_0): {model.intercept_:.4f}")
            st.latex(f"Pendiente (b_1): {model.coef_[0]:.4f}")
            st.latex(f"R^2: {r2:.4f}")
            st.latex(f"MSE: {mse:.4f}")
            
            # Ecuación de la recta
            st.subheader('Ecuación de la recta de regresión')
            st.latex(f"Y = {model.intercept_:.4f} + {model.coef_[0]:.4f}X")
            
            # Gráfico de dispersión con línea de regresión
            st.subheader('Gráfico de dispersión con línea de regresión')
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(X, y, alpha=0.5)
            ax.plot(X, y_pred, color='red', linewidth=2)
            ax.set_xlabel(x_column)
            ax.set_ylabel(y_column)
            ax.set_title('Gráfico de Dispersión con Línea de Regresión')
            st.pyplot(fig)
            st.write("Interpretación del gráfico de dispersión:")
            st.write("- Los puntos azules representan los datos observados.")
            st.write("- La línea roja representa la recta de regresión ajustada.")
            st.write("- Cuanto más cerca estén los puntos a la línea, mejor será el ajuste del modelo.")
            st.write("- La pendiente de la línea indica la dirección y fuerza de la relación entre las variables.")
            
            # Verificación de supuestos
            st.subheader('Verificación de supuestos de la regresión lineal')
            check_assumptions(X, y, y_pred, residuals)
            
            # Predicción
            st.subheader('Predicción')
            x_pred = st.number_input('Ingresa un valor de X para predecir Y:', value=float(X.mean()))
            prediction = model.predict([[x_pred]])
            st.latex(f"Predicción: Y = {prediction[0]:.4f} \\text{{ para }} X = {x_pred:.4f}")
            
            # Procedimiento de cálculo
            show_procedure(X.flatten(), y, model, r2, mse)
            
            # Interpretación detallada
            st.subheader('Interpretación Detallada de los Resultados')
            
            # Interpretación del coeficiente (pendiente)
            st.write("1. Coeficiente (Pendiente):")
            st.latex(f"b_1 = {model.coef_[0]:.4f}")
            st.write(f"   - Por cada unidad que aumenta {x_column}, {y_column} cambia en promedio {model.coef_[0]:.4f} unidades.")
            if model.coef_[0] > 0:
                st.write(f"   - Existe una relación positiva entre {x_column} y {y_column}.")
            elif model.coef_[0] < 0:
                st.write(f"   - Existe una relación negativa entre {x_column} y {y_column}.")
            else:
                st.write(f"   - No parece haber una relación lineal entre {x_column} y {y_column}.")

            # Interpretación del intercepto
            st.write("2. Intercepto:")
            st.latex(f"b_0 = {model.intercept_:.4f}")
            st.write(f"   - Cuando {x_column} es 0, se espera que {y_column} sea {model.intercept_:.4f}.")
            st.write(f"   - El intercepto puede no tener una interpretación práctica si {x_column} nunca toma el valor 0 en el contexto del problema.")

            # Interpretación del R-cuadrado
            st.write("3. R-cuadrado:")
            st.latex(f"R^2 = {r2:.4f}")
            st.write(f"   - El modelo explica el {r2*100:.2f}% de la variabilidad en {y_column}.")
            if r2 < 0.3:
                st.write("   - El ajuste del modelo es débil. Pueden existir otras variables importantes que no se están considerando.")
            elif 0.3 <= r2 < 0.7:
                st.write("   - El ajuste del modelo es moderado. Explica una parte significativa de la variabilidad, pero aún hay variación no explicada.")
            else:
                st.write("   - El ajuste del modelo es fuerte. El modelo explica una gran parte de la variabilidad en los datos.")

            # Interpretación de la fuerza de la relación
            st.write("4. Fuerza de la relación:")
            if abs(model.coef_[0]) < 0.1:
                st.write(f"   - Hay una relación débil entre {x_column} y {y_column}.")
                st.write(f"   - Los cambios en {x_column} tienen un impacto mínimo en {y_column}.")
            elif 0.1 <= abs(model.coef_[0]) < 0.5:
                st.write(f"   - Hay una relación moderada entre {x_column} y {y_column}.")
                st.write(f"   - Los cambios en {x_column} tienen un impacto notable, pero no dramático, en {y_column}.")
            else:
                st.write(f"   - Hay una relación fuerte entre {x_column} y {y_column}.")
                st.write(f"   - Los cambios en {x_column} tienen un impacto sustancial en {y_column}.")

            st.write("5. Consideraciones adicionales:")
            st.write("   - Recuerda que la correlación no implica causalidad.")
            st.write("   - Verifica los supuestos de linealidad, normalidad y homocedasticidad en los gráficos proporcionados.")
            st.write("   - Considera si existen valores atípicos o influyentes que puedan afectar los resultados.")
            st.write("   - Ten en cuenta el contexto del problema y la relevancia práctica de las predicciones del modelo.")

if __name__ == '__main__':
    main()