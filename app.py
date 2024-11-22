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

# Función para verificar los supuestos de la regresión lineal múltiple
def check_assumptions(X, y, y_pred, residuals):
    st.subheader('Prueba de Normalidad de Residuos')
    # Normalidad de residuos - Gráfico Q-Q
    fig, ax = plt.subplots(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Gráfico Q-Q de Normalidad de Residuos")
    st.pyplot(fig)
    st.write("*Prueba de Normalidad:*")
    st.write("- Si los puntos se alinean cerca de la línea diagonal, los residuos siguen una distribución normal.")
    st.write("- Desviaciones significativas de la línea diagonal sugieren que los residuos no son normales.")
    st.write("- La normalidad de los residuos es importante para asegurarnos de que las inferencias y los intervalos de confianza del modelo sean válidos. Si los residuos no son normales, podríamos estar violando este supuesto, lo que afectaría la precisión de las pruebas de hipótesis.")

    # Gráfico de Linealidad
    st.subheader('Ajuste del Modelo de Regresión Lineal Múltiple')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y, y_pred, alpha=0.5)
    ax.plot([min(y), max(y)], [min(y_pred), max(y_pred)], color='red', linewidth=2)
    ax.set_xlabel('Valores Observados (Y)')
    ax.set_ylabel('Valores Predichos (Ŷ)')
    ax.set_title('Gráfico de Linealidad: Observados vs Predichos')
    st.pyplot(fig)
    st.write("*Ajuste del Modelo:*")
    st.write("- La línea roja indica un ajuste perfecto entre los valores observados y los valores predichos.")
    st.write("- Si los puntos están alineados con la línea roja, el modelo tiene un buen ajuste.")
    st.write("- Un buen ajuste lineal indica que la relación entre las variables independientes y la dependiente es aproximadamente lineal. Si se observan patrones curvos o sistemáticos, deberías considerar ajustes adicionales, como modelos no lineales o transformaciones.")

    # Homocedasticidad
    st.subheader('Evaluación del Modelo: Homocedasticidad de Residuos')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.set_xlabel('Valores Predichos (Ŷ)')
    ax.set_ylabel('Residuos')
    ax.set_title('Gráfico de Homocedasticidad de Residuos')
    ax.axhline(y=0, color='r', linestyle='--')
    st.pyplot(fig)
    st.write("*Evaluación del Modelo:*")
    st.write("- Si los puntos se distribuyen uniformemente alrededor de la línea roja horizontal (residuo = 0), se cumple el supuesto de homocedasticidad.")
    st.write("- Si se observa un patrón en la distribución de los residuos (como un embudo), podría indicar heterocedasticidad, lo que afecta la precisión del modelo.")
    st.write("- La homocedasticidad es clave porque asegura que la variabilidad de los errores es constante a lo largo de los valores predichos. Si no se cumple, las pruebas de significancia y los intervalos de confianza pueden ser incorrectos o poco fiables.")

# Función para calcular la regresión lineal múltiple
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
    
    st.write("1. Ecuación del modelo de regresión múltiple:")
    coef_str = " + ".join([f"{coef:.4f} X{i+1}" for i, coef in enumerate(model.coef_)])
    st.latex(f"Y = {model.intercept_:.4f} + {coef_str}")
    
    st.write("2. Cálculo del coeficiente de determinación (R²):")
    st.latex(f"R^2 = 1 - \\frac{{\\sum(Y - \\hat{{Y}})^2}}{{\\sum(Y - \\bar{{Y}})^2}} = {r2:.4f}")
    
    st.write("3. Cálculo del Error Cuadrático Medio (MSE):")
    st.latex(f"MSE = \\frac{{\\sum(Y - \\hat{{Y}})^2}}{{n}} = {mse:.4f}")

# Función principal de la aplicación Streamlit
def main():
    st.title('Aplicación Completa de Regresión Lineal Múltiple')
    st.write("Esta aplicación realiza un análisis de regresión lineal múltiple, incluyendo visualizaciones, interpretaciones y cálculos detallados.")
    
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
            
                # Cálculo de la regresión
                model, y_pred, residuals, r2, mse = calculate_regression(X, y)
                
                # Resultados de la regresión
                st.subheader('Resultados de la Regresión Lineal Múltiple')
                st.latex(f"Intercepto (b_0): {model.intercept_:.4f}")
                coef_str = ", ".join([f"b_{i+1} = {coef:.4f}" for i, coef in enumerate(model.coef_)])
                st.latex(f"Coeficientes: {coef_str}")
                st.latex(f"R^2: {r2:.4f}")
                st.latex(f"MSE: {mse:.4f}")
                
                # Ecuación del modelo
                st.subheader('Ecuación del modelo de regresión')
                coef_str = " + ".join([f"{coef:.4f} * X{i+1}" for i, coef in enumerate(model.coef_)])
                st.latex(f"Y = {model.intercept_:.4f} + {coef_str}")
                
                # Gráfico de dispersión
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
                st.subheader('Verificación de supuestos de la regresión lineal múltiple')
                check_assumptions(X, y, y_pred, residuals)
                
                # Predicción
                st.subheader('Realización de Predicciones')
                x_pred = [st.number_input(f'Ingresa un valor de {col} para predecir Y:', value=float(df[col].mean())) for col in x_columns]
                prediction = model.predict([x_pred])
                st.latex(f"Predicción: Y = {prediction[0]:.4f} \\text{{ para }} X = {x_pred}")
                
                # Interpretación de la predicción
                st.subheader('Interpretación de la Predicción')
                st.write(f"La predicción estimada para la variable dependiente *{y_column}* dada la entrada de las variables independientes *{x_columns}* es *{prediction[0]:.4f}*.")
                st.write(f"Esto significa que, dado el valor de las variables independientes: {x_pred}, se espera que el valor de {y_column} sea aproximadamente {prediction[0]:.4f}.")
                st.write("Es importante recordar que esta predicción está basada en los datos y el modelo lineal ajustado. Si el modelo tiene un buen ajuste (con un valor alto de \( R^2 \)), la predicción será más confiable. Sin embargo, si \( R^2 \) es bajo, es posible que otras variables importantes no estén incluidas en el modelo, lo que puede afectar la precisión de la predicción.")
                st.write("Además, siempre es recomendable validar el modelo con datos adicionales o utilizar técnicas de validación cruzada para evaluar la robustez de las predicciones.")
                
                # Procedimiento de cálculo
                show_procedure(X, y, model, r2, mse)
                
                # Interpretación detallada
                st.subheader('Interpretación Detallada de los Resultados')
                
                st.write("1. Coeficientes:")
                for i, col in enumerate(x_columns):
                    st.latex(f"b_{i+1} = {model.coef_[i]:.4f}")
                    st.write(f"   - Por cada unidad que aumenta {col}, {y_column} cambia en promedio {model.coef_[i]:.4f} unidades.")
                    if model.coef_[i] > 0:
                        st.write(f"   - Existe una relación positiva entre {col} y {y_column}.")
                    elif model.coef_[i] < 0:
                        st.write(f"   - Existe una relación negativa entre {col} y {y_column}.")
                    else:
                        st.write(f"   - No parece haber una relación lineal entre {col} y {y_column}.")
                
                # Interpretación del intercepto
                st.write("2. Intercepto:")
                st.latex(f"b_0 = {model.intercept_:.4f}")
                st.write(f"   - Cuando todas las variables independientes son 0, se espera que {y_column} sea {model.intercept_:.4f}.")
                st.write("   - El intercepto puede no tener una interpretación práctica si las variables independientes no toman el valor 0 en el contexto del problema.")
                
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
                for i, col in enumerate(x_columns):
                    if abs(model.coef_[i]) < 0.1:
                        st.write(f"   - La relación entre {col} y {y_column} es débil. Los cambios en {col} tienen un impacto mínimo en {y_column}.")
                    elif 0.1 <= abs(model.coef_[i]) < 0.5:
                        st.write(f"   - La relación entre {col} y {y_column} es moderada. Los cambios en {col} tienen un impacto notable.")
                    else:
                        st.write(f"   - La relación entre {col} y {y_column} es fuerte. Los cambios en {col} tienen un impacto sustancial.")
                
                # Condiciones adicionales con interpretación detallada
                st.subheader('Condiciones Adicionales con Interpretación Detallada')
                st.write("*1. Correlación vs Causalidad:*")
                st.write("- *Correlación* significa que dos variables tienen una relación estadística, lo que implica que cambian juntas de alguna manera, pero no necesariamente porque una sea la causa de la otra.")
                st.write("- *Causalidad*, en cambio, indica que un cambio en una variable provoca directamente un cambio en la otra.")
                st.write("- Una correlación no prueba causalidad. Aunque dos variables estén correlacionadas, esto puede ser debido a otros factores subyacentes o a una coincidencia.")
                st.write("- Para determinar una relación causal, se requiere un análisis más riguroso, como un experimento controlado o el uso de técnicas estadísticas avanzadas que puedan descartar variables externas y establecer un vínculo directo.")

                
                st.write("*2. Verificación de Supuestos:*")
                st.write("- Es crucial confirmar que los *supuestos fundamentales de la regresión lineal múltiple* se cumplan para garantizar la validez de los resultados y las inferencias.")
                st.write("  - *Linealidad*: La relación entre las variables independientes y la variable dependiente debe ser aproximadamente lineal. Si esta suposición no se cumple, las predicciones del modelo pueden ser incorrectas.")
                st.write("  - *Normalidad de residuos*: Los residuos (diferencia entre los valores observados y predichos) deben estar distribuidos normalmente. Esto asegura que las pruebas de hipótesis y los intervalos de confianza sean confiables.")
                st.write("  - *Homocedasticidad*: La varianza de los residuos debe ser constante a lo largo de todos los niveles de las variables independientes. Si hay heterocedasticidad (varianza no constante), los errores estándar pueden estar mal calculados, lo que afectaría la interpretación de los coeficientes y la significancia estadística.")

                st.write("*3. Valores Atípicos e Influyentes:*") 
                st.write("- Los *valores atípicos* pueden distorsionar los coeficientes de regresión y llevar a conclusiones incorrectas.")
                st.write("- Los *puntos influyentes* son aquellos que, debido a su posición en el conjunto de datos, pueden cambiar drásticamente el ajuste del modelo si se eliminan.")
                st.write("- Para detectarlos, puedes utilizar herramientas visuales como el *gráfico de residuos* (residual plot) o estadísticas específicas, como la distancia de Cook, que ayudan a identificar estos puntos problemáticos.")

                
                st.write("*4. Contexto del Problema:*")
                st.write("- Al interpretar los resultados de cualquier análisis, es fundamental hacerlo en función del *contexto del problema*. Los resultados estadísticos, por sí solos, pueden no ser suficientes para obtener conclusiones válidas si no se consideran las circunstancias y la lógica detrás del fenómeno que se está estudiando.")
                st.write("- Asegúrate de que los hallazgos no solo sean estadísticamente significativos, sino que también tengan *sentido práctico* dentro del marco del problema que estás abordando. Si los resultados no tienen aplicabilidad real, su valor puede ser limitado.")
                st.write("- También es importante estar atento a *relaciones inesperadas* o inconsistencias que puedan surgir en los resultados. Si algo contradice las teorías conocidas o el conocimiento previo sobre el tema, podría ser una señal de que necesitas revisar tus supuestos, el modelo, o la calidad de los datos.")

if __name__ == '__main__':
    main()
