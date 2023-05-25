#este método es a modo de ejemplo, ya que también utiliza el método clásico de regresión:

#importamos las librerías a utilizar:
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

#cargamos los datos:
file = 'https://raw.githubusercontent.com/fhernanb/Python-para-estadistica/master/03%20Regression/Regresi%C3%B3n%20lineal%20simple/softdrink.csv'
dt = pd.read_csv(file)
dt.head()

# graficamos los datos para ver la distribución:
dt.plot(kind='scatter', x='x1', y='y');
X = dt["x1"] # definimos la variable independiente
X = sm.add_constant(X) # agragamos la ordenada al origen
y = dt["y"] # definimos la variable respuesta
plt.xlabel('Cantidad de cajas')
plt.ylabel('Tiempo (min)')
plt.title('Datos');
# creamos el modelo de regresión:
mod = smf.ols('y ~ x1', data=dt).fit() #con esta instrucción se realiza el método de regresión directamente
print(mod.summary()) # imprimimos los datos del modelo


#graficamos la recta junto con los puntos y las etiquetas correspondientes:
fig, ax = plt.subplots(figsize=(8,6))
ax.plot(dt["x1"] , dt["y"] , 'o', label="Datos")
ax.plot(dt["x1"], mod.fittedvalues, 'r--.', label="Ajustado")
legend = ax.legend(loc="best")
plt.xlabel('Cantidad de cajas')
plt.ylabel('Tiempo (min)')
plt.title('Diagrama de dispersión con la recta del modelo ajustado');
plt.show()
