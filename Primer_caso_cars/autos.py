#!/usr/bin/env python3

# ESTE ES UN PRIMER CASO, SIMILAR AL QUE HICIMOS EN LAS PRÁCTICAS, PARA ENTENDER CON UN EJEMPLO
# EL MÉTODO DE AJUSTES POR CUADRADOS MÍNIMOS A LA HORA DE HACER REGRESIÓN LINEAL (SIMPLE).
# EN LA PRESENTACIÓN SE DA UN MARCO TEÓRICO SOBRE EL MÉTODO Y SE VEN LAS GRÁFICAS CORRESPONDIENTES

# Se importan la librerias a utilizar
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x, y =np.loadtxt("autos.dat", usecols=(0,1), unpack=True)  #carganmos la tabla de datos

# El conjunto de datos da la velocidad de los coches y las distancias necesarias para detenerse.
# Fueron registrados en experimentaciones con autos de la década de 1920.


# damos la instrucción de hacer la regresión, definiendo los parámetros que se van a estimar:
radient, intercept, r_value, p_value, std_err=stats.linregress(x,y)
min_x=np.min(x)
max_x=np.max(x)

# Ahora utilizamos los datos anteriores y armamos la ecuación que estima la recta del modelo lineal:
x_fit=np.linspace(min_x,max_x,1000)
y_fit=gradient*x_fit+intercept

# imprimimos en pantalla los datos de la estimación:
print('la ordenada al origen es:')
print(intercept)
print('la pendiente es:')
print(gradient)
print('el coeficiente de correlación es:')
print(r_value) #coef de correlacion
print('el coeficiente de determinación es:')
print(r_value**2) #coeficiente de determinación (coeficiente de correlación al cuadrado)


# damos la orden de que se impriman la pendiente y la ordenada al origen en el gráfico que vamos a hacer:
pendiente_label="%s %.3f" % ('Pendiente=', gradient)
plt.annotate(pendiente_label, xy=(3,6), xycoords='data', xytext=(0.02, 0.8), textcoords='axes fraction', horizontalalignment='left', verticalalignment='top', color='blue')
intercept_label="%s %.3f" % ('Ordenada al origen=', intercept)
plt.annotate(intercept_label, xy=(6,6), xycoords='data', xytext=(0.02, 0.9), textcoords='axes fraction', horizontalalignment='left', verticalalignment='top', color='blue')

# graficamos:
plt.scatter(x, y, color='blue')
plt.scatter(x_fit,y_fit,label='Fiteo Lineal', color='red', s=1)
plt.xlabel('velocidad')
plt.ylabel('distancia de frenado')
# guardamos el gráfico listo en formato png:
plt.title('Ajuste de regresión lineal simple para datos de autos')
plt.savefig('ajuste_autos_python.png')
plt.show()
