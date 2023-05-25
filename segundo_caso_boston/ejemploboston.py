#Se importan la librerias a utilizar
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

#cargamos los datos
boston = datasets.load_boston()
print(boston)
print()

print('Cantidad de datos:')
print(boston.data.shape)
print()

#Seleccionamos la columna 7 del dataset correspondiente a la distancia de las casas de Boston al centro de trabajo:
X = boston.data[:, np.newaxis, 7]
#Seleccionamos la columna 4 del dataset correspondiente a la concentración de óxido nítrico en el agua:
y = boston.data[:, np.newaxis, 4]

#Graficamos los datos correspondientes para ver la distribución:
plt.scatter(X, y, color="black")
plt.xlabel(' distancia al centro de Boston')
plt.ylabel('concentración de óxido nítrico')
plt.show()

#Hacemos la regresión:
from sklearn.model_selection import train_test_split
#Separo los datos de "train" en entrenamiento y prueba para probar los algoritmos
#(esta función divide los datos en prueba y entrenamiento para realizar la predicción)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9) #se ejecutó para 0.05, 0.1, 0.2, 0.5 y 0.9
#utilizamos la siguiente instrucción para la regresión
lr = linear_model.LinearRegression()
#Entreno el modelo
lr.fit(X_train, y_train)
#Realizo la predicción
Y_pred = lr.predict(X_test)

#A continuación se grafican los datos junto con el modelo
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, Y_pred, color='blue', linewidth=1)
plt.title('Regresión Lineal Simple')
plt.xlabel('distancia al centro de Boston')
plt.ylabel('concentración de óxido nítrico')
plt.show()
print()
print('DATOS DEL MODELO REGRESIÓN LINEAL SIMPLE')
print()
print('Valor de la pendiente:')
print(lr.coef_)
print('Valor de la ordenada al origen:')
print(lr.intercept_)
print()
print('La ecuación del ajuste es:')
print('y = ', lr.coef_, 'x ', lr.intercept_)
print()
print('Precisión del modelo:')
print(lr.score(X_train, y_train))
