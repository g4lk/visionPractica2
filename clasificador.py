from sklearn.neighbors import KNeighborsClassifier
from extCaracteristicas import extCaracteristicas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class Clasificador():
    def __init__(self, resultado = None, clasificador = LinearDiscriminantAnalysis()):
        self.clasificador = clasificador
        self.resultado = resultado 

    def fit(self,imagenes,ets_train):
        self.clasificador.fit(extCaracteristicas(imagenes).caracteristicas,ets_train)

    def predict(self,imagenes):
        self.resultado = self.clasificador.predict(extCaracteristicas(imagenes).caracteristicas)
        return self.resultado

    def comprobar_resultados(self,array_resultados):
        '''
        Calcula porcentaje de aciertos
        :param array_resultados: Array resultado de la operacion predict del algoritmo
        :return: Porcentaje de aciertos del algoritmo
        '''
        porcentaje = np.sum(np.equal(self.resultado,array_resultados))/len(self.resultado) * 100
        return round(porcentaje, 2)
