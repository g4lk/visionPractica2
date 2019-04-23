from extCaracteristicas import extCaracteristicas
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

class LDA():
    def __init__(self,imagenes = None,ets_train = None,resultado = None):
        self.clf = LinearDiscriminantAnalysis()
        self.resultado = resultado 

    def fit(self,imagenes,ets_train):
        self.clf.fit(extCaracteristicas(imagenes).caracteristicas,ets_train)

    def predict(self,imagenes):
        self.resultado = self.clf.predict(extCaracteristicas(imagenes).caracteristicas)
        return self.resultado

    def comprobar_resultados(self,array_resultados):
        porcentaje = np.sum(np.equal(self.resultado,array_resultados))/len(self.resultado) * 100
        return round(porcentaje, 2)
