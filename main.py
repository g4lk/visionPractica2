from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import cv2 as cv
import os, argparse, sys
from clasificador import Clasificador
from sklearn.pipeline import Pipeline

def recoger_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', help='Ruta a imagenes de entrenamiento')
    parser.add_argument('--test_path', help='Ruta a imagenes de test')
    parser.add_argument('--classifier', help='Clase del detector')
    args = parser.parse_args()
    if args.test_path and args.train_path:
        return args
    else:
        print('Necesitamos ruta de imagenes de entrenamiento y imagenes de test')
        sys.exit(-1)

def cogerImagenes(path):
        imagenes = []
        nombre_imagenes = []
        for imagen in sorted(os.listdir(path)):
            img = cv.imread(f'{path}/{imagen}')
            if img is not None:
                imagenes.append(img)
                nombre_imagenes.append(imagen)
        return (imagenes,nombre_imagenes)

def recorrerDirectorios(path):
    imagenes = []
    y = []
    for dirc in sorted(os.listdir(path)):
        imgs, _ = cogerImagenes(f'{path}/{dirc}')
        imagenes += imgs
        for i in range(0,len(os.listdir(f'{path}/{dirc}')),1):
            if dirc[0] == '0':
                y.append(int(dirc[1]))
            else:
                y.append(int(dirc))
    return (imagenes,y)

def clasificar(test_path,train_path, clasificador):
    imagenes, y = recorrerDirectorios(train_path)
    clasif = Clasificador(clasificador = clasificador)
    clasif.fit(imagenes,y)
    imagenes2, nombre_imagenes = cogerImagenes(test_path)
    resultados = clasif.predict(imagenes2)
    # Cogemos los dos primeros digitos del nombre de la imagen (etiqueta) o uno                                                                                      # en caso de estar entre el cero y el 9                                                                             
    array_resultados = np.array([int(nombre[1:2]) if nombre[0] == '0' else int(nombre[:2]) for nombre in nombre_imagenes])
    porcentaje = clasif.comprobar_resultados(array_resultados)
    print(f'Porcentaje de aciertos: {porcentaje}')

def main():
    args = recoger_args()
    if args.classifier:
        if args.classifier.lower() == 'lda':
            clasificar(args.test_path,args.train_path,LinearDiscriminantAnalysis())
        if args.classifier.lower() == 'knn':
            clasificar(args.test_path,args.train_path,KNeighborsClassifier(n_neighbors=3))
        if args.classifier.lower() == 'randomforest':
            clasificar(args.test_path,args.train_path,RandomForestClassifier(n_estimators=\
                                                                             500, n_jobs =-1))
        if args.classifier.lower() == 'lda-knn':
            clasificar(args.test_path,args.train_path,Pipeline([\
                                                                ('lda',LinearDiscriminantAnalysis()),\
                                                                ('knn',KNeighborsClassifier(n_neighbors=6))]))
    else:
        clasificar(args.test_path,args.train_path,LinearDiscriminantAnalysis())

if __name__ == '__main__':
    main()
