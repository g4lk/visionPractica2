import numpy as np
import cv2 as cv
import os, argparse, sys
from lda import LDA

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

def clasificar_lda(test_path,train_path):
    imagenes, y = recorrerDirectorios(train_path)
    lda = LDA()
    lda.fit(imagenes,y)
    imagenes2, nombre_imagenes = cogerImagenes(test_path)
    resultados = lda.predict(imagenes2)
    # Cogemos los dos primeros digitos del nombre de la imagen (etiqueta) o uno                                                                                      # en caso de estar entre el cero y el 9                                                                             
    array_resultados = np.array([int(nombre[1:2]) if nombre[0] == '0' else int(nombre[:2]) for nombre in nombre_imagenes])
    porcentaje = lda.comprobar_resultados(array_resultados)
    print(f'Porcentaje de aciertos: {porcentaje}')

def main():
    args = recoger_args()
    if args.classifier:
        if args.classifier == 'LDA-BAYES':
            clasificar_lda(args.test_path,args.train_path)
        if args.classifier == 'KNN-PCA':
            clasificar_knn(args.test_path,args.train_path)
    else:
        clasificar_lda(args.test_path,args.train_path)

if __name__ == '__main__':
    main()
