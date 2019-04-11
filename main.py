from extCaracteristicas import extCaracteristicas
import numpy as np
import cv2 as cv
import os

def cogerImagen(name_img):
    try:
        img = cv.imread(name_img)
        return img
    except Exception:
        print('No se puede cargar la imagen,saliendo')
    return None


def main():
    path = '../train_recortadas/00'
    for imagen in os.listdir(path):
        img = cogerImagen(f'{path}/{imagen}')
        if img is not None:
            extC = extCaracteristicas(img)
            breakpoint();
if __name__ == '__main__':
    main()
