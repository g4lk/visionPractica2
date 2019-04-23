import cv2 as cv
import numpy as np

class extCaracteristicas():
    def __init__(self,imagenes):
        self.imagenes = imagenes
        self.caracteristicas = self.extraerCaracteristicas(imagenes)

    def extraerCaracteristicas(self,imagenes):
        imagenes_escaladas = [self.tratar(img) for img in imagenes]
        hog = cv.HOGDescriptor(_winSize = (32,32),_blockSize = \
                               (16,16),_blockStride = (8,8), _cellSize = \
                               (8,8),_nbins = 9)
        descriptores = [hog.compute(img).reshape(-1) for img in imagenes_escaladas]
        return np.array(descriptores)

    def tratar(self,img):

        imagen_ecualizada = self.ecualizar(img)
        imagen_gris = cv.cvtColor(imagen_ecualizada,cv.COLOR_BGR2GRAY)
        # Alinear bloques con HOG
        imagen_escalada = cv.resize(imagen_gris,(32,32),interpolation=cv.INTER_LINEAR)
        return imagen_escalada

    def ecualizar(self,img):
        try:
        #-----Converting image to LAB Color model-----------------------------------
            lab= cv.cvtColor(img, cv.COLOR_BGR2LAB)
        #-----Splitting the LAB image to different channels-------------------------
            l, a, b = cv.split(lab)
        #-----Applying CLAHE to L-channel-------------------------------------------
            clahe = cv.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            cl = clahe.apply(l)
        #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
            limg = cv.merge((cl,a,b)) 
        #-----Converting image from LAB Color model to RGB model--------------------
            final = cv.cvtColor(limg, cv.COLOR_LAB2BGR)
            return final
        except Exception:
            sys.exit(-1)
