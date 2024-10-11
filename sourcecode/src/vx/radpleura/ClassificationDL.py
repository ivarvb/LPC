
import pandas as pd
import os
import numpy as np

import cv2 as cv2
import tensorflow as tf
import SimpleITK as sitk

from tensorflow import keras
from keras.layers import *

def getClassesLabels():
    # https://www.kaggle.com/code/th3niko/transfer-learning-xception
    pathtestdata = "./vx/radpleura/import/database/pleuradb"
    test_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
    test_set = test_datagen.flow_from_directory(pathtestdata, target_size=(256,256))
    inv_map = {v: k for k, v in test_set.class_indices.items()}
    labels = [0 for i in range(len(inv_map)) ]
    for k, v in inv_map.items():
        labels[k] = v

    del test_datagen
    del test_set
    return labels

def getDLModels():
    pathmodelsDL = "./vx/radpleura/import/models/DL"
    idmodelversion = "001"
    model = {}
    model["ResNet_odspleura200tile1black"] = keras.models.load_model(os.path.join(pathmodelsDL,
                                                    idmodelversion,
                                                    'ResNet_odspleura200tile1black.h5'))
    return model


class ClassificationDL:
    model = getDLModels()
    labels = getClassesLabels()
    @staticmethod
    def readTiles(pathquery, idrois):
        image_mask = sitk.ReadImage(os.path.join(pathquery, "roids.nrrd"))
        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(image_mask)
        labels = lsif.GetLabels()

        tiles = []
        for label in labels:
            input_arr = cv2.imread(os.path.join(pathquery, "roids", str(label)+".jpg"), cv2.IMREAD_GRAYSCALE)
            if len(input_arr.shape)==2:
                input_arr = cv2.merge([input_arr, input_arr, input_arr])
            tiles.append(input_arr/255.0)
        tiles = np.array(tiles)

        #for i in idrois:
        return tiles
        
        """
        X = np.load(os.path.join(pathquery,"roids.npy"))
        tiles = []
        for i in idrois:
            input_arr = X[i]
            if len(input_arr.shape)==2:
                input_arr = cv2.merge([input_arr, input_arr, input_arr])
            tiles.append(input_arr/255.0)
        del X
        tiles = np.array(tiles)
        return tiles
        """

    @staticmethod
    def predict(pathquery, idmodel, idrois):
        ypred = []
        if idmodel in ClassificationDL.model:
            X = ClassificationDL.readTiles(pathquery, idrois)
            wh = X[0].shape[0]
            print("X.shape", X.shape)

            X = X.reshape(-1,wh,wh,3) 
            preds = ClassificationDL.model[idmodel].predict(X)
            for pp in preds:
                ypred.append(pp.argmax())
        return ypred, ClassificationDL.labels

