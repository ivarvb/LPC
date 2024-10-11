#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:35:39 2020
@author: oscar
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import pandas as pd
from sklearn.metrics import classification_report, plot_confusion_matrix
#from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def SplitData(dataPath, dataFileName):
    data = pd.read_csv(dataPath+dataFileName) #data with headers and meta-data
    data.dropna(inplace=True);
        
    xSet = data.loc[:, 'feature_1':'feature_64'] #fractal + lbp + coo + moments
    ySet = data['Label']
    
    scaler = StandardScaler()
    scaler.fit(xSet)
    xSet = pd.DataFrame(scaler.transform(xSet)) # to pandas data frame
    
    return data, xSet, ySet

def FeatureSelection(xSet, ySet, nFeatures=10):
    
    sffs = SFS(svm.SVC(kernel='rbf', C=1), 
           k_features=nFeatures, 
           forward=True, 
           floating=True, 
           verbose=2,
           scoring='accuracy',
           cv=10,
           n_jobs=-1) #-1
    
    sffs = sffs.fit(xSet, ySet)
    
    print('\nSequential Forward Floating Selection:')
    print(sffs.k_feature_idx_)
    print('CV Score:')
    print(sffs.k_score_)
    return sffs
    

def Train(xSet, ySet):
        
    paramGrid = {'C': [1, 10, 100, 1000, 10000], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 'scale', 'auto'], }
    estimator = GridSearchCV(svm.SVC(kernel='rbf'), paramGrid)
    #clf = svm.SVC(gamma='auto', C=10 , kernel="rbf") 
    estimator.fit(xSet, ySet);
    
    print("Best estimator found by grid search:")
    print(estimator.best_estimator_)
    return estimator


    
def Test(data, xSet, ySet , estimator, targetNames, dataPath, outputFileName, writeCSV=False):
    pred = estimator.predict(xSet)
    print(classification_report(ySet, pred, target_names=targetNames))
    
    #writing results 
    data['Prediction'] = pred
    #print(data)
    if writeCSV:
        data.to_csv(dataPath+outputFileName+".csv");
      
        


def main():

    dataPath = "/home/oscar/data/biopsy/tiff/dataset_1/csv/kernel_size_"
    trainFileName = "50_training.csv"
    testFileName  = "50_test.csv"
    
    
    data, xSet, ySetTrain = SplitData(dataPath, trainFileName);
    
       
    sffs = FeatureSelection(xSet, ySetTrain, 5)
    
    xSetTrain = sffs.transform(xSet);
        
    
    
    estimator = Train(xSetTrain, ySetTrain);
    
    data, xSet, ySetTest = SplitData(dataPath, testFileName);
    xSetTest = sffs.transform(xSet);
        
    targetNames = ['Non_pleura', 'Pleura']
    Test(data, xSetTest, ySetTest, estimator, targetNames, dataPath, testFileName+"_classification", True)
    
    
if __name__=="__main__":
    main()











