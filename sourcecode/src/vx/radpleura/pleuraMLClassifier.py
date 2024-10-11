
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
# !pip install pyradiomics
from skimage.feature.texture import local_binary_pattern
from radiomics.featureextractor import RadiomicsFeatureExtractor
import SimpleITK as sitk

import multiprocessing
from multiprocessing import Pool, Manager, Process, Lock
from xgboost import XGBClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
def classifier():
    classi = {
                "SVMC":{
                    "model":svm.SVC(kernel="rbf", probability=True, C=1, gamma=0.001),
                    "atts":{
                        "best_params":{'C': 300, 'gamma': 0.0001, 'kernel': 'rbf'},
                        "norm":"std",
                        "grid":False,
                        "isKfold":False,
                    }
                },
                "XGBC":{
                    "model":XGBClassifier(
                            objective= 'binary:logistic',
                            nthread=4,
                            seed=42,

                            learning_rate=0.1,
                            max_depth=4,
                            n_estimators=600,
                    ),
                    "atts":{
                        "best_params":{"learning_rate":0.1,"max_depth":4,"n_estimators":600},
                        "norm":"std",
                        "grid":False,
                        "isKfold":False,
                    }
                },
                # "KNNC":{
                #     "model":KNeighborsClassifier(n_neighbors = 3),
                #     "atts":{
                #         "best_params":{},
                #         "norm":"std",
                #         "grid":False,
                #         "isKfold":False,
                #     }
                # },                
            }
    return classi


def evaluation_metric(scores, y_true, y_pred):
    y_true, y_pred = y_true.tolist(), y_pred.tolist()
        
    acc = metrics.accuracy_score(y_true, y_pred, normalize=True)
    f1 = metrics.f1_score(y_true, y_pred)
    roc_auc_score = metrics.roc_auc_score(y_true, y_pred)
    pre = metrics.precision_score(y_true, y_pred)
    rec = metrics.recall_score(y_true, y_pred)
        
    scores["acc"].append(acc)
    scores["f1"].append(f1)
    scores["auc"].append(roc_auc_score)
    scores["pre"].append(pre)
    scores["rec"].append(rec)


def getmetricsv():
    return {"acc":[],"pre":[],"rec":[],"f1":[], "auc":[]}

def scaler(X, sc, model):
    #sc = StandardScaler()
    if model=="SVMC":
        X = pd.DataFrame(sc.fit_transform(X))
        #print(X)
    return X

def featureselecction(fe, X, y):
    # Train model
    #fe = X.columns.tolist()
    #print("fe", fe)

    estimator = ExtraTreesClassifier(n_estimators=150, n_jobs=-1)
    model = estimator.fit(X, y)
    #selection = estimator.feature_importances_.argsort().tolist()
    imp = sorted(zip(fe, model.feature_importances_), key=lambda x: x[1] * -1)
    rk = []
    for r in imp:
        # kys.append(r[0])
        # vls.append(r[1])
        #print("key", r[0], "value", r[1])
        rk.append(r[0])
    #print("rk", rk)
    return rk

def readdata(pathdir, bf=[]):
    
    dftrain = pd.read_csv(os.path.join(pathdir, "features", "train.csv"))
    ytrain = np.array(dftrain["label"].astype('category').cat.codes.tolist())
    dftrain = dftrain.drop(["image","label"], axis=1)
    #print("dftrain.columns", dftrain.columns)
    if len(bf)==0:
        bf = dftrain.columns

    Xtrain = dftrain[bf]
    fe  = dftrain.columns.tolist()


    Xtrain, Xvali, ytrain, yvali = train_test_split(
            Xtrain, ytrain, test_size=(0.05), random_state=42, stratify=ytrain)




    dftest = pd.read_csv(os.path.join(pathdir, "features", "test.csv"))
    classnamesv = dict(enumerate(dftest.label.astype('category').cat.categories))
    classnames = []
    for key, value in classnamesv.items():
        classnames.append(value)


    ytest = np.array(dftest["label"].astype('category').cat.codes.tolist())
    dftest = dftest.drop(["image","label"], axis=1)
    Xtest = dftest[bf]
    
    return Xtrain,  ytrain,  Xvali,  yvali,  Xtest,  ytest, classnames, fe


def setbf(fes, Xtrain, Xvali, Xtest):
    return Xtrain[fes], Xvali[fes],  Xtest[fes]


def makeclassification(pathdir):
    #bf = ["LPB"+str(i) for i in range(1,83)]

    #bfx = np.array(["LBP1","LBP2","LBP3","LBP4","LBP5","LBP6","LBP7","LBP8","LBP9","LBP10","LBP11","LBP12","LBP13","LBP14","LBP15","LBP16","LBP17","LBP18","LBP19","LBP20","LBP21","LBP22","LBP23","LBP24","LBP25","LBP26","LBP27","LBP28","LBP29","LBP30","LBP31","LBP32","LBP33","LBP34","LBP35","LBP36","LBP37","LBP38","LBP39","LBP40","LBP41","LBP42","LBP43","LBP44","LBP45","LBP46","LBP47","LBP48","LBP49","LBP50","LBP51","LBP52","LBP53","LBP54","LBP55","LBP56","LBP57","LBP58","LBP59","LBP60","LBP61","LBP62","LBP63","LBP64","LBP65","LBP66","LBP67","LBP68","LBP69","LBP70","LBP71","LBP72","LBP73","LBP74","LBP75","LBP76","LBP77","LBP78","LBP79","LBP80","LBP81","LBP82","original_firstorder_10Percentile","original_firstorder_90Percentile","original_firstorder_Energy","original_firstorder_Entropy","original_firstorder_InterquartileRange","original_firstorder_Kurtosis","original_firstorder_Maximum","original_firstorder_MeanAbsoluteDeviation","original_firstorder_Mean","original_firstorder_Median","original_firstorder_Minimum","original_firstorder_Range","original_firstorder_RobustMeanAbsoluteDeviation","original_firstorder_RootMeanSquared","original_firstorder_Skewness","original_firstorder_TotalEnergy","original_firstorder_Uniformity","original_firstorder_Variance","original_glcm_Autocorrelation","original_glcm_ClusterProminence","original_glcm_ClusterShade","original_glcm_ClusterTendency","original_glcm_Contrast","original_glcm_Correlation","original_glcm_DifferenceAverage","original_glcm_DifferenceEntropy","original_glcm_DifferenceVariance","original_glcm_Id","original_glcm_Idm","original_glcm_Idmn","original_glcm_Idn","original_glcm_Imc1","original_glcm_Imc2","original_glcm_InverseVariance","original_glcm_JointAverage","original_glcm_JointEnergy","original_glcm_JointEntropy","original_glcm_MCC","original_glcm_MaximumProbability","original_glcm_SumAverage","original_glcm_SumEntropy","original_glcm_SumSquares","original_glrlm_GrayLevelNonUniformity","original_glrlm_GrayLevelNonUniformityNormalized","original_glrlm_GrayLevelVariance","original_glrlm_HighGrayLevelRunEmphasis","original_glrlm_LongRunEmphasis","original_glrlm_LongRunHighGrayLevelEmphasis","original_glrlm_LongRunLowGrayLevelEmphasis","original_glrlm_LowGrayLevelRunEmphasis","original_glrlm_RunEntropy","original_glrlm_RunLengthNonUniformity","original_glrlm_RunLengthNonUniformityNormalized","original_glrlm_RunPercentage","original_glrlm_RunVariance","original_glrlm_ShortRunEmphasis","original_glrlm_ShortRunHighGrayLevelEmphasis","original_glrlm_ShortRunLowGrayLevelEmphasis","original_glszm_GrayLevelNonUniformity","original_glszm_GrayLevelNonUniformityNormalized","original_glszm_GrayLevelVariance","original_glszm_HighGrayLevelZoneEmphasis","original_glszm_LargeAreaEmphasis","original_glszm_LargeAreaHighGrayLevelEmphasis","original_glszm_LargeAreaLowGrayLevelEmphasis","original_glszm_LowGrayLevelZoneEmphasis","original_glszm_SizeZoneNonUniformity","original_glszm_SizeZoneNonUniformityNormalized","original_glszm_SmallAreaEmphasis","original_glszm_SmallAreaHighGrayLevelEmphasis","original_glszm_SmallAreaLowGrayLevelEmphasis","original_glszm_ZoneEntropy","original_glszm_ZonePercentage","original_glszm_ZoneVariance","original_ngtdm_Busyness","original_ngtdm_Coarseness","original_ngtdm_Complexity","original_ngtdm_Contrast","original_ngtdm_Strength","original_gldm_DependenceEntropy","original_gldm_DependenceNonUniformity","original_gldm_DependenceNonUniformityNormalized","original_gldm_DependenceVariance","original_gldm_GrayLevelNonUniformity","original_gldm_GrayLevelVariance","original_gldm_HighGrayLevelEmphasis","original_gldm_LargeDependenceEmphasis","original_gldm_LargeDependenceHighGrayLevelEmphasis","original_gldm_LargeDependenceLowGrayLevelEmphasis","original_gldm_LowGrayLevelEmphasis","original_gldm_SmallDependenceEmphasis","original_gldm_SmallDependenceHighGrayLevelEmphasis","original_gldm_SmallDependenceLowGrayLevelEmphasis"])
    #bfi = [1,13,14,15,17,19,25,28,33,40,42,53,54,59,62,63,64,65,66,67,68,78,79,80,81,84,88,96,97,100,102,105,113,114,116,118,119,121,124,126,130,133,140,146,148,158,161,162,165,170]
    #print(len(bfi))
    #bf = bfx[bfi[:65]]
    bf = []
    restuls = {}
    models = classifier()
    for key, value in models.items():
        restuls[key] = {"vali":getmetricsv(), "test":getmetricsv(), "timetrain":[], "timevali":[], "timetest":[]}


    sc = StandardScaler()
    for key, value in models.items():    
        Xtrain,  ytrain,  Xvali,  yvali,  Xtest,  ytest, classnames, fe = readdata(pathdir)

        # fes = featureselecction(fe, Xvali, yvali)
        # #print("fes", fes)
        # Xtrain, Xvali, Xtest = setbf(fes[:int(len(fes)/2)], Xtrain, Xvali, Xtest)

        

        tstart = time.time()
        model = value["model"]
        Xtrain = scaler(Xtrain, sc, key)
        model.fit(Xtrain, ytrain)
        restuls[key]["timetrain"].append(time.time()-tstart)
        


        tstart = time.time()
        Xvali = scaler(Xvali, sc, key)
        yvali_pred = model.predict(Xvali)
        restuls[key]["timevali"].append(time.time()-tstart)
        evaluation_metric(restuls[key]["vali"], yvali, yvali_pred)

        tstart = time.time()
        Xtest = scaler(Xtest, sc, key)
        ytest_pred = model.predict(Xtest)
        restuls[key]["timetest"].append(time.time()-tstart)
        evaluation_metric(restuls[key]["test"], ytest, ytest_pred)

        print("classnames", classnames)
        print(classification_report(ytest, ytest_pred, target_names=classnames))

    print(restuls)

das = [
        # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/500/20230825140850",
        # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/400/20230826121633",
        # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/300/20230826124152",
        "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/200/20230826124856",

]
for mp in das:
    for i in range(1,1+1):
        inputdir = mp+"/"+str(i)
        makeclassification(inputdir)