
import multiprocessing
from multiprocessing import Pool, Manager, Process, Lock
from tabnanny import verbose
import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.twodim_base import mask_indices
import pandas as pd
import os
from skimage.feature.texture import local_binary_pattern
import time
import sys
import random
from radiomics.featureextractor import RadiomicsFeatureExtractor
import SimpleITK as sitk
from skimage import exposure
import time
from multiprocessing import Pool, Manager, Process, Lock


from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.ensemble import BaggingClassifier
from sklearn.multiclass import OneVsRestClassifier            

import joblib
import ujson


from xgboost import XGBClassifier
#import xgboost as xgb

class Classification:

    @staticmethod
    def execute_train(pathfilefeatures, pathfileinfo, pathout):
        clfs =  Classification.classifiers()
        
        #nameclf = "KNNC"
        nameclf = "XGBC"
        #nameclf = "RFC"
        #nameclf = "SVMC"
        #nameclf = "SVMC_BAG"
        #nameclf = "XGBC_GRID"
        #nameclf = "RFC_GRID"
        #nameclf = "SVMC_GRID"
        #nameclf = "SVMC_10FOLD"
        #nameclf = "XGBC_10FOLD"
        
        # make model path directory
        pathoutclf = os.path.join(pathout, nameclf)
        Classification.makedir(pathoutclf)

        clf = clfs[nameclf]["model"]
        atts = clfs[nameclf]["atts"]

        Classification.write(os.path.join(pathoutclf, "atts.json"), atts)

        # load best parameters
        if len(atts["best_params"])>0:
            clf = clf.set_params(**atts["best_params"])

        # read data
        df_fe = []
        fi_fe = os.path.join(pathfilefeatures, "features.csv")
        if True:
        #if not os.path.exists(fi_fe):
            df_aux = []
            for imageName in os.listdir(os.path.join(pathfilefeatures)):
                if imageName == "features.csv":
                    continue

                auxdf = pd.read_csv(os.path.join(pathfilefeatures, imageName))
                cols = len(auxdf.columns)
                #print(imageName,cols)
                #if cols < 700:
                #    print(imageName, cols)
                df_aux.append(pd.read_csv(os.path.join(pathfilefeatures, imageName)))
            df_fe = pd.concat(df_aux)
            df_fe.columns = df_aux[0].columns
            df_fe.to_csv(fi_fe, index=False)

        else:
            df_fe = pd.read_csv(fi_fe)
        
        
        df_if = pd.read_csv(os.path.join(pathfileinfo, "info.csv"))
        df_rest = df_fe.merge(df_if, on=["image","idseg"])


        aaa = df_rest.isnull().sum().sum()
        #print("aaa", aaa)
        #print("df_fe", df_rest)
        #print("df_if", df_if["image"])

        # save label classes
        datcat = dict(enumerate(df_rest.target.astype('category').cat.categories))
        Classification.write(os.path.join(pathoutclf, "labels.json"), datcat)

        # make X and Y
        Y = df_rest.target.astype('category').cat.codes
        X = df_rest.drop(["image","percentage","loc1","loc2","loc3","loc4","idseg","target"], axis=1)

        print(Y, X)
         
        # scaler
        if atts["norm"] != "none":
            scaler = Classification.getScaler(atts["norm"])
            #X = pd.DataFrame(scaler.fit_transform(X))
            X = scaler.fit_transform(X)


        scores = Classification.evaluation_tmp()

        if atts["isKfold"]:
        #if False:
            scoring = ['precision', 'recall', 'f1', 'accuracy', 'jaccard']
            cvskf = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
            #cvskf = 10

            #ytrue, ypred, fpr, tpr, roc_auc = Classification.kfolfcv(clf, X, Y, scores)
            scor = cross_validate(clf, X, Y, scoring=scoring, cv=cvskf, n_jobs=-1)

            scores["pre"] = scor['test_precision'].tolist()
            scores["rec"] = scor['test_recall'].tolist()
            scores["f1"] = scor['test_f1'].tolist()
            scores["acc"] = scor['test_accuracy'].tolist()
            scores["jac"] = scor['test_jaccard'].tolist()

            #dares = {"ytrue":ytrue.tolist(), "ypred":ypred.tolist(), "fpr":fpr.tolist(), "tpr":tpr.tolist(), "roc_auc":roc_auc, "scores":scores}
            #print("name, i, roc_auc, scores", name, i, roc_auc, scores)
            #Classification.write(outdir+"/"+name+"_"+str(i)+".json", dares)

        else:
            # split data
            trainX, testX, trainY, testY = train_test_split(
                                X, Y, stratify=Y, test_size=0.2, random_state=7)
            
            """ 
            clfdisk = Classification.readmodel(os.path.join(pathoutclf, "model.sav"))
            y_pred = clfdisk.predict(testX)
            """

            # training
            clf.fit(trainX, trainY)
            # testing
            y_pred = clf.predict(testX)

            # evaluation
            Classification.evaluation(scores, testY, y_pred)

            # save best parameters
            if atts["grid"]:
                Classification.write(os.path.join(pathoutclf, "bestparams.json"), clf.best_params_)

        print(scores)
        Classification.evaluationmean(scores)


        # save scores
        Classification.write(os.path.join(pathoutclf, "scores.json"), scores)

        # save model
        Classification.writemodel(clf, os.path.join(pathoutclf, "model.sav"))


    @staticmethod
    def classifiers():
        #************* clf.best_params_ {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

        #cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=7)
        #cv = ShuffleSplit(10, test_size=0.2, train_size=0.2, random_state=0)
        cv = ShuffleSplit(3, test_size=0.2, train_size=0.2, random_state=0)
        # Usando o constructor para criar o objeto 
        """
        cv = StratifiedShuffleSplit(n_splits = 20,       # 20 simulações.
                                    test_size = 0.2,     # 20% do dataset será de testes.
                                    train_size = 0.2,    # 20% do dataset será de train.
                                    random_state = 42)   # Permitir a reprodutibilidade.
        """
        """ 
        rf_parameters = {

                "n_estimators" : [10, 100, 1000],
                "max_features" : ['sqrt', 'log2']
        }
        """

        rf_parameters = {
            'n_estimators': [400, 700, 1000],
            'max_depth': [15,20,25],
            'max_leaf_nodes': [50, 100, 200]
        }

        svm_parameters = [
            {   
                "kernel": ["rbf"],
                "gamma": [0.1, 0.01, 0.001, 0.0001, 0.00001],
                "C": [1, 10, 100, 300, 500, 700, 900, 1000]
            },
            #{"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]        

        xgbc_parameters = [
            {
                'max_depth': [2, 4, 6, 8],
                'n_estimators': [50, 100, 400, 600],
                'learning_rate': [0.1, 0.01, 0.05]            
            }
        ]

        svm_bag_n_estimators = 1



        #clf = clf.set_params(**bestpar)
        classifiers = {
            "RFC":{
                "model":RandomForestClassifier(n_estimators=200, random_state=7, n_jobs=-1),
                "atts":{
                    "best_params":{},
                    "norm":"none",
                    "grid":False,
                    "isKfold":False,
                }
            },
            "KNNC":{
                "model":KNeighborsClassifier(n_neighbors = 3),
                "atts":{
                    "best_params":{},
                    "norm":"std",
                    "grid":False,
                    "isKfold":False,
                }
            },
            #{"C":300,"gamma":0.0001,"kernel":"rbf"}
            "SVMC":{
                "model":svm.SVC(kernel="rbf", probability=True, C=10, gamma=0.001),
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
                        seed=42                    
                ),
                "atts":{
                    "best_params":{"learning_rate":0.1,"max_depth":4,"n_estimators":600},
                    "norm":"std",
                    "grid":False,
                    "isKfold":False,
                }
            },
            "SVMC_BAG":{
                "model":OneVsRestClassifier(BaggingClassifier(svm.SVC(C=100, gamma=0.001, kernel='rbf'), max_samples=1.0 / svm_bag_n_estimators, n_estimators=svm_bag_n_estimators)),
                "atts":{
                    "best_params":{},
                    "norm":"std",
                    "grid":False,
                    "isKfold":False,
                }
            },




            "RFC_GRID":{
                "model":GridSearchCV( 
                    estimator = RandomForestClassifier(n_estimators=100, random_state=7, n_jobs=-1),
                    param_grid = rf_parameters,
                    #cv = 10,
                    cv = cv,
                    verbose=2,
                    #scoring="roc_auc",
                    scoring='accuracy',
                    error_score=0,
                    n_jobs=-1
                ),
                "atts":{
                    "best_params":{},
                    "norm":"none",
                    "grid":True,
                    "isKfold":False,
                }
            },        
            "SVMC_GRID":{
                "model":GridSearchCV(
                    estimator = svm.SVC(kernel='rbf'),
                    scoring='accuracy',
                    param_grid = svm_parameters,
                    cv = cv,
                    verbose=2,
                    n_jobs=-1
                ),
                "atts":{
                    "best_params":{},
                    "norm":"std",
                    "grid":True,
                    "isKfold":False,
                }
            },
            "XGBC_GRID":{
                "model":GridSearchCV(
                    estimator = XGBClassifier(
                        objective= 'binary:logistic',
                        nthread=4,
                        seed=42
                    ),
                    scoring='accuracy',
                    param_grid = xgbc_parameters,
                    cv = cv,
                    verbose=2,
                    n_jobs=-1
                ),
                "atts":{
                    "best_params":{},
                    "norm":"std",
                    "grid":True,
                    "isKfold":False,
                }
            },



            "SVMC_10FOLD":{
                #{"C":300,"gamma":0.0001,"kernel":"rbf"}
                "model":svm.SVC(kernel="rbf", probability=True, C=100, gamma=0.0001),
                "atts":{
                    "best_params":{'C': 300, 'gamma': 0.0001, 'kernel': 'rbf'},
                    "norm":"std",
                    "grid":False,
                    "isKfold":True,
                }
            },

            "XGBC_10FOLD":{
                "model":XGBClassifier(
                    objective= 'binary:logistic',
                    nthread=4,
                    seed=42                    
                ),
                "atts":{
                    "best_params":{"learning_rate":0.1,"max_depth":4,"n_estimators":600},
                    "norm":"std",
                    "grid":False,
                    "isKfold":True,
                }
            },

        }
        return classifiers
    
    @staticmethod
    def evaluation(scores, y_true, y_pred):

        y_true, y_pred = y_true.tolist(), y_pred.tolist()

        acc = metrics.accuracy_score(y_true, y_pred, normalize=True)
        f1 = metrics.f1_score(y_true, y_pred)
        jac = metrics.jaccard_score(y_true, y_pred)
        pre = metrics.precision_score(y_true, y_pred)
        rec = metrics.recall_score(y_true, y_pred)
        
        scores["acc"].append(acc)
        scores["f1"].append(f1)
        scores["jac"].append(jac)
        scores["pre"].append(pre)
        scores["rec"].append(rec)

    @staticmethod
    def evaluationmean(da):
        for k, v in da.items():
            da[k] = np.array(v).mean()

    @staticmethod
    def kfolfcv(model, X, y, scores):
        #auc
        tprs = []
        aucs = []
        mean_tpr = 0.0
        mean_fpr = np.linspace(0, 1, 100)        



        #Implementing cross validation        
        k = 10
        #kf = KFold(n_splits=k, random_state=7)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=7)

        yltrue, ylpred = np.array([]),np.array([])
        for train_index , test_index in skf.split(X, y):
            #X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            #y_train, y_test = y[train_index], y[test_index]
            X_train, X_test = X[train_index,:],X[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            
            Classification.evaluation(scores, y_test, y_pred)
            #print("kfolfcv", acc)
            #acc_score.append(acc)
            yltrue = np.concatenate((yltrue, y_test), axis=None)
            ylpred = np.concatenate((ylpred, y_pred), axis=None)



            probs = model.predict_proba(X_test)
            preds = probs[:,1]
            fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
            mean_tpr[0] = 0.0
            roc_auc = metrics.auc(fpr, tpr)
            aucs.append(roc_auc)
            #tprs.append(interp(mean_fpr, fpr, tpr))
            

        mean_tpr /= float(k)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)

        return yltrue, ylpred, mean_fpr, mean_tpr, mean_auc
        #avg_acc_score = sum(acc_score)/k        




    @staticmethod
    def evaluation_tmp():
        return {
            "acc":[],
            "f1":[],
            "jac":[],
            "pre":[],
            "rec":[],
            }

    @staticmethod
    def getScaler(norm):
        sc = None
        if norm == "std":
            sc = StandardScaler()
        elif norm == "minmax":
            sc = MinMaxScaler()

        return sc
    @staticmethod
    def writemodel(model, file):
        joblib.dump(model, file)
    
    @staticmethod
    def readmodel(file):
        return joblib.load(file)

    @staticmethod
    def write(file, obj):
        with open(file, "w") as filef:
            filef.write(ujson.dumps(obj))

    @staticmethod
    def read(file):
        data = {}
        with open(file,"r") as filef:
            data = (ujson.load(filef))
        return data

    @staticmethod
    def makedir(ndir):
        if not os.path.exists(ndir):
            os.makedirs(ndir)



    @staticmethod
    def predict(pathquery, idmodel, idrois):
        idmodelversion = "001"
        pathmodel = "./vx/radpleura/import/models/ML/{}/{}".format(idmodelversion, idmodel)
        clfdisk = Classification.readmodel(os.path.join(pathmodel, "model.sav"))

        Xtest = pd.read_csv(os.path.join(pathquery, "features.csv"))
        Xtest = Xtest.drop(["image","idseg"], axis=1)
        labels = Classification.read(os.path.join(pathmodel, "labels.json"))
        print("Xtest", Xtest)
        scaler = Classification.getScaler("std")
        Xtest = scaler.fit_transform(Xtest)

        y_pred = clfdisk.predict(Xtest).tolist()

        return y_pred, labels

if __name__ == "__main__":
    """ 
    fept = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/csv_media"
    inpt = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/tiles/erode_radius_30/0.01/500"
    oupt = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/csv_media"    
    Classification.execute_train(fept, inpt, oupt) """

    fept = "/mnt/sda6/software/frameworks/sourcecode/src/vx/radpleura/import/models/001/features"
    inpt = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/tiles/erode_radius_30/0.01/500"
    oupt = "/mnt/sda6/software/frameworks/sourcecode/src/vx/radpleura/import/models/001"    
    Classification.execute_train(fept, inpt, oupt)

    """
    fept = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/csv_media/002/features"
    inpt = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/tiles/erode_radius_30/0.01/500"
    oupt = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/csv_media/002"    
    Classification.execute_train(fept, inpt, oupt)"""





