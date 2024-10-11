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


def makedir(ndir):
    if not os.path.exists(ndir):
        os.makedirs(ndir)

def isnullmask(fmask):
    mask = cv2.imread(fmask, cv2.IMREAD_GRAYSCALE)
    lab = 255
    ismask = mask[np.where(mask == lab)]
    r = False
    if len(ismask)>0:
        r = True
    return r


def lbp(fi, fmask):
    filename = os.path.basename(fi)
    bfile = os.path.splitext(filename)


    inputImage = cv2.imread(fi, cv2.IMREAD_GRAYSCALE)
    #image_mask = sitk.ReadImage(fmask)
    #image_mask = cv2.imread(fi, cv2.IMREAD_GRAYSCALE) > 0
    mask = cv2.imread(fmask, cv2.IMREAD_GRAYSCALE)
    #mask = sitk.GetArrayFromImage(image_mask)
    
    lab = 255
    radiusx = [10]
    nPoints = 8 * radius
    for radius in radiusx:
        #print("inputImage", inputImage, fi)
        lbpx = local_binary_pattern(inputImage, nPoints, radius, method='uniform')
        xnBins = int(lbpx.max() + 1)
        lbp = lbpx[np.where(mask == lab)]
        histogram, _ = np.histogram(lbp, bins=xnBins, range=(0, xnBins))
    aux = histogram.tolist()
    del inputImage
    del mask
    print("LBP",len(aux))
    return aux
    

def radiomics(fi, fmask):
    image = sitk.ReadImage(fi, sitk.sitkFloat32)
    mask = sitk.ReadImage(fmask)###########?????????????
    label = 255
    ###def pyradiomics(image, mask, label):
    settings = {}
    extractor = RadiomicsFeatureExtractor(**settings)

    # nuevo
    extractor.enableImageTypes(
        Original={},
        Wavelet={},
        #LoG={'sigma':[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]},
        #LoG={'sigma':[1, 1.5, 2, 2.5, 3]},
        #LoG={'sigma':[1.5]},
        #LBP2D={},
    )


    extractor.disableAllFeatures()
    extractor.enableFeatureClassByName('firstorder')#19
    extractor.enableFeatureClassByName('glcm')#24
    extractor.enableFeatureClassByName('glrlm')#16   
    extractor.enableFeatureClassByName('glszm')#16
    extractor.enableFeatureClassByName('ngtdm')#5
    extractor.enableFeatureClassByName('gldm')#14

    result = extractor.execute(image, mask, label=label)
    fe_values = []
    #names = []
    for k,v in result.items():
        if k.startswith('original') or k.startswith('wavelet') or k.startswith('log') or k.startswith('lbp') or k.startswith('logarithm')  or k.startswith('exponential') or k.startswith('squareroot') or k.startswith('gradient'): 
            fe_values.append("{:.8f}".format(v))
            #names.append(k)
    print("fe_rad_values",fe_values)
    del image
    del mask
    del extractor
    print("RAD",len(fe_values))
    #print("RAD names",names)
    return fe_values
    
    

def oneimage(arg):
    filename = os.path.basename(arg["file"])
    #bfile = os.path.splitext(filename)
    
    #fmask = os.path.join(arg["pathmk"], arg["ispleura"], filename)
    #print("fmask", fmask)
    flbp = lbp(arg["file"], arg["pathmk"])
    frad = radiomics(arg["file"], arg["pathmk"])

    #+radiomics(arg["file"])

    return [filename]+flbp+frad+[arg["ispleura"]]
    #return flbp

def feature_extraction(dat):

    pathin, pathmk, pathout = dat["pathin"],  dat["pathmk"],  dat["pathout"]
    makedir(pathout)    

    fes = []
    #print("pathin", pathin)
    for sset in ["test","train"]:
        arg = []
        for fi in os.listdir(os.path.join(pathin, sset, "pleura")):
            ar = {  "file":os.path.join(pathin, sset, "pleura", fi),
                    "ispleura":"pleura",
                    "pathmk":os.path.join(pathmk, "pleura", fi),
                }

            if isnullmask(ar["pathmk"]):
                arg.append(ar)
            else:
                print("fi null pleura", sset, fi)

            #fes.append(oneimage(ar))
            print("pleura",fi)
        for fi in os.listdir(os.path.join(pathin, sset, "nonpleura")):
            ar = {  "file":os.path.join(pathin, sset, "nonpleura", fi),
                    "ispleura":"nonpleura",
                    "pathmk":os.path.join(pathmk, "nonpleura", fi),
                }
            if isnullmask(ar["pathmk"]):
                arg.append(ar)
            else:
                print("fi null nonpleura", sset, fi)
            #fes.append(oneimage(ar))
            print("nonpleura",fi)

        pool = Pool(processes = int(multiprocessing.cpu_count()/2))
        rs = pool.map(oneimage, arg)
        pool.close()

        dfs = []
        for row in rs:
            dfs.append(row)

        lbpnames = ["LBP"+str(i+1) for i in range(82)]
        # radnames = [
        #     'original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis']

        radnames = ['original_firstorder_10Percentile', 'original_firstorder_90Percentile', 'original_firstorder_Energy', 'original_firstorder_Entropy', 'original_firstorder_InterquartileRange', 'original_firstorder_Kurtosis', 'original_firstorder_Maximum', 'original_firstorder_MeanAbsoluteDeviation', 'original_firstorder_Mean', 'original_firstorder_Median', 'original_firstorder_Minimum', 'original_firstorder_Range', 'original_firstorder_RobustMeanAbsoluteDeviation', 'original_firstorder_RootMeanSquared', 'original_firstorder_Skewness', 'original_firstorder_TotalEnergy', 'original_firstorder_Uniformity', 'original_firstorder_Variance', 'original_glcm_Autocorrelation', 'original_glcm_ClusterProminence', 'original_glcm_ClusterShade', 'original_glcm_ClusterTendency', 'original_glcm_Contrast', 'original_glcm_Correlation', 'original_glcm_DifferenceAverage', 'original_glcm_DifferenceEntropy', 'original_glcm_DifferenceVariance', 'original_glcm_Id', 'original_glcm_Idm', 'original_glcm_Idmn', 'original_glcm_Idn', 'original_glcm_Imc1', 'original_glcm_Imc2', 'original_glcm_InverseVariance', 'original_glcm_JointAverage', 'original_glcm_JointEnergy', 'original_glcm_JointEntropy', 'original_glcm_MCC', 'original_glcm_MaximumProbability', 'original_glcm_SumAverage', 'original_glcm_SumEntropy', 'original_glcm_SumSquares', 'original_glrlm_GrayLevelNonUniformity', 'original_glrlm_GrayLevelNonUniformityNormalized', 'original_glrlm_GrayLevelVariance', 'original_glrlm_HighGrayLevelRunEmphasis', 'original_glrlm_LongRunEmphasis', 'original_glrlm_LongRunHighGrayLevelEmphasis', 'original_glrlm_LongRunLowGrayLevelEmphasis', 'original_glrlm_LowGrayLevelRunEmphasis', 'original_glrlm_RunEntropy', 'original_glrlm_RunLengthNonUniformity', 'original_glrlm_RunLengthNonUniformityNormalized', 'original_glrlm_RunPercentage', 'original_glrlm_RunVariance', 'original_glrlm_ShortRunEmphasis', 'original_glrlm_ShortRunHighGrayLevelEmphasis', 'original_glrlm_ShortRunLowGrayLevelEmphasis', 'original_glszm_GrayLevelNonUniformity', 'original_glszm_GrayLevelNonUniformityNormalized', 'original_glszm_GrayLevelVariance', 'original_glszm_HighGrayLevelZoneEmphasis', 'original_glszm_LargeAreaEmphasis', 'original_glszm_LargeAreaHighGrayLevelEmphasis', 'original_glszm_LargeAreaLowGrayLevelEmphasis', 'original_glszm_LowGrayLevelZoneEmphasis', 'original_glszm_SizeZoneNonUniformity', 'original_glszm_SizeZoneNonUniformityNormalized', 'original_glszm_SmallAreaEmphasis', 'original_glszm_SmallAreaHighGrayLevelEmphasis', 'original_glszm_SmallAreaLowGrayLevelEmphasis', 'original_glszm_ZoneEntropy', 'original_glszm_ZonePercentage', 'original_glszm_ZoneVariance', 'original_ngtdm_Busyness', 'original_ngtdm_Coarseness', 'original_ngtdm_Complexity', 'original_ngtdm_Contrast', 'original_ngtdm_Strength', 'original_gldm_DependenceEntropy', 'original_gldm_DependenceNonUniformity', 'original_gldm_DependenceNonUniformityNormalized', 'original_gldm_DependenceVariance', 'original_gldm_GrayLevelNonUniformity', 'original_gldm_GrayLevelVariance', 'original_gldm_HighGrayLevelEmphasis', 'original_gldm_LargeDependenceEmphasis', 'original_gldm_LargeDependenceHighGrayLevelEmphasis', 'original_gldm_LargeDependenceLowGrayLevelEmphasis', 'original_gldm_LowGrayLevelEmphasis', 'original_gldm_SmallDependenceEmphasis', 'original_gldm_SmallDependenceHighGrayLevelEmphasis', 'original_gldm_SmallDependenceLowGrayLevelEmphasis', 'wavelet-LH_firstorder_10Percentile', 'wavelet-LH_firstorder_90Percentile', 'wavelet-LH_firstorder_Energy', 'wavelet-LH_firstorder_Entropy', 'wavelet-LH_firstorder_InterquartileRange', 'wavelet-LH_firstorder_Kurtosis', 'wavelet-LH_firstorder_Maximum', 'wavelet-LH_firstorder_MeanAbsoluteDeviation', 'wavelet-LH_firstorder_Mean', 'wavelet-LH_firstorder_Median', 'wavelet-LH_firstorder_Minimum', 'wavelet-LH_firstorder_Range', 'wavelet-LH_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-LH_firstorder_RootMeanSquared', 'wavelet-LH_firstorder_Skewness', 'wavelet-LH_firstorder_TotalEnergy', 'wavelet-LH_firstorder_Uniformity', 'wavelet-LH_firstorder_Variance', 'wavelet-LH_glcm_Autocorrelation', 'wavelet-LH_glcm_ClusterProminence', 'wavelet-LH_glcm_ClusterShade', 'wavelet-LH_glcm_ClusterTendency', 'wavelet-LH_glcm_Contrast', 'wavelet-LH_glcm_Correlation', 'wavelet-LH_glcm_DifferenceAverage', 'wavelet-LH_glcm_DifferenceEntropy', 'wavelet-LH_glcm_DifferenceVariance', 'wavelet-LH_glcm_Id', 'wavelet-LH_glcm_Idm', 'wavelet-LH_glcm_Idmn', 'wavelet-LH_glcm_Idn', 'wavelet-LH_glcm_Imc1', 'wavelet-LH_glcm_Imc2', 'wavelet-LH_glcm_InverseVariance', 'wavelet-LH_glcm_JointAverage', 'wavelet-LH_glcm_JointEnergy', 'wavelet-LH_glcm_JointEntropy', 'wavelet-LH_glcm_MCC', 'wavelet-LH_glcm_MaximumProbability', 'wavelet-LH_glcm_SumAverage', 'wavelet-LH_glcm_SumEntropy', 'wavelet-LH_glcm_SumSquares', 'wavelet-LH_glrlm_GrayLevelNonUniformity', 'wavelet-LH_glrlm_GrayLevelNonUniformityNormalized', 'wavelet-LH_glrlm_GrayLevelVariance', 'wavelet-LH_glrlm_HighGrayLevelRunEmphasis', 'wavelet-LH_glrlm_LongRunEmphasis', 'wavelet-LH_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-LH_glrlm_LongRunLowGrayLevelEmphasis', 'wavelet-LH_glrlm_LowGrayLevelRunEmphasis', 'wavelet-LH_glrlm_RunEntropy', 'wavelet-LH_glrlm_RunLengthNonUniformity', 'wavelet-LH_glrlm_RunLengthNonUniformityNormalized', 'wavelet-LH_glrlm_RunPercentage', 'wavelet-LH_glrlm_RunVariance', 'wavelet-LH_glrlm_ShortRunEmphasis', 'wavelet-LH_glrlm_ShortRunHighGrayLevelEmphasis', 'wavelet-LH_glrlm_ShortRunLowGrayLevelEmphasis', 'wavelet-LH_glszm_GrayLevelNonUniformity', 'wavelet-LH_glszm_GrayLevelNonUniformityNormalized', 'wavelet-LH_glszm_GrayLevelVariance', 'wavelet-LH_glszm_HighGrayLevelZoneEmphasis', 'wavelet-LH_glszm_LargeAreaEmphasis', 'wavelet-LH_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-LH_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-LH_glszm_LowGrayLevelZoneEmphasis', 'wavelet-LH_glszm_SizeZoneNonUniformity', 'wavelet-LH_glszm_SizeZoneNonUniformityNormalized', 'wavelet-LH_glszm_SmallAreaEmphasis', 'wavelet-LH_glszm_SmallAreaHighGrayLevelEmphasis', 'wavelet-LH_glszm_SmallAreaLowGrayLevelEmphasis', 'wavelet-LH_glszm_ZoneEntropy', 'wavelet-LH_glszm_ZonePercentage', 'wavelet-LH_glszm_ZoneVariance', 'wavelet-LH_ngtdm_Busyness', 'wavelet-LH_ngtdm_Coarseness', 'wavelet-LH_ngtdm_Complexity', 'wavelet-LH_ngtdm_Contrast', 'wavelet-LH_ngtdm_Strength', 'wavelet-LH_gldm_DependenceEntropy', 'wavelet-LH_gldm_DependenceNonUniformity', 'wavelet-LH_gldm_DependenceNonUniformityNormalized', 'wavelet-LH_gldm_DependenceVariance', 'wavelet-LH_gldm_GrayLevelNonUniformity', 'wavelet-LH_gldm_GrayLevelVariance', 'wavelet-LH_gldm_HighGrayLevelEmphasis', 'wavelet-LH_gldm_LargeDependenceEmphasis', 'wavelet-LH_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-LH_gldm_LargeDependenceLowGrayLevelEmphasis', 'wavelet-LH_gldm_LowGrayLevelEmphasis', 'wavelet-LH_gldm_SmallDependenceEmphasis', 'wavelet-LH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LH_gldm_SmallDependenceLowGrayLevelEmphasis', 'wavelet-HL_firstorder_10Percentile', 'wavelet-HL_firstorder_90Percentile', 'wavelet-HL_firstorder_Energy', 'wavelet-HL_firstorder_Entropy', 'wavelet-HL_firstorder_InterquartileRange', 'wavelet-HL_firstorder_Kurtosis', 'wavelet-HL_firstorder_Maximum', 'wavelet-HL_firstorder_MeanAbsoluteDeviation', 'wavelet-HL_firstorder_Mean', 'wavelet-HL_firstorder_Median', 'wavelet-HL_firstorder_Minimum', 'wavelet-HL_firstorder_Range', 'wavelet-HL_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-HL_firstorder_RootMeanSquared', 'wavelet-HL_firstorder_Skewness', 'wavelet-HL_firstorder_TotalEnergy', 'wavelet-HL_firstorder_Uniformity', 'wavelet-HL_firstorder_Variance', 'wavelet-HL_glcm_Autocorrelation', 'wavelet-HL_glcm_ClusterProminence', 'wavelet-HL_glcm_ClusterShade', 'wavelet-HL_glcm_ClusterTendency', 'wavelet-HL_glcm_Contrast', 'wavelet-HL_glcm_Correlation', 'wavelet-HL_glcm_DifferenceAverage', 'wavelet-HL_glcm_DifferenceEntropy', 'wavelet-HL_glcm_DifferenceVariance', 'wavelet-HL_glcm_Id', 'wavelet-HL_glcm_Idm', 'wavelet-HL_glcm_Idmn', 'wavelet-HL_glcm_Idn', 'wavelet-HL_glcm_Imc1', 'wavelet-HL_glcm_Imc2', 'wavelet-HL_glcm_InverseVariance', 'wavelet-HL_glcm_JointAverage', 'wavelet-HL_glcm_JointEnergy', 'wavelet-HL_glcm_JointEntropy', 'wavelet-HL_glcm_MCC', 'wavelet-HL_glcm_MaximumProbability', 'wavelet-HL_glcm_SumAverage', 'wavelet-HL_glcm_SumEntropy', 'wavelet-HL_glcm_SumSquares', 'wavelet-HL_glrlm_GrayLevelNonUniformity', 'wavelet-HL_glrlm_GrayLevelNonUniformityNormalized', 'wavelet-HL_glrlm_GrayLevelVariance', 'wavelet-HL_glrlm_HighGrayLevelRunEmphasis', 'wavelet-HL_glrlm_LongRunEmphasis', 'wavelet-HL_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-HL_glrlm_LongRunLowGrayLevelEmphasis', 'wavelet-HL_glrlm_LowGrayLevelRunEmphasis', 'wavelet-HL_glrlm_RunEntropy', 'wavelet-HL_glrlm_RunLengthNonUniformity', 'wavelet-HL_glrlm_RunLengthNonUniformityNormalized', 'wavelet-HL_glrlm_RunPercentage', 'wavelet-HL_glrlm_RunVariance', 'wavelet-HL_glrlm_ShortRunEmphasis', 'wavelet-HL_glrlm_ShortRunHighGrayLevelEmphasis', 'wavelet-HL_glrlm_ShortRunLowGrayLevelEmphasis', 'wavelet-HL_glszm_GrayLevelNonUniformity', 'wavelet-HL_glszm_GrayLevelNonUniformityNormalized', 'wavelet-HL_glszm_GrayLevelVariance', 'wavelet-HL_glszm_HighGrayLevelZoneEmphasis', 'wavelet-HL_glszm_LargeAreaEmphasis', 'wavelet-HL_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-HL_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-HL_glszm_LowGrayLevelZoneEmphasis', 'wavelet-HL_glszm_SizeZoneNonUniformity', 'wavelet-HL_glszm_SizeZoneNonUniformityNormalized', 'wavelet-HL_glszm_SmallAreaEmphasis', 'wavelet-HL_glszm_SmallAreaHighGrayLevelEmphasis', 'wavelet-HL_glszm_SmallAreaLowGrayLevelEmphasis', 'wavelet-HL_glszm_ZoneEntropy', 'wavelet-HL_glszm_ZonePercentage', 'wavelet-HL_glszm_ZoneVariance', 'wavelet-HL_ngtdm_Busyness', 'wavelet-HL_ngtdm_Coarseness', 'wavelet-HL_ngtdm_Complexity', 'wavelet-HL_ngtdm_Contrast', 'wavelet-HL_ngtdm_Strength', 'wavelet-HL_gldm_DependenceEntropy', 'wavelet-HL_gldm_DependenceNonUniformity', 'wavelet-HL_gldm_DependenceNonUniformityNormalized', 'wavelet-HL_gldm_DependenceVariance', 'wavelet-HL_gldm_GrayLevelNonUniformity', 'wavelet-HL_gldm_GrayLevelVariance', 'wavelet-HL_gldm_HighGrayLevelEmphasis', 'wavelet-HL_gldm_LargeDependenceEmphasis', 'wavelet-HL_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HL_gldm_LargeDependenceLowGrayLevelEmphasis', 'wavelet-HL_gldm_LowGrayLevelEmphasis', 'wavelet-HL_gldm_SmallDependenceEmphasis', 'wavelet-HL_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-HL_gldm_SmallDependenceLowGrayLevelEmphasis', 'wavelet-HH_firstorder_10Percentile', 'wavelet-HH_firstorder_90Percentile', 'wavelet-HH_firstorder_Energy', 'wavelet-HH_firstorder_Entropy', 'wavelet-HH_firstorder_InterquartileRange', 'wavelet-HH_firstorder_Kurtosis', 'wavelet-HH_firstorder_Maximum', 'wavelet-HH_firstorder_MeanAbsoluteDeviation', 'wavelet-HH_firstorder_Mean', 'wavelet-HH_firstorder_Median', 'wavelet-HH_firstorder_Minimum', 'wavelet-HH_firstorder_Range', 'wavelet-HH_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-HH_firstorder_RootMeanSquared', 'wavelet-HH_firstorder_Skewness', 'wavelet-HH_firstorder_TotalEnergy', 'wavelet-HH_firstorder_Uniformity', 'wavelet-HH_firstorder_Variance', 'wavelet-HH_glcm_Autocorrelation', 'wavelet-HH_glcm_ClusterProminence', 'wavelet-HH_glcm_ClusterShade', 'wavelet-HH_glcm_ClusterTendency', 'wavelet-HH_glcm_Contrast', 'wavelet-HH_glcm_Correlation', 'wavelet-HH_glcm_DifferenceAverage', 'wavelet-HH_glcm_DifferenceEntropy', 'wavelet-HH_glcm_DifferenceVariance', 'wavelet-HH_glcm_Id', 'wavelet-HH_glcm_Idm', 'wavelet-HH_glcm_Idmn', 'wavelet-HH_glcm_Idn', 'wavelet-HH_glcm_Imc1', 'wavelet-HH_glcm_Imc2', 'wavelet-HH_glcm_InverseVariance', 'wavelet-HH_glcm_JointAverage', 'wavelet-HH_glcm_JointEnergy', 'wavelet-HH_glcm_JointEntropy', 'wavelet-HH_glcm_MCC', 'wavelet-HH_glcm_MaximumProbability', 'wavelet-HH_glcm_SumAverage', 'wavelet-HH_glcm_SumEntropy', 'wavelet-HH_glcm_SumSquares', 'wavelet-HH_glrlm_GrayLevelNonUniformity', 'wavelet-HH_glrlm_GrayLevelNonUniformityNormalized', 'wavelet-HH_glrlm_GrayLevelVariance', 'wavelet-HH_glrlm_HighGrayLevelRunEmphasis', 'wavelet-HH_glrlm_LongRunEmphasis', 'wavelet-HH_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-HH_glrlm_LongRunLowGrayLevelEmphasis', 'wavelet-HH_glrlm_LowGrayLevelRunEmphasis', 'wavelet-HH_glrlm_RunEntropy', 'wavelet-HH_glrlm_RunLengthNonUniformity', 'wavelet-HH_glrlm_RunLengthNonUniformityNormalized', 'wavelet-HH_glrlm_RunPercentage', 'wavelet-HH_glrlm_RunVariance', 'wavelet-HH_glrlm_ShortRunEmphasis', 'wavelet-HH_glrlm_ShortRunHighGrayLevelEmphasis', 'wavelet-HH_glrlm_ShortRunLowGrayLevelEmphasis', 'wavelet-HH_glszm_GrayLevelNonUniformity', 'wavelet-HH_glszm_GrayLevelNonUniformityNormalized', 'wavelet-HH_glszm_GrayLevelVariance', 'wavelet-HH_glszm_HighGrayLevelZoneEmphasis', 'wavelet-HH_glszm_LargeAreaEmphasis', 'wavelet-HH_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-HH_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-HH_glszm_LowGrayLevelZoneEmphasis', 'wavelet-HH_glszm_SizeZoneNonUniformity', 'wavelet-HH_glszm_SizeZoneNonUniformityNormalized', 'wavelet-HH_glszm_SmallAreaEmphasis', 'wavelet-HH_glszm_SmallAreaHighGrayLevelEmphasis', 'wavelet-HH_glszm_SmallAreaLowGrayLevelEmphasis', 'wavelet-HH_glszm_ZoneEntropy', 'wavelet-HH_glszm_ZonePercentage', 'wavelet-HH_glszm_ZoneVariance', 'wavelet-HH_ngtdm_Busyness', 'wavelet-HH_ngtdm_Coarseness', 'wavelet-HH_ngtdm_Complexity', 'wavelet-HH_ngtdm_Contrast', 'wavelet-HH_ngtdm_Strength', 'wavelet-HH_gldm_DependenceEntropy', 'wavelet-HH_gldm_DependenceNonUniformity', 'wavelet-HH_gldm_DependenceNonUniformityNormalized', 'wavelet-HH_gldm_DependenceVariance', 'wavelet-HH_gldm_GrayLevelNonUniformity', 'wavelet-HH_gldm_GrayLevelVariance', 'wavelet-HH_gldm_HighGrayLevelEmphasis', 'wavelet-HH_gldm_LargeDependenceEmphasis', 'wavelet-HH_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-HH_gldm_LargeDependenceLowGrayLevelEmphasis', 'wavelet-HH_gldm_LowGrayLevelEmphasis', 'wavelet-HH_gldm_SmallDependenceEmphasis', 'wavelet-HH_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-HH_gldm_SmallDependenceLowGrayLevelEmphasis', 'wavelet-LL_firstorder_10Percentile', 'wavelet-LL_firstorder_90Percentile', 'wavelet-LL_firstorder_Energy', 'wavelet-LL_firstorder_Entropy', 'wavelet-LL_firstorder_InterquartileRange', 'wavelet-LL_firstorder_Kurtosis', 'wavelet-LL_firstorder_Maximum', 'wavelet-LL_firstorder_MeanAbsoluteDeviation', 'wavelet-LL_firstorder_Mean', 'wavelet-LL_firstorder_Median', 'wavelet-LL_firstorder_Minimum', 'wavelet-LL_firstorder_Range', 'wavelet-LL_firstorder_RobustMeanAbsoluteDeviation', 'wavelet-LL_firstorder_RootMeanSquared', 'wavelet-LL_firstorder_Skewness', 'wavelet-LL_firstorder_TotalEnergy', 'wavelet-LL_firstorder_Uniformity', 'wavelet-LL_firstorder_Variance', 'wavelet-LL_glcm_Autocorrelation', 'wavelet-LL_glcm_ClusterProminence', 'wavelet-LL_glcm_ClusterShade', 'wavelet-LL_glcm_ClusterTendency', 'wavelet-LL_glcm_Contrast', 'wavelet-LL_glcm_Correlation', 'wavelet-LL_glcm_DifferenceAverage', 'wavelet-LL_glcm_DifferenceEntropy', 'wavelet-LL_glcm_DifferenceVariance', 'wavelet-LL_glcm_Id', 'wavelet-LL_glcm_Idm', 'wavelet-LL_glcm_Idmn', 'wavelet-LL_glcm_Idn', 'wavelet-LL_glcm_Imc1', 'wavelet-LL_glcm_Imc2', 'wavelet-LL_glcm_InverseVariance', 'wavelet-LL_glcm_JointAverage', 'wavelet-LL_glcm_JointEnergy', 'wavelet-LL_glcm_JointEntropy', 'wavelet-LL_glcm_MCC', 'wavelet-LL_glcm_MaximumProbability', 'wavelet-LL_glcm_SumAverage', 'wavelet-LL_glcm_SumEntropy', 'wavelet-LL_glcm_SumSquares', 'wavelet-LL_glrlm_GrayLevelNonUniformity', 'wavelet-LL_glrlm_GrayLevelNonUniformityNormalized', 'wavelet-LL_glrlm_GrayLevelVariance', 'wavelet-LL_glrlm_HighGrayLevelRunEmphasis', 'wavelet-LL_glrlm_LongRunEmphasis', 'wavelet-LL_glrlm_LongRunHighGrayLevelEmphasis', 'wavelet-LL_glrlm_LongRunLowGrayLevelEmphasis', 'wavelet-LL_glrlm_LowGrayLevelRunEmphasis', 'wavelet-LL_glrlm_RunEntropy', 'wavelet-LL_glrlm_RunLengthNonUniformity', 'wavelet-LL_glrlm_RunLengthNonUniformityNormalized', 'wavelet-LL_glrlm_RunPercentage', 'wavelet-LL_glrlm_RunVariance', 'wavelet-LL_glrlm_ShortRunEmphasis', 'wavelet-LL_glrlm_ShortRunHighGrayLevelEmphasis', 'wavelet-LL_glrlm_ShortRunLowGrayLevelEmphasis', 'wavelet-LL_glszm_GrayLevelNonUniformity', 'wavelet-LL_glszm_GrayLevelNonUniformityNormalized', 'wavelet-LL_glszm_GrayLevelVariance', 'wavelet-LL_glszm_HighGrayLevelZoneEmphasis', 'wavelet-LL_glszm_LargeAreaEmphasis', 'wavelet-LL_glszm_LargeAreaHighGrayLevelEmphasis', 'wavelet-LL_glszm_LargeAreaLowGrayLevelEmphasis', 'wavelet-LL_glszm_LowGrayLevelZoneEmphasis', 'wavelet-LL_glszm_SizeZoneNonUniformity', 'wavelet-LL_glszm_SizeZoneNonUniformityNormalized', 'wavelet-LL_glszm_SmallAreaEmphasis', 'wavelet-LL_glszm_SmallAreaHighGrayLevelEmphasis', 'wavelet-LL_glszm_SmallAreaLowGrayLevelEmphasis', 'wavelet-LL_glszm_ZoneEntropy', 'wavelet-LL_glszm_ZonePercentage', 'wavelet-LL_glszm_ZoneVariance', 'wavelet-LL_ngtdm_Busyness', 'wavelet-LL_ngtdm_Coarseness', 'wavelet-LL_ngtdm_Complexity', 'wavelet-LL_ngtdm_Contrast', 'wavelet-LL_ngtdm_Strength', 'wavelet-LL_gldm_DependenceEntropy', 'wavelet-LL_gldm_DependenceNonUniformity', 'wavelet-LL_gldm_DependenceNonUniformityNormalized', 'wavelet-LL_gldm_DependenceVariance', 'wavelet-LL_gldm_GrayLevelNonUniformity', 'wavelet-LL_gldm_GrayLevelVariance', 'wavelet-LL_gldm_HighGrayLevelEmphasis', 'wavelet-LL_gldm_LargeDependenceEmphasis', 'wavelet-LL_gldm_LargeDependenceHighGrayLevelEmphasis', 'wavelet-LL_gldm_LargeDependenceLowGrayLevelEmphasis', 'wavelet-LL_gldm_LowGrayLevelEmphasis', 'wavelet-LL_gldm_SmallDependenceEmphasis', 'wavelet-LL_gldm_SmallDependenceHighGrayLevelEmphasis', 'wavelet-LL_gldm_SmallDependenceLowGrayLevelEmphasis']
        
        fenaemes = ["image"]+lbpnames+radnames+["label"]
        
        df = pd.DataFrame(dfs, columns=fenaemes)
        po = os.path.join(dat["pathout"], sset+".csv")
        print("po",po)
        df.to_csv(po, index=False)


das = [
    "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/500/20230825140850",
    #"/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/400/20230826121633",
    # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/300/20230826124152",
    # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/200/20230826124856",
]
for mp in das:
    for i in range(1,2):
        dat = { "mp":mp,
                "pathin":mp+"/"+str(i),
                "pathmk":mp+"/"+str(i)+"/mask",
                "pathout":mp+"/"+str(i)+"/features",
                }
        feature_extraction(dat)


# mp = "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/400/20230826121633"
# dat = { "mp":mp,
#         "pathin":mp+"/1",
#         "pathmk":mp+"/mask",
#         "pathout":mp+"/features/main.csv",
#         }
# feature_extraction(dat)


# mp = "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/300/20230826124152"
# dat = { "mp":mp,
#         "pathin":mp+"/1",
#         "pathmk":mp+"/mask",
#         "pathout":mp+"/features/main.csv",
#         }
# feature_extraction(dat)


# mp = "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/200/20230826124856"
# dat = { "mp":mp,
#         "pathin":mp+"/1",
#         "pathmk":mp+"/mask",
#         "pathout":mp+"/features/main.csv",
#         }
# feature_extraction(dat)

