
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

from sklearn.feature_extraction import image

#import logging
#logging.getLogger('radiomics').setLevel(logging.CRITICAL + 1)  

""" 
@staticmethod
def getLabels(file):
    df_info = pd.read_csv(os.path.join(file))
    labels = [row["idseg"] for index, row in df_info.iterrows()]
    return labels  """




class Features:
    #fecount = 0  #assign as integer type

    #manager = Manager()
    #fecount = 0
    @staticmethod
    def execute(path, tile):
        imagefi = os.path.join(path, "preprocessing.tiff")
        maskfi = os.path.join(path, str(tile), "roids.nrrd")
        featfi = os.path.join(path, str(tile), "features.csv")
        df_info = pd.read_csv(os.path.join(path, str(tile), "info.csv"))

        start_time = time.time()
        flbp = LBP(imagefi, maskfi, df_info)
        lbp_vect, lbp_names = flbp.execute_mp()
        print("--- %s lbp seconds ---" % (time.time() - start_time))

        start_time = time.time()
        radobj = Radiomics(imagefi, maskfi, df_info)
        rad_vect, rad_names = radobj.execute_mp()
        print("--- %s lbp seconds ---" % (time.time() - start_time))

        df_lbp = pd.DataFrame(lbp_vect, columns=lbp_names)
        df_rad = pd.DataFrame(rad_vect, columns=rad_names)
        #df_fe = pd.concat([df_lbp, df_rad], axis=1)
        df_fe = df_lbp.merge(df_rad, on=["image","idseg"])

        #df_fe.columns = lbp_names+list(rad_names)
        df_fe.to_csv(featfi, index=False)


    @staticmethod
    def execute_images_sp(pathgrayimg, pathmaskimg, pathout, fileout):
        arg = []
        test, train = [], []
        for setdata, path in pathgrayimg.items():
            for imageName in os.listdir(path):
        
                base = os.path.basename(imageName)
                base = os.path.splitext(base)
                imgon = str(base[0])

                imagefi = os.path.join(path, imgon+".tiff")
                maskfi = os.path.join(pathmaskimg, imgon+".nrrd")
                #print("imagefi", imagefi)

                arg.append({"imagefi":imagefi, "maskfi":maskfi})
               
                if setdata == "test":
                    test.append(imageName)
                elif setdata == "train":
                    train.append(imageName)



        
        pool = Pool( processes = int(multiprocessing.cpu_count()/2))
        rs = pool.map(Features.execute_images_process, arg)
        pool.close()
        #print("rs", rs)
        dfs = []
        for row in rs:
            dfs.append(row)

        df_fe = pd.concat(dfs)
        df_fe.columns = dfs[0].columns
        df_fe.to_csv(os.path.join(pathout, fileout), index=False)

    def execute_images_process(arg):
        imagefi = arg["imagefi"]
        maskfi = arg["maskfi"]

        #global Features.fecount

        #Features.fecount.acquire()
        #Features.fecount += 1
        #Features.fecount.release()

        #Features.fecount += 1
               
        start_time = time.time()        
        lbp_vect, lbp_names = LBP.execute_sp(imagefi, maskfi)
        rad_vect, rad_names = Radiomics.execute_sp(imagefi, maskfi)
        
        print("lbp_vect, lbp_names, rad_vect, rad_names", len(lbp_vect), len(lbp_names), len(rad_vect), len(rad_names))
        
        df_lbp = pd.DataFrame(lbp_vect, columns=lbp_names)

        df_rad = pd.DataFrame(rad_vect, columns=rad_names)
        df_lbp.columns = lbp_names
        df_rad.columns = rad_names

        df_rest = df_lbp.merge(df_rad, on=["image","idseg"])
        #print("df_rest", df_rest)

        #print("--- %s lbp seconds ---" % (time.time() - start_time))

        print("imagefi", imagefi, (time.time() - start_time))
        
        return df_rest


    @staticmethod
    def execute_images_mp(pathgrayimg, pathmaskimg, pathout, fileout):
        dfs = []
        #arg = []
        test, train = [], []
        count = 1
        for setdata, path in pathgrayimg.items():
            for imageName in os.listdir(path):
        
                base = os.path.basename(imageName)
                base = os.path.splitext(base)
                imgon = str(base[0])

                imagefi = os.path.join(path, imgon+".tiff")
                maskfi = os.path.join(pathmaskimg, imgon+".nrrd")
                

                #arg.append({"imagefi":imagefi, "maskfi":maskfi})
               
                if setdata == "test":
                    test.append(imageName)
                elif setdata == "train":
                    train.append(imageName)


                start_time = time.time()

                lbp_o = LBP(imagefi, maskfi)
                lbp_vect, lbp_names = lbp_o.execute_mp()
                #print("lbp_vect, lbp_names", lbp_vect, lbp_names)
                df_lbp = pd.DataFrame(lbp_vect, columns=lbp_names)
                df_lbp.columns = lbp_names
                del lbp_o

                rad_o = Radiomics(imagefi, maskfi)
                rad_vect, rad_names = rad_o.execute_mp()
                df_rad = pd.DataFrame(rad_vect, columns=rad_names)
                df_rad.columns = rad_names
                del rad_o
                
                df_rest = df_lbp.merge(df_rad, on=["image","idseg"])
                #df_rest.columns = lbp_names+rad_names
                df_rest.to_csv(os.path.join(pathout, imgon+".csv"), index=False)
                #dfs.append(df_rest)
                del df_rest
                del df_lbp
                del df_rad
                print("imagefi", imagefi, (time.time() - start_time), count)
                count += 1

        """
        df_fe = pd.concat(dfs)
        df_fe.columns = dfs[0].columns
        df_fe.to_csv(os.path.join(pathout, fileout), index=False) 
        """
        #print("rest", rest)


class LBP:
    
    def __init__(self, imagefi, maskfi, dflabels):
        self.inputImage = cv2.imread(imagefi, cv2.IMREAD_GRAYSCALE)
        image_mask = sitk.ReadImage(maskfi)
        self.lbpmask = sitk.GetArrayFromImage(image_mask)


        self.nameimage = os.path.basename(imagefi)
        #base = os.path.splitext(base)
        #imgon = str(base[0])

        """ lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(image_mask)
        self.labels = lsif.GetLabels() """

        self.labels = [row["idseg"] for index, row in dflabels.iterrows()]
        #print("self.labels lbp", self.labels)

        manager = Manager()
        #self.rads = manager.list( [1,2,3,4,5,6,7,8,9,10] )
        self.rads = manager.list( [10] )
        self.lbp = manager.dict( {i:[] for i in self.rads } )
        #self.maskloc = manager.dict( {label:[] for label in self.labels} )
        self.xnBins = manager.dict( {i:[] for i in self.rads } )

        #start_time = time.time()
        arg = [{"radius":radius} for radius in self.rads] 
        pool = Pool( processes = int(multiprocessing.cpu_count()/2) )
        pool.map(self.process_lbp_radius, arg)
        pool.close()
        #print("--- %s lbp radious seconds ---" % (time.time() - start_time))
    
    def __del__(self):
        #del self.maskloc
        del self.xnBins
        del self.inputImage
        del self.lbp
        del self.lbpmask
        del self.rads
        del self.labels

    def process_lbp_radius(self, arg):
        radius = arg["radius"]
        nPoints = 8 * radius
        self.lbp[radius] = local_binary_pattern(self.inputImage, nPoints, radius, method='uniform')
        self.xnBins[radius] = int(self.lbp[radius].max() + 1)
        
        """
        for label in self.labels:
            ppp = np.where(self.lbpmask == label)
            #print("ppp",ppp)
            #self.maskloc[radius][label][0], self.maskloc[radius][label][1] = ppp
            self.maskloc[label] = np.where(self.lbpmask == label)
        print("self.maskloc", self.maskloc[label])
        """

    def execute_mp(self):

        arg = [{"nameimage":self.nameimage, "label":lab} for lab in self.labels]
        
        pool = Pool( processes = int(multiprocessing.cpu_count()/2) )
        rs = pool.map(self.process_lbp_mp, arg)
        pool.close()
        #vects = [ [] for i in range(len(self.labels))]
        vects = []
        names = []
        for ss in rs:
            vects.append(ss[1])

            names = ss[2]
        del rs
        return vects, names


    def process_lbp_mp(self, arg):
        #rad = [1,2,3,4,5,6,7,8,9,10]
        #rad = [10]
        nameimage = arg["nameimage"]
        label = arg["label"]
        vect = []
        vect_names = []
        for radius in self.rads:
            #nPoints = 8 * radius
            #lbpi = self.lbpimage[radius]
            #lbpimage = local_binary_pattern(lbpi, nPoints, radius, method='uniform')
            xnBins = self.xnBins[radius]
            #print("--- %s xnBins time ---" % xnBins )
            
            lbpv = self.lbp[radius][np.where(self.lbpmask == label)]
            #lbpv = self.lbp[radius][self.maskloc[label]]
            
            #print("self.maskloc[radius][label]", lbpv)
            #lbpx[np.where(mask == lab)]
            histogram, _ = np.histogram(lbpv, bins=xnBins, range=(0, xnBins))
            histogram = histogram.tolist()
            #print("histogram", histogram)
            vect += histogram
            vect_names += ["LBP_r"+str(radius)+"_"+str(i+1) for i in range(len(histogram))]
            del lbpv
        vect = [nameimage]+[label]+vect

        #if len(self.names) == 0:
        #    #for a in vect_names:
        #    self.names = ["image"]+["idseg"]+vect_names
        #    #    self.names.append(a)
        
        #print("self.names", self.names)

        return label, vect, ["image"]+["idseg"]+vect_names




    """ 
    def process_lbp_mp(self, arg):
        nameimage = arg["nameimage"]
        lab     = arg["label"]
        xnBins  = arg["xnBins"]

        #lbpTile = self.lbpimage[np.where(self.lbpmask == lab) ]
        #xnBins2 = int(lbpTile.max() + 1)
        #print("inst xxx", lab, xnBins, xnBins2)

        histogram, _ = np.histogram(self.lbpimage[np.where(self.lbpmask == lab) ], bins=xnBins, range=(0, xnBins))
        histogram = histogram.tolist()
        self.fecolsize  = len(histogram)
        self.vectors[lab] = [nameimage]+[lab]+histogram
    """



    @staticmethod
    def execute_sp(imagefi, maskfi, labels):
        inputImage = cv2.imread(imagefi, cv2.IMREAD_GRAYSCALE)
        image_mask = sitk.ReadImage(maskfi)
        mask = sitk.GetArrayFromImage(image_mask)

        base = os.path.basename(imagefi)
        #base = os.path.splitext(base)
        #imgon = str(base[0])      

        """
        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(image_mask)
        labels = lsif.GetLabels() """

        #labels = [row["idseg"] for index, row in dflabels.iterrows()]

        #rads = [1,2,3,4,5,6,7,8,9,10]
        rads = [10]
        #rads = [10]

        
        lbp = {i:{l:[] for l in labels} for i in rads}
        xnBins = {i:[] for i in rads}
        for radius in rads:
            nPoints = 8 * radius
            lbpx = local_binary_pattern(inputImage, nPoints, radius, method='uniform')
            xnBins[radius] = int(lbpx.max() + 1)
            for lab in labels:
                lbp[radius][lab] = lbpx[np.where(mask == lab)]
            del lbpx

        vect = [ ]
        vect_names = []        
        for lab in labels:
            names = []
            vect_aux = []
            for radius in rads:
                histogram, _ = np.histogram(lbp[radius][lab], bins=xnBins[radius], range=(0, xnBins[radius]))
                aux = histogram.tolist()
                vect_aux += aux
                names += ["LPB_r"+str(radius)+"_"+str(i+1) for i in range(len(aux))]
            vect.append([base]+[lab]+vect_aux)
            vect_names = names

        #vect = [base]+[lab]+vect
        vect_names = ["image"]+["idseg"]+vect_names

        del inputImage
        del image_mask
        del mask
        del lbp
        del xnBins
        
        return vect, vect_names



class Radiomics:
    def __init__(self, imagefi, maskfi, dflabels):
        self.radimage = sitk.ReadImage(imagefi, sitk.sitkFloat32)
        self.radmask = sitk.ReadImage(maskfi)

        self.nameimage = os.path.basename(imagefi)

        """ lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(self.radmask)
        self.labels = lsif.GetLabels() """

        self.labels = [row["idseg"] for index, row in dflabels.iterrows()]

        manager = Manager()
        #self.vectors = manager.dict({i:[] for i in self.labels})
        self.names = manager.list([ ])

    def __del__(self):
        del self.radimage
        del self.radmask
        del self.labels
        #del self.vectors
        del self.names

    def execute_mp(self):
        arg = [{"nameimage":self.nameimage,"label":int(lab)} for lab in self.labels]
        #print("arg", arg)
        pool = Pool( processes = int(multiprocessing.cpu_count()/2) )
        rs = pool.map(self.process_pyradiomics_mp, arg)
        pool.close()

        vects = []
        for row in rs:
            vects.append(row)

        """ 
        vects = [ [] for i in range(len(self.vectors))]
        for k, v in self.vectors.items():
            #print("radsise", k, len(v))
            vects[k-1] = v """

        return vects, ["image"]+["idseg"]+list(self.names)

    def process_pyradiomics_mp(self, arg):
        nameimage = arg["nameimage"]
        label = arg["label"]
        ve, na = Radiomics.pyradiomics(self.radimage, self.radmask, label)
        #self.vectors[label] = [nameimage]+[label]+ve
        if len(self.names) == 0:
            for a in na:
                self.names.append(a)
            #self.names = na
        #print("namexs", na)
        return [nameimage]+[label]+ve
      
    @staticmethod
    def pyradiomics(image, mask, label):

        settings = {}
        #settings['binWidth'] = 25
        #settings['verbose'] = False
        #settings['distances']  = [1, 2, 5, 10]

        #settings = {'minimumROIDimensions': 2, 'minimumROISize': None, 'normalize': False, 'normalizeScale': 1, 'removeOutliers': None, 'resampledPixelSpacing': None, 'interpolator': 'sitkBSpline', 'preCrop': False, 'padDistance': 5, 'distances': [1], 'force2D': False, 'force2Ddimension': 0, 'resegmentRange': None, 'label': 1}
        extractor = RadiomicsFeatureExtractor(**settings)
        #extractor = RadiomicsFeatureExtractor(verbose=False)
        
         
        """ extractor.enableImageTypes(
            #Original={},
            #Wavelet={},
            #$LoG={'sigma':[1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6.0]},
            #LoG={'sigma':[1, 1.5, 2, 2.5, 3]},
            #LBP2D={},
        ) """

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
        names = []
        for k,v in result.items():
            if k.startswith('original') or k.startswith('wavelet') or k.startswith('log') or k.startswith('lbp') or k.startswith('logarithm')  or k.startswith('exponential') or k.startswith('squareroot') or k.startswith('gradient'): 
                fe_values.append("{:.8f}".format(v))
                names.append(k)
        #print("fe_values",fe_values)
        return fe_values, names


    @staticmethod
    def execute_sp(imagefi, maskfi, labels):
        radimage = sitk.ReadImage(imagefi, sitk.sitkFloat32)
        radmask = sitk.ReadImage(maskfi)

        base = os.path.basename(imagefi)
        #base = os.path.splitext(base)
        #imgon = str(base[0])

        """ 
        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(radmask)
        labels = lsif.GetLabels() """
        #labels = [row["idseg"] for index, row in dflabels.iterrows()]

        features = []
        columns = []
        for lab in labels:
            fe, na = Radiomics.pyradiomics(radimage, radmask, lab)
            features.append([base]+[lab]+fe)
            columns = ["image"]+["idseg"]+na
        
        del radimage
        del radmask

        return features, columns




class FeaturesX:
    def __init__(self):
        pass

    @staticmethod
    def execute_images_sp(pathgrayimg, pathmaskimg, pathout, fileout):
        dfs = []

        df_info = pd.read_csv(os.path.join(pathmaskimg, "info.csv"))

        arg = []
        #count = 1
        for setdata, path in pathgrayimg.items():
            for imageName in os.listdir(path):
                df_filter = df_info[(df_info.image == imageName)]
        
                base = os.path.basename(imageName)
                base = os.path.splitext(base)
                imagename = str(base[0])

                imagefi = os.path.join(path, imagename+".tiff")
                maskfi = os.path.join(pathmaskimg, imagename+".nrrd")
                #print("imagefi", imagefi)

                labels = [row["idseg"] for index, row in df_filter.iterrows()]
                rowarg =    {
                                "imagefi":imagefi,
                                "maskfi":maskfi,
                                "labels":labels,
                                "setdata":setdata,
                                "imagename":imagename,
                                "pathout":pathout,
                                
                            }
                #start_time = time.time()
                if os.path.exists(os.path.join(pathout, imagename+".csv")):
                    continue

                arg.append(rowarg)
                #rs = FeaturesX.execute_images_process(rowarg)
                #dfs.append(rs)
                #print("imagefi", base, (time.time() - start_time), count)
                #count += 1

        
        pool = Pool( processes = int(multiprocessing.cpu_count()/2))
        rs = pool.map(FeaturesX.execute_images_process, arg)
        pool.close()
        for row in rs:
            dfs.append(row)

        df_fe = pd.concat(dfs)
        df_fe.columns = dfs[0].columns
        df_fe.to_csv(os.path.join(pathout, fileout), index=False)

    @staticmethod
    def execute_images_process(arg):
        start_time = time.time()

        imagefi = arg["imagefi"]
        maskfi = arg["maskfi"]
        labels = arg["labels"]
        pathout = arg["pathout"]
        imagename = arg["imagename"]
        
        #print("dflabels", dflabels)


        #lbp_o = LBP(imagefi, maskfi, dflabels)
        #lbp_vect, lbp_names = lbp_o.execute_mp()
        lbp_vect, lbp_names = LBP.execute_sp(imagefi, maskfi, labels)

        #rad_o = Radiomics(imagefi, maskfi, dflabels)
        #rad_vect, rad_names = rad_o.execute_mp()
        rad_vect, rad_names = Radiomics.execute_sp(imagefi, maskfi, labels)
        
        #del lbp_o
        #del rad_o
        
        print("lbp_vect, lbp_names, rad_vect, rad_names", len(lbp_vect), len(lbp_names), len(rad_vect), len(rad_names))
        
        df_lbp = pd.DataFrame(lbp_vect, columns=lbp_names)

        df_rad = pd.DataFrame(rad_vect, columns=rad_names)
        df_lbp.columns = lbp_names
        df_rad.columns = rad_names

        df_rest = df_lbp.merge(df_rad, on=["image","idseg"])
        
        df_rest.to_csv(os.path.join(pathout, imagename+".csv"), index=False)

        del df_lbp
        del df_rad
        print("imagefi", imagefi, (time.time() - start_time))        

        return df_rest

    """ 
    @staticmethod
    def lbp(imagefi, maskfi, dflabels):
        inputImage = cv2.imread(imagefi, cv2.IMREAD_GRAYSCALE)
        image_mask = sitk.ReadImage(maskfi)
        mask = sitk.GetArrayFromImage(image_mask)
        imagename = os.path.basename(imagefi)

        labels = [row["idseg"] for index, row in dflabels.iterrows()]

        #rads = [1,2,3,4,5,6,7,8,9,10]
        rads = [10]
       
        lbp = {i:{l:[] for l in labels} for i in rads}
        xnBins = {i:[] for i in rads}
        for radius in rads:
            nPoints = 8 * radius
            lbpx = local_binary_pattern(inputImage, nPoints, radius, method='uniform')
            xnBins[radius] = int(lbpx.max() + 1)
            for lab in labels:
                lbp[radius][lab] = lbpx[np.where(mask == lab)]
            del lbpx

        vect = [ ]
        vect_names = []        
        for lab in labels:
            names = []
            vect_aux = []
            for radius in rads:
                histogram, _ = np.histogram(lbp[radius][lab], bins=xnBins[radius], range=(0, xnBins[radius]))
                aux = histogram.tolist()
                vect_aux += aux
                names += ["LPB_r"+str(radius)+"_"+str(i+1) for i in range(len(aux))]
            vect.append([imagename]+[lab]+vect_aux)
            vect_names = names

        #vect = [base]+[lab]+vect
        vect_names = ["image"]+["idseg"]+vect_names

        del inputImage
        del image_mask
        del mask
        del lbp
        del xnBins
        
        return vect, vect_names


    @staticmethod
    def radiomics(imagefi, maskfi, dflabels):
        radimage = sitk.ReadImage(imagefi, sitk.sitkFloat32)
        radmask = sitk.ReadImage(maskfi)

        labels = [row["idseg"] for index, row in dflabels.iterrows()]

        imagename = os.path.basename(imagefi)
        
        features = []
        columns = []
        for lab in labels:
            fe, na = Radiomics.pyradiomics(radimage, radmask, lab)
            features.append([imagename]+[lab]+fe)
            columns = ["image"]+["idseg"]+na
        return features, columns """



if __name__ == "__main__":
    pathgrayimg = {
            "test":"/mnt/sda6/software/frameworks/data/lha/dataset_3/grayscale/test",
            "train":"/mnt/sda6/software/frameworks/data/lha/dataset_3/grayscale/train"
            }
    pathmaskimg = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/tiles/erode_radius_30/0.01/500"
    pathout = "/mnt/sda6/software/frameworks/data/lha/dataset_3/build/csv_media/002/features"
    fileout = "features.csv"
    #Features.execute_images_sp(pathgrayimg, pathmaskimg, pathout, fileout)
    #Features.execute_images_mp(pathgrayimg, pathmaskimg, pathout, fileout)

    FeaturesX.execute_images_sp(pathgrayimg, pathmaskimg, pathout, fileout)





