#!/usr/bin/python

import sys
import javabridge
import bioformats
from bioformats import log4j
from bioformats.omexml import OMEXML
import numpy as np
import cv2 as cv2
import os


import multiprocessing
from multiprocessing import cpu_count
from multiprocessing import Pool, Manager, Process, Lock
from multiprocessing.managers import BaseManager

""" 
reader = None
class MyManager(BaseManager):
    pass

def mManager():
    m = MyManager()
    m.start()
    return m 

class ImageReaderVSI(object):
    def __init__(self):
        #self.reader = None
        #imageReader = bioformats.formatreader.make_image_reader_class()
        self.reader = None

    def setReader(self, reader):
        self.reader = reader
    
    def getReader(self):
        return self.reader
"""


def minMax(inputValue, orgMin, orgMax, newMin, newMax):
    '''
    Rescaling:
    min-max normalization function
    '''
    den = 0.00000001 if  orgMax == orgMin else orgMax - orgMin
    return  (((newMax - newMin) * (inputValue - orgMin)) / den) + newMin


def computeResizingFactors(inputPhyX, inputPhyY, inputMag, outputMag):
    '''
    Compute the scale factors along the [x-y] axis
    It uses the input physical pixels size and the original nominal magnification 
    '''
        
    # Physical rescale factor
    phyFactor = inputMag / outputMag
           
    newPhySizeX = phyFactor * inputPhyX
    newPhySizeY = phyFactor * inputPhyY    
    
    return (inputPhyX / newPhySizeX, inputPhyY / newPhySizeY)

   

def computeResolution(physicalX, physicalY, sizeX, sizeY, inputMagnification, outputMagnification):
    '''
    Compute the scale factors along the [x-y] axis
    It uses the input physical pixels size and the original nominal magnification 
    '''
        
    # Physical rescale factor
    phyFactor = inputMagnification / outputMagnification
        
    newPhySizeX = phyFactor * physicalX
    newPhySizeY = phyFactor * physicalY    
    
    return (physicalX / newPhySizeX, physicalY / newPhySizeY)


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def process(arg):
    idi = arg["id"]
    height = arg["height"]
    width = arg["width"]
    xFac = arg["xFac"]
    yFac = arg["yFac"]
    nTilesX = arg["nTilesX"]
    nTilesY = arg["nTilesY"]
    sizeY = arg["sizeY"]
    sizeX = arg["sizeX"]
    tileBeginX = arg["tileBeginX"]
    tileBeginY = arg["tileBeginY"]
    inputFileName = arg["inputFileName"]
    #reader = arg["reader"]
    
    #obj = arg["obj"].getReader()
    
    reader = gettReader(inputFileName)
    #global READD
    print("11111")
    tile = reader.openBytesXYWH(0, tileBeginX, tileBeginY, width, height)
    print("222222")
    tile.shape = (int(height), int(width), 3)
    print("333333")
    tileResized = cv2.resize(tile, None, fx=xFac, fy=yFac, interpolation=cv2.INTER_AREA)
    print("444444")
    #arg["dat"] = tileResized
    arg["dat"][idi] = tileResized
    print("555555555")


def gettReader(inputFileName):
    reader = None

    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True, max_heap_size='8G')
    log4j.basic_config()
    
    imageReader = bioformats.formatreader.make_image_reader_class()
    reader = imageReader()
    reader.setId(inputFileName)

    #javabridge.kill_vm()
    

    return reader

class VSI(object):
    def __init__(self, inputFileName, outputMag=5, nTilesX=20, nTilesY=20):
        '''
        Read a vsi image tile by tile and return a resized RGB TIFF image
        param outMag:  output magnification
        '''

        """ self.reader = None """
        self.aux = None
        self.reader = None
        self.nTilesX = nTilesX
        self.nTilesY = nTilesY

        # starting jvm
        javabridge.start_vm(class_path=bioformats.JARS, run_headless=True, max_heap_size='8G')
        log4j.basic_config()

        ome = OMEXML(bioformats.get_omexml_metadata(path=inputFileName))
        sizeX = ome.image().Pixels.get_SizeX()
        sizeY = ome.image().Pixels.get_SizeY()
            
        nominalMag = float(ome.instrument().Objective.get_NominalMagnification())
        #nominalMag = np.float(ome.instrument().Objective.get_NominalMagnification())

        physicalX = ome.image().Pixels.get_PhysicalSizeX()
        physicalY = ome.image().Pixels.get_PhysicalSizeY()


        
        #imageReader = bioformats.formatreader.make_image_reader_class()
        #reader = imageReader()
        #reader.setId(inputFileName)
        
        """ 
        BaseManager.register('ImageReaderVSI', ImageReaderVSI)
        manager = BaseManager()
        manager.start()
        myreader = manager.ImageReaderVSI()
        myreader.setReader(reader) 
        """

        """
        MyManager.register('ImageReaderVSI', ImageReaderVSI, exposed=['setId', 'getReader'])
        manager = mManager()
        myreader = manager.ImageReaderVSI()
        myreader.setId(inputFileName)
        """

        # Printing some info
        print('Nominal Magnification: ',nominalMag)
        print('Image size: ', sizeX, sizeY)
        print('Physical pixel size in um: ', physicalX, physicalY) # um = micrometers
        
        # Aux variables
        tileBeginX  = 0
        tileBeginY  = 0
        tileCounter = 0
            
        xFac, yFac = computeResizingFactors( physicalX, physicalY,  nominalMag, outputMag);
        print("xFac, yFac", xFac, yFac)
            
        self.aux = []
        jobs = []
        for y in range(0, self.nTilesY):
            # computing begin and height size
            tileBeginY = minMax(y , 0, self.nTilesY, 0, sizeY)
            height = minMax(y + 1 , 0, self.nTilesY, 0, sizeY) - tileBeginY
            for x in range(0, self.nTilesX):
                tileBeginX = minMax(x , 0, self.nTilesX, 0, sizeX)
                width = minMax(x + 1 , 0, self.nTilesX, 0, sizeX) - tileBeginX
                    
                # tile = reader.openBytesXYWH(0, tileBeginX, tileBeginY, width, height)
                # tile.shape = (int(height), int(width), 3)
                    
                #xFac, yFac = computeResolution(physicalX, physicalY, width, height, nominalMag , outMag)
                # resize tile 
                self.aux.append({
                            "id":tileCounter,
                            "width":width,
                            "height":height,
                            "xFac":xFac,
                            "yFac":yFac,
                            "nTilesX":self.nTilesX,
                            "nTilesY":self.nTilesY,
                            "sizeY":sizeY,
                            "sizeX":sizeX,
                            "tileBeginX":tileBeginX,
                            "tileBeginY":tileBeginY,
                            "inputFileName":inputFileName,
                            "reader":None,
                            "dat":None,
                            })
                tileCounter = tileCounter + 1


        """ 
        with Pool(5) as p:
        p.map(process, self.aux)
        #print()
        print("ssdf")
        """

        
    def __del__ (self):
        javabridge.kill_vm()

    #def getReader(self):
    #    return self.reader

    def getAux(self):
        return self.aux

    def getnTilesX(self):
        return self.nTilesX

    def getnTilesY(self):
        return self.nTilesY

    @staticmethod
    def makeimage(obj):
        aux = obj.getAux()
        #rea = obj.getReader()

        manager = Manager()
        dat = manager.list([ [] for i in range(len(aux))])
        for i in range(len(aux)):
            aux[i]["dat"] = dat
            #aux[i]["obj"] = obj

        print("obj.getAux()", aux)

        pool = Pool(processes=cpu_count()-1)
        rs = pool.map(process, aux)
        pool.close()

        hMosaic = []
        vMosaic = []
        tileCounter = 0
        for y in range(0, obj.getnTilesY()):
            # computing begin and height size
            """
            tileBeginY = minMax(y , 0, nTilesY, 0, sizeY)
            height = minMax(y + 1 , 0, nTilesY, 0, sizeY) - tileBeginY
            """
            for x in range(0, obj.getnTilesX()):
                """
                tileBeginX = minMax(x , 0, nTilesX, 0, sizeX)
                width = minMax(x + 1 , 0, nTilesX, 0, sizeX) - tileBeginX
                tile = reader.openBytesXYWH(0, tileBeginX, tileBeginY, width, height)
                tile.shape = (int(height), int(width), 3)
                tileResized = cv2.resize(tile, None, fx=xFac, fy=yFac, interpolation=cv2.INTER_AREA)
                """
 
                tileResized = dat[tileCounter]
                if(x > 0):
                    hMosaic = np.concatenate((hMosaic, tileResized), axis=1)
                else:
                    hMosaic = tileResized

                tileCounter = tileCounter + 1

            if(y > 0):
                vMosaic = np.concatenate((vMosaic, hMosaic), axis=0)
            else:
                vMosaic = hMosaic

            hMosaic = []
            """
            progress = (tileCounter * 100) / (nTilesX * nTilesY)
            print("processing", str(progress) + '%')
            """

            #print("Resize microscope magnification OK")
        return vMosaic



    """
    h = image.shape[0] 
    w = image.shape[1] 
    s = (700.0/w)*100.0
    ndim = (w*s,h*s)
    tumbail = cv2.resize(image, ndim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(outtumbailjpg, tumbail)
    """

    """    
        print("1")
    except:
        print("0")
    """

if __name__ == '__main__':
    ##just for testing
    #inFileName = "/home/oscar/data/biopsy/Dataset 1/B 2009 8854/B 2009 8854 A.vsi"
    #inFileName = "/home/oscar/data/biopsy/B2046-18 B20181107/Image01B2046-18 B.vsi"
    #outFileName = "/home/oscar/image.tiff"

    ar1="ds1/B 2009 8854 A.vsi"
    ar2=3.0
    ar3="image.jpg"

    
#    print('py ', sys.argv[1], sys.argv[2], sys.argv[3])    
    print('py ', ar1, ar2, ar3)

    # arg1 = input file name
    # arg2 = output file name
    # arg3 = magnification factor
 
    try:
        image = readVSI(ar1, float(ar2))
        cv2.imwrite(ar3, image)
        #plt.imshow(image)
        #plt.show()
        print("1")
    except:
        print("0")
        quit()

