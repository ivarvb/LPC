#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2021
# E-mail: ivar@usp.br



import pandas as pd
import cv2 as cv2

from os import stat
import os
#from turtle import st
import numpy as np

import ujson
import random

import SimpleITK as sitk

import collections
#from shapely import geometry
#from  PyRadiomics import *

import cv2
import time

#from px.image.regions import *
#from Util import *

#from scipy.spatial import ConvexHull, convex_hull_plot_2d
#from vx.media.Features import *

class ROI:
   
    @staticmethod
    def execute(path, tilesizear, tilepercentage):
        start_time = time.time()
        """
        ### (1) execute Oscar's algorithm
        """
        #segmentation
        #os.popen("ddddls ")

        cm = "./vx/radpleura/import/algorithms/cc/PleuraSegmentation/build/PleuraDetection {} {}".format(path, 30)
        print("cm", cm)
        sss = os.system(cm)
        print("--- %s segmentation seconds ---" % (time.time() - start_time))

        
        """
        ### (2) validate conex pixels for each region
        """

        imgpathoriginal = os.path.join(path, "original.tiff")
        imgpathpreprocessing = os.path.join(path, "preprocessing.tiff")
        imgpathpleura = os.path.join(path, "pleuramask.tiff")

        pleuraMask = cv2.imread(imgpathpleura, cv2.IMREAD_GRAYSCALE) > 0
        pleuraPreprocessing = cv2.imread(imgpathpreprocessing, cv2.IMREAD_GRAYSCALE)
        
        image_size = pleuraMask.shape
        for tilesize in tilesizear:
            ROI.makedir(os.path.join(path, str(tilesize)))
            ROI.makedir(os.path.join(path, str(tilesize), "roids"))

            imgpathtiles = os.path.join(path, str(tilesize), "tiles.png")
            imgpathrois = os.path.join(path, str(tilesize), "roids.nrrd")
            imgpathcontours = os.path.join(path, str(tilesize), "contours.json")
            csvtiles = os.path.join(path, str(tilesize), "info.csv")

            """ make tiles """
            start_time = time.time()
            print("pleuraPreprocessing", pleuraPreprocessing)
            pmask, index_roids, pleuraDataset = ROI.makeTiles(pleuraMask, tilesize, tilepercentage)
            #pmask, pleuraDataset = ROI.makeTilesBB(path, pleuraMask, pleuraPreprocessing, tilesize, tilepercentage)
            print("--- %s tiles seconds ---" % (time.time() - start_time))

            start_time = time.time()
            """ make contours """
            contours = ROI.make_contours(pmask, ispath=False)
            ROI.write(imgpathcontours, contours)
            print("--- %s contours seconds ---" % (time.time() - start_time))



            """ save tiles in binary file .nrrd """
            image = sitk.ReadImage(imgpathoriginal)
            image_mask = sitk.GetImageFromArray(pmask)
            image_mask.CopyInformation(image)
            sitk.WriteImage(image_mask, imgpathrois, True)  # True specifies it can use compression
            
            
            """ save tiles in random color .png """
            """
            ma_arr = np.zeros(image_size, dtype=int)
            ma_arr.fill(255)
            rgb_img = np.stack((ma_arr,)*3, axis=-1)
            for r in index_roids:
                rgb_img[r] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            cv2.imwrite(imgpathtiles, rgb_img)
            """


            """ save metadata """
            df = pd.DataFrame(pleuraDataset, columns=["size", "tilepercentage", "idseg"])
            df.to_csv(csvtiles, index=False)
            #print(df)        

            ROI.readTilesFrom_nrrd(path, tilesize)
            
        del pleuraMask
        del pleuraPreprocessing



    @staticmethod            
    def splitImage(image, tileSize):
        height, width = image.shape

        tiles = []
        positions = []
        for i in range(0, height, tileSize):
            for j in range(0, width, tileSize):
                aux_i = i + tileSize
                ls_i = aux_i if aux_i<(height-1) else height-1
                
                aux_j = j + tileSize
                ls_j = aux_j if aux_j<(width-1) else width-1
                
                positions.append(np.asarray((i, ls_i, j, ls_j)))
                tiles.append(image[i:ls_i, j:ls_j])

        return tiles, positions
        
    @staticmethod
    def makeTiles(pleuraMask, tilesize, tilepercentage):
        image_size = pleuraMask.shape
        height, width = pleuraMask.shape
        #maskTiles, positions = ROI.splitImage(pleuraMask, tilesize) # get positions just one here because it is the same
        smask = np.zeros(image_size, dtype=int)
        icc=1
        ind = []
        for i in range(0, height, tilesize):
            for j in range(0, width, tilesize):
                aux_i = i + tilesize
                ls_i = aux_i if aux_i<(height-1) else height-1
                
                aux_j = j + tilesize
                ls_j = aux_j if aux_j<(width-1) else width-1
        
                mTile = pleuraMask[i:ls_i, j:ls_j]

                #for position, mTile in zip(positions, maskTiles):
                # if not True in mTile:
                if mTile.shape[0] * mTile.shape[1] <= 0:
                    print("error ", mTile.shape[0] * mTile.shape[1])
                if mTile.shape[0] < tilesize/3.0 or mTile.shape[1] < tilesize/3.0:
                    continue
                if np.sum(mTile == True) <= ((mTile.shape[0] * mTile.shape[1]) * (tilepercentage)):
                    continue
                #pt = position
                imageTile = smask[i:ls_i, j:ls_j]
                imageTile[np.where(mTile == True)] = icc
                indices = np.where(smask == icc)
            
                ind.append(indices)
                icc+=1

        start_time = time.time()
        smask, ind = ROI.performregions(smask, ind)
        print("--- %s permor regions seconds ---" % (time.time() - start_time))

        rowvector = []
        for r in range(len(ind)):
            rowvector.append([len(ind[r][0])] + [tilepercentage] + [r+1] )

        return smask, ind, rowvector

    @staticmethod
    def makeTilesBB(pathimg, pleuraMask, imggray, tilesize, tilepercentage, isblackbg=True):
        image_size = pleuraMask.shape
        height, width = pleuraMask.shape
        smask = np.zeros(image_size, dtype=np.int)        
        #tiles = []
        sizeindc = []
        #start_time = time.time()
        co = 1
        for i in range(0, height, tilesize):
            for j in range(0, width, tilesize):
                aux_i = i + tilesize
                ls_i = aux_i if aux_i<(height-1) else height-1
                
                aux_j = j + tilesize
                ls_j = aux_j if aux_j<(width-1) else width-1
        
                mTile = pleuraMask[i:ls_i, j:ls_j]
                if mTile.shape[0] * mTile.shape[1] <= 0:
                    print("error ", mTile.shape[0] * mTile.shape[1])
                
                if mTile.shape[0] < tilesize/3.0 or mTile.shape[1] < tilesize/3.0:
                    continue

                if np.sum(mTile == True) <= ((mTile.shape[0] * mTile.shape[1]) * (tilepercentage)):
                    continue
                tilegray = imggray[i:ls_i, j:ls_j]
                tile = pleuraMask[i:ls_i, j:ls_j]
                
                #tile[np.where(mTile != True)] = 255
                #start_time2 = time.time()
                amaskTiles = ROI.conectedcomponent(tile)
                #print("--- %s permor regions seconds ---" % (time.time() - start_time2))
                
                #print("tilexx",tile)
                newg = np.zeros(tile.shape, dtype=np.int)
                #cv2.imwrite(os.path.join(pathimg, str(tilesize), "roids", str(co)+".jpg"), tile)
                for mm in amaskTiles:
                    
                    #if isblackbg:
                    #    mm = 255-mm
                    #tiles.append(mm)

                    indc = np.where(mm == True)
                    newg[indc] = co

                    imgcc = np.zeros((tilesize,tilesize))
                    #rowi, coli = mm.shape[0], mm.shape[1]
                    imgcc[indc] = tilegray[indc]
                    
                    cv2.imwrite(os.path.join(pathimg, str(tilesize), "roids", str(co)+".jpg"), imgcc)

                    sizeindc.append(indc)

                    co+=1
                    #print("ccWWWWWWW",co)
                    
                smask[i:ls_i, j:ls_j] = newg
        #print("ZZsmask", smask)        
        rowvector = []
        for r in range(len(sizeindc)):
            rowvector.append([len(sizeindc[r][0])] + [tilepercentage] + [r+1] )

        #print("--- %s ROI DL seconds ---" % (time.time() - start_time))
        return smask, rowvector

    @staticmethod
    def conectedcomponent(gray_im):
        """
        gray_im_inv = 255-gray_im
        gray_im_inv_bi = np.zeros(gray_im.shape, dtype='uint8')
        gray_im_inv_bi[np.where(gray_im_inv > 1)] = 255
        #thresh = 255-gray_im
        
        #gray_im_inv_bi = cv2.threshold(gray_im,10,255,cv2.THRESH_BINARY_INV)
        #gray_im_inv_bi, __ = cv2.threshold(gray_im_inv,1,255,cv2.THRESH_BINARY)

        # Dilatation et erosion
        kernel = np.ones((9,9), np.uint8)
        img_dilation = cv2.dilate(gray_im_inv_bi, kernel, iterations=1)
        img_erode = cv2.erode(img_dilation,kernel, iterations=1)
        # clean all noise after dilatation and erosion
        img_erode = cv2.medianBlur(img_erode, 9)


        #gray_im_inv_bi = cv2.morphologyEx(gray_im_inv_bi, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
        """        

        # Labeling
        print("gray_imXX", gray_im)
        gray_im = gray_im.astype(np.uint8)
        binary = cv2.threshold(gray_im, 0, 1, cv2.THRESH_BINARY)
        print("binary", binary)
        
        ret, labels = cv2.connectedComponents(gray_im)
        #ret, labels = cv2.connectedComponents(gray_im_inv_bi)
        #label_hue = np.uint8(179 * labels / np.max(labels))
        #blank_ch = 255 * np.ones_like(label_hue)

        #print("blank_ch", np.max(labels))

        """ labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
        labeled_img[label_hue == 0] = 0 """

        """ ret = []        
        nn = np.max(labels)
        for i in range(1,nn+1):
            indices = np.where(labels == i)            
            print("indices", len(indices))
            if len(indices[0])>1000:
            arr = np.zeros(gray_im.shape, dtype='uint8')
            arr[indices] = gray_im_inv[indices]
            ret.append(arr)
            #print("len(ret)", len(ret))
            #else: """
         
        ret = []
        nn = np.max(labels)
        if nn==1:
            ret.append(gray_im)
        elif nn>1 and nn<10:    
            cc = 0
            rett = []
            for i in range(1,nn+1):
                arr = np.zeros(gray_im.shape, dtype=np.bool)
                #arr.fill(255)
                indices = np.where(labels == i)
                #print("arr.shape, gray_im.shape", arr.shape, gray_im.shape)

                #if len(indices)>((wh*wh)*1.0/100.0):
                nnn = len(indices[0])
                if nnn>900*3:
                    #arr[indices] = gray_im[indices]
                    arr[indices] = gray_im[indices]

                    #arr = arr
                    rett.append(arr)
                else:
                    cc += 1
            if cc > nn/3.0:
                ret.append(gray_im)
            else:
                for rr in rett:
                    ret.append(rr)
        else:
            ret.append(gray_im)

        return ret


    @staticmethod
    def make_contours(filearray, ispath=True):
        """ return in format JSON contours of the rois in polygon form """
        image_mask = None
        if ispath:
            image_mask = sitk.ReadImage(filearray)
        else:
            image_mask = sitk.GetImageFromArray(filearray)

        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(image_mask)
        labels = lsif.GetLabels()
        #print("labels", labels)

        im_size = np.array(image_mask.GetSize())[::-1]
        image_array = sitk.GetArrayViewFromImage(image_mask)
        """ 
        dd1 = sitk.LabelContour(image_mask)
        reference_surface_arr = sitk.GetArrayViewFromImage(dd1)
        refpp = np.where(reference_surface_arr == 1)
        print("dd1", refpp)
         """
        """ 
        rng = np.random.default_rng()
        points = rng.random((30, 2))   # 30 random points in 2-D
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            print("hulls", points[simplex, 0], points[simplex, 1])
        """
        xp = [1, 1, 0,-1,-1,-1, 0, 1]
        yp = [0, 1, 1, 1, 0,-1,-1,-1]
        #print("im_size", im_size)
        #exit()
        results = []
        dd1 = sitk.LabelContour(image_mask)
        reference_surface_arr = sitk.GetArrayViewFromImage(dd1)

        for label in labels:
            maskk = np.zeros(im_size, dtype=np.uint8)
            maskk[np.where(image_array == label)] = 1
             
            #contours, hierarchy = cv2.findContours(maskk, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            #contours, hierarchy = cv2.findContours(maskk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(maskk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            

            # make polygons
            hierarchy = hierarchy[0]
            aupp = []
            for roww, hier in zip(contours, hierarchy):
                pauxpp = np.array([ [r[0][0], r[0][1]] for r in roww ], dtype=int)
                aupp.append(pauxpp.tolist())

            # make interiors contours
            auhier = [[] for e in hierarchy]
            for i in range(len(aupp)):
                i1d = hierarchy[i][3]
                if i1d != -1:
                    auhier[i1d].append(aupp[i])

            # append only outter contours
            for i in range(len(aupp)):
                if hierarchy[i][3] == -1:
                    results.append({"outters":aupp[i], "intters":auhier[i], "class":label, "group":label, "type":1})

        return results

    @staticmethod
    def make_contours2(roids):
        """ return in format JSON contours of the rois in polygon form """

        image_mask = sitk.ReadImage(roids)

        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(image_mask)
        labels = lsif.GetLabels()
        #print("labels", labels)
        
        """ 
        dd1 = sitk.LabelContour(image_mask)
        reference_surface_arr = sitk.GetArrayViewFromImage(dd1)
        refpp = np.where(reference_surface_arr == 1)
        print("dd1", refpp)
         """
        """ 
        rng = np.random.default_rng()
        points = rng.random((30, 2))   # 30 random points in 2-D
        hull = ConvexHull(points)
        for simplex in hull.simplices:
            print("hulls", points[simplex, 0], points[simplex, 1])
        """
        xp = [1, 1, 0,-1,-1,-1, 0, 1]
        yp = [0, 1, 1, 1, 0,-1,-1,-1]
        
        results = []
        dd1 = sitk.LabelContour(image_mask)
        reference_surface_arr = sitk.GetArrayViewFromImage(dd1)
        for label in labels:
            points = np.where(reference_surface_arr == label)
            #points = np.vstack((points[0], points[1])).T
            """ 
            aux = []
            print(len(points[1]))
            for i in range(len(points[1])):
                li = [points[0][i], points[1][i]]
                aux.append(li)

            points = aux """
            #print("points", points)
            """ 
            points = np.vstack((points[0], points[1])).T
            points = points.tolist() """


            """
            poly = geometry.Polygon(points)
            points = list(poly.exterior.coords)
            print(points)
            """

            #poly = np.array(poly.exterior.coords)
            #points = poly.tolist()
            
            #print(poly)
            #print("points", points)
            px = points[1].tolist()
            py = points[0].tolist()
            results.append({"pointsx":px, "pointsy":py, "class":label, "group":label})
            #print("results", results)
       

        #print(results)
        return results


    @staticmethod
    def performregions(mask, roids):
        x8 = [ 0, -1, 0, 1, -1, -1, 1,  1]
        y8 = [-1,  0, 1, 0, -1,  1, 1, -1]

        h, w = mask.shape[0], mask.shape[1]
        vis = [False for i in range(h*w)]
        mask_out = np.zeros(mask.shape, dtype=int)
        newroids = []
        for r in range(len(roids)):
            node = roids[r]
            for id in range(len(node[0])):
                x = node[0][id]
                y = node[1][id]
                i = x*w+y
                l = mask[x,y]
                if vis[i] == False:
                    vis[i] = True;
                    deq = collections.deque()
                    vecx, vecy = [],[]

                    deq.appendleft((x,y))
                    vecx.append(x)
                    vecy.append(y)
                    while deq:
                        ix, iy = deq.popleft()
                        for xd, yd in zip(x8, y8):
                            xc = ix+xd
                            yc = iy+yd
                            j = xc*w+yc
                            if xc>=0 and xc<h and yc>=0 and yc<w and mask[xc,yc]==l and vis[j]==False:
                                vis[j] = True;
                                deq.appendleft((xc,yc))
                                vecx.append(xc)
                                vecy.append(yc)
                    if len(vecx)>900:
                        newroids.append((np.array(vecx), np.array(vecy)))
        del vis
        for r in range(len(newroids)):
            node = newroids[r]
            mask_out[node] = (r+1)

        return mask_out, newroids

    @staticmethod
    def readTilesFrom_nrrd(pathimg, tiesize):
        ROI.makedir(os.path.join(pathimg, str(tiesize)))
        ROI.makedir(os.path.join(pathimg, str(tiesize), "roids"))
        
        image_gray = cv2.imread(os.path.join(pathimg, "preprocessing.tiff"), cv2.IMREAD_GRAYSCALE)
        image_mask = sitk.ReadImage(os.path.join(pathimg, str(tiesize), "roids.nrrd"))
        mask = sitk.GetArrayFromImage(image_mask)

        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(image_mask)
        labels = lsif.GetLabels()

        im_size = np.array(image_mask.GetSize())[::-1]
        #tiles = []
        #ii = 0
        for label in labels:
            img = np.zeros(im_size, dtype=np.uint8)
            img.fill(255)
            idx = np.where(mask == label)
            r1, r2 = np.min(idx[0]), np.max(idx[0])
            c1, c2 = np.min(idx[1]), np.max(idx[1])
            img[idx] = image_gray[idx]
            imgc = img[r1:r2, c1:c2]

            imgcc = np.zeros((tiesize,tiesize), dtype=np.uint8)
            imgcc.fill(255)
            rr, cc = imgc.shape[0], imgc.shape[1]
            imgcc[0:rr, 0:cc] = imgc

            imgcc = ROI.resegment(imgcc)
            #imgcc[np.where(imgcc == 255)] = 0
            cv2.imwrite(os.path.join(pathimg, str(tiesize), "roids", str(label)+".jpg"), imgcc)
            #tiles.append(imgcc)
            del img
            #ii += 1
        del image_mask

    def resegment(gray_im):
        
        gray_im_inv = 255-gray_im
        gray_im_inv_bi = np.zeros(gray_im.shape, dtype='uint8')
        gray_im_inv_bi[np.where(gray_im_inv > 5)] = 255

        # Dilatation et erosion
        kernel = np.ones((9,9), np.uint8)
        img_dilation = cv2.dilate(gray_im_inv_bi, kernel, iterations=1)
        img_erode = cv2.erode(img_dilation,kernel, iterations=1)
        # clean all noise after dilatation and erosion
        img_erode = cv2.medianBlur(img_erode, 9)

        # Labeling
        ret, labels = cv2.connectedComponents(img_erode)
         
        nn = np.max(labels)
        #print("blank_ch", nn)
        arr = np.zeros(gray_im.shape, dtype='uint8')
        #arr.fill(255)
        for i in range(1,nn+1):
            indices = np.where(labels == i)
            arr[indices] = gray_im[indices]
        
        #arr = 255 - arr 
        arr[np.where(arr >= 250)] = 0
        return arr

    @staticmethod
    def makedir(ndir):
        if not os.path.exists(ndir):
            os.makedirs(ndir)

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
    




if __name__ == "__main__":        

    """ 
    path ="/mnt/sda6/software/projects/data/media/lung/2021/05/ad2fa6a5c8dd472b8372eee7450c0156"
    rois_polygosn = Media.make_contours(path+"/"+"rois.nrrd")
    
    Media.write(path+"/"+"rois.json", rois_polygosn)
    rois_polygosn= Media.read(path+"/"+"rois.json")
    print(rois_polygosn)
    """

    path = "/mnt/sda6/software/frameworks/data/lha/dataset_4/testxx/"
    tilesize = 500
    tileperce = 0.01
    ROI.processROIs(path, tilesize, tileperce)



    