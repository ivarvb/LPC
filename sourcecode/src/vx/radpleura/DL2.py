
from pickletools import uint8
import cv2 as cv2
import time
import numpy as np
import os
import random
import SimpleITK as sitk

import collections
from sklearn.model_selection import train_test_split
from multiprocessing import Pool, Manager, Process, Lock, cpu_count
from datetime import datetime
import ujson

from skimage.feature import local_binary_pattern


import shutil

class Tiles:
    @staticmethod
    def execute(pleuraMask, imggray, tilesize, tilepercentage, isblackbg):
        start_time = time.time()

        height, width = pleuraMask.shape
        
        maskTiles = []
        positions = []
        for i in range(0, height, tilesize):
            for j in range(0, width, tilesize):
                aux_i = i + tilesize
                ls_i = aux_i if aux_i<(height-1) else height-1
                
                aux_j = j + tilesize
                ls_j = aux_j if aux_j<(width-1) else width-1
        
                mTile = pleuraMask[i:ls_i, j:ls_j]
                if mTile.shape[0] * mTile.shape[1] <= 0:
                    print("error ", mTile.shape[0] * mTile.shape[1])
                
                #if tilesize != mTile.shape[0] or tilesize != mTile.shape[1]:
                if mTile.shape[0] < tilesize/3.0 or mTile.shape[1] < tilesize/3.0:
                    continue
                if np.sum(mTile == True) <= ((mTile.shape[0] * mTile.shape[1]) * (tilepercentage)):
                    continue
                #print("mTile ", mTile.shape)
                
                #imggray[np.where(mTile != True)] = 0
                tile = imggray[i:ls_i, j:ls_j]
                tile[np.where(mTile != True)] = 255
                #if isblackbg:
                #    tile[np.where(tile == 255)] = 0


                start_time2 = time.time()
                #amaskTiles = Tiles.conectedcomponent(tile)########?????????????
                
                #amaskTiles = Tiles.conectedcomponent2(tile, mTile)
                amaskTiles = Tiles.conectedcomponent2black(tile, mTile)#white
                
                #print("--- %s permor regions seconds ---" % (time.time() - start_time2))
                for mm in amaskTiles:
                    indc = np.where(mm != 0)
                    imgcc = np.zeros((tilesize,tilesize))
                    imgcc.fill(255)############ is white
                    imgcc[indc] = mm[indc]############
                    
                    #imgcc = local_binary_pattern(imgcc,8,2,method='uniform')
                    #imgcc = (imgcc - np.min(imgcc)) / (np.max(imgcc)-np.min(imgcc))*255

                    
                    ##
                    #imgcc = (imgcc - np.min(imgcc)) / (np.max(imgcc)-np.min(imgcc))*255
                    ###imgcc = (imgcc - np.mean(imgcc)) / np.std(imgcc)*255

                    #imgcc[np.where(imgcc > 180)] = 0
                    #imgcc[np.where(imgcc > 0)] = 255


                    """
                    if isblackbg:
                        maskTiles.append(255-mm)
                    else:
                        maskTiles.append(mm)    
                    """
                    maskTiles.append(imgcc)


                """ tile = SplitPleura.resize(tile, resize) """

                #maskTiles.append(np.asarray(tile, dtype=np.uint8))
                #positions.append(np.asarray((i, ls_i, j, ls_j)))

        print("--- %s ROI DL seconds ---" % (time.time() - start_time))

        """ start_time = time.time()
        maskTiles, ind = Tiles.performregions(imggray, ind)
        print("--- %s permor regions seconds ---" % (time.time() - start_time)) """

        return maskTiles, positions

    @staticmethod
    def performregions(mask, roids):
        x8 = [ 0, -1, 0, 1, -1, -1, 1,  1]
        y8 = [-1,  0, 1, 0, -1,  1, 1, -1]

        h, w = mask.shape[0], mask.shape[1]
        vis = [False for i in range(h*w)]
        mask_out = np.zeros(mask.shape, dtype=np.uint8)
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
                    if len(vecx)>100:
                        newroids.append((np.array(vecx), np.array(vecy)))
        del vis
        for r in range(len(newroids)):
            node = newroids[r]
            mask_out[node] = (r+1)

        return mask_out, newroids

    @staticmethod
    def conectedcomponent(gray_im):
        """ # Contrast adjusting with gamma correction y = 1.2
        gray_im = np.array(255 * (gray_im / 255) ** 1.2 , dtype='uint8')
        #gray_equ = cv2.equalizeHist(gray_im)

        # Local adaptative threshold
        thresh = cv2.adaptiveThreshold(gray_im, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 255, 19)
        thresh = cv2.bitwise_not(thresh) """
        
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

        # Labeling
        ret, labels = cv2.connectedComponents(img_erode)
        #ret, labels = cv2.connectedComponents(gray_im_inv_bi)
        #label_hue = np.uint8(179 * labels / np.max(labels))
        #blank_ch = 255 * np.ones_like(label_hue)

        print("blank_ch", np.max(labels))

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
                arr = np.zeros(gray_im.shape, dtype='uint8')
                arr.fill(255)
                indices = np.where(labels == i)

                nnn = len(indices[0])
                if nnn>900*3:
                    #arr[indices] = gray_im[indices]
                    arr[indices] = gray_im[indices]

                    arr = arr
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
    def conectedcomponent2(gray_im, mask):
        gray_im_inv = 255-gray_im
        gray_im_inv_bi = np.zeros(gray_im.shape, dtype='uint8')
        gray_im_inv_bi[np.where(gray_im_inv > 1)] = 255

        # Dilatation et erosion
        kernel = np.ones((9,9), np.uint8)
        img_dilation = cv2.dilate(gray_im_inv_bi, kernel, iterations=1)
        img_erode = cv2.erode(img_dilation,kernel, iterations=1)
        # clean all noise after dilatation and erosion
        img_erode = cv2.medianBlur(img_erode, 9)

        # Labeling
        ret, labels = cv2.connectedComponents(img_erode)

        print("blank_ch", np.max(labels))
         
        ret = []
        nn = np.max(labels)
        if nn==1:
            #gray_im[np.where(gray_im >= 250)] = 0
            arr = np.zeros(gray_im.shape, dtype='uint8')
            indxx = np.where(mask==True)
            arr[indxx] = gray_im[indxx]
            ret.append(arr)
        elif nn>1 and nn<10:    
            cc = 0
            rett = []
            for i in range(1,nn+1):
                arr = np.zeros(gray_im.shape, dtype='uint8')
                #arr.fill(255)
                indices = np.where(labels == i)

                nnn = len(indices[0])
                if nnn>900*3:
                    arr[indices] = gray_im[indices]
                    #arr[np.where(arr >= 250)] = 0
                    rett.append(arr)
                else:
                    cc += 1
            if cc > nn/3.0:
                #gray_im[np.where(gray_im >= 250)] = 0
                arr = np.zeros(gray_im.shape, dtype='uint8')
                indxx = np.where(mask==True)
                arr[indxx] = gray_im[indxx]
                ret.append(arr)
            else:
                for rr in rett:
                    ret.append(rr)
        else:
            #gray_im[np.where(gray_im >= 250)] = 0
            arr = np.zeros(gray_im.shape, dtype='uint8')
            indxx = np.where(mask==True)
            arr[indxx] = gray_im[indxx]
            ret.append(arr)

        return ret


    @staticmethod
    def conectedcomponent2black(gray_im, mask):
        gray_im_inv = 255-gray_im
        #limiar new filter 
        #gray_im_inv[np.where(gray_im_inv > 180)] = 0

        gray_im_inv_bi = np.zeros(gray_im.shape, dtype='uint8')
        gray_im_inv_bi[np.where(gray_im_inv > 1)] = 255

        # Dilatation et erosion
        kernel = np.ones((9,9), np.uint8)
        img_dilation = cv2.dilate(gray_im_inv_bi, kernel, iterations=1)
        img_erode = cv2.erode(img_dilation,kernel, iterations=1)
        # clean all noise after dilatation and erosion
        img_erode = cv2.medianBlur(img_erode, 9)

        # Labeling
        ret, labels = cv2.connectedComponents(img_erode)

        print("blank_ch", np.max(labels))
         
        ret = []
        nn = np.max(labels)
        if nn==1:
            #gray_im[np.where(gray_im >= 250)] = 0
            arr = np.zeros(gray_im.shape, dtype='uint8')
            arr.fill(255)
            indxx = np.where(mask==True)
            
            #indxx = np.where(labels == 0)

            #indenull = np.where(gray_im_inv[indxx]==0)
            #gray_im_inv[indenull] = 255

            #arr[indxx] = gray_im_inv[indxx]
            arr[indxx] = gray_im[indxx]
            
            #arr = 255-arr
            ret.append(arr)

        elif nn>1 and nn<10:    
            cc = 0
            rett = []
            for i in range(1,nn+1):
                arr = np.zeros(gray_im.shape, dtype='uint8')
                arr.fill(255)
                indices = np.where(labels == i)

                nnn = len(indices[0])
                if nnn>900*3:
                    #arr[indices] = gray_im[indices]
                    #arr[indices] = 255
                    
                    #arr[indices] = gray_im_inv[indices]
                    arr[indices] = gray_im[indices]
                    #arr[indenull] = 255
                    #arr = 255-arr
                    rett.append(arr)
                else:
                    cc += 1
            if cc > nn/3.0:
                arr = np.zeros(gray_im.shape, dtype='uint8')
                arr.fill(255)
                indxx = np.where(mask==True)


                #arr[indxx] = gray_im[indxx]

                #arr[indxx] = 255
                #indenull = np.where(gray_im_inv[indxx]>0)        
                
                #arr[indxx] = gray_im_inv[indxx]
                arr[indxx] = gray_im[indxx]
                
                #arr[indxx] = gray_im_inv[indxx]
                #arr = 255-arr
                ret.append(arr)
            else:
                for rr in rett:
                    ret.append(rr)
        else:
            #gray_im[np.where(gray_im >= 250)] = 0
            arr = np.zeros(gray_im.shape, dtype='uint8')
            arr.fill(255)            
            indxx = np.where(mask==True)
            #arr[indxx] = gray_im[indxx]

            #arr[indxx] = 255
            #indenull = np.where(gray_im_inv[indxx]>0)        
            
            #arr[indxx] = gray_im_inv[indxx]
            arr[indxx] = gray_im[indxx]
            
            #arr[indxx] = gray_im_inv[indxx]
            #arr = 255-arr
            ret.append(arr)

        return ret

class DLPleura:
    def __init__(self) -> None:
        pass
    
    def train(X, y):
        pass
    
    def test(X, y):
        pass

    def kfold(X, y):
        pass
        


class SplitPleura:
    ARGS = []
    def __init__(self, args):
        pass

    @staticmethod
    def process(dat):
        pleuraPathMask =  SplitPleura.ARGS["pleuraPathMask"]
        nonpleuraPathMask =  SplitPleura.ARGS["nonpleuraPathMask"]
        pathgray = SplitPleura.ARGS["pathgray"]
        pathrgb = SplitPleura.ARGS["pathrgb"]
        
        tilesize = SplitPleura.ARGS["tilesize"]
        tilepercentage = SplitPleura.ARGS["tilepercentage"]
        isblackbg = SplitPleura.ARGS["isblackbg"]
        isRGB = SplitPleura.ARGS["isRGB"]      

        outputdir = SplitPleura.ARGS["outputdir"]

        filename = dat["filename"]
        bfile = os.path.basename(filename)
        bfile = os.path.splitext(bfile)

        pleuraMask = cv2.imread(os.path.join(pleuraPathMask, filename), cv2.IMREAD_GRAYSCALE) > 0
        nonpleuraMask = cv2.imread(os.path.join(nonpleuraPathMask, filename), cv2.IMREAD_GRAYSCALE) > 0
        imgGray = cv2.imread(os.path.join(pathgray, filename), cv2.IMREAD_GRAYSCALE)

        tp_tiles, tp_positions = Tiles.execute(pleuraMask, imgGray, tilesize, tilepercentage, isblackbg)
        fp_tiles, fp_positions = Tiles.execute(nonpleuraMask, imgGray, tilesize, tilepercentage, isblackbg)

        tp_tiles = np.array(tp_tiles, dtype=np.uint8)
        fp_tiles = np.array(fp_tiles, dtype=np.uint8)


        # shutil.copyfile(os.path.join(pleuraPathMask, filename), os.path.join(outputdir, bfile[0], "pleura.tiff"))
        # shutil.copyfile(os.path.join(nonpleuraPathMask, filename), os.path.join(outputdir, bfile[0], "nonpleura.tiff"))
        # shutil.copyfile(os.path.join(pathgray, filename), os.path.join(outputdir, bfile[0], "gray.tiff"))


        SplitPleura.saveimgarry(tp_tiles, os.path.join(outputdir, "tiles", bfile[0], "pleura"), isRGB)
        SplitPleura.saveimgarry(fp_tiles, os.path.join(outputdir, "tiles", bfile[0], "nonpleura"), isRGB)

        del pleuraMask
        del nonpleuraMask
        del imgGray
        del tp_positions

        #return tp_tiles, fp_tiles, bfile[0]


    @staticmethod
    def major(l1,l2):
        n1 = len(l1)
        n2 = len(l2)
        print("n1, n2", n1, n2)
        mx = n1
        if n2<mx:
            mx = n2
        return mx

    @staticmethod
    def copyfi(l_test, outputdir, outputdir_test):
        for pfi, npfi in zip(l_test[0], l_test[1]):
            tip, fip = pfi
            tinp, finp = npfi
            pt_pl = os.path.join(outputdir, "tiles", str(tip), "pleura")
            pt_npl = os.path.join(outputdir, "tiles", str(tinp), "nonpleura")

            shutil.copyfile(os.path.join(pt_pl, fip), os.path.join(outputdir_test, "pleura", str(tip)+"_"+fip))
            shutil.copyfile(os.path.join(pt_npl, finp), os.path.join(outputdir_test, "nonpleura", str(tinp)+"_"+finp))
        
    @staticmethod
    def readdirectory(trainx, inputdir):
        lpleura, lnonpleura = [],[]
        for ti in trainx:
            pt_pl = os.path.join(inputdir, "tiles", str(ti), "pleura")
            pt_npl = os.path.join(inputdir, "tiles", str(ti), "nonpleura")
            for fi in os.listdir(pt_pl):
                lpleura.append([ti,fi])
            for fi in os.listdir(pt_npl):
                lnonpleura.append([ti,fi])
        return lpleura, lnonpleura


    @staticmethod
    #make_nrrd(os.path.join(imgfile), outputdir, "nonpleura", bfile[0])
    def make_nrrd(imagepath, outputdir, ispleura, bfile):
        gray_im = cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE)
        #print("gray_im", gray_im)
        gray_im_inv = 255-gray_im

        gray_im_inv_bi = np.zeros(gray_im.shape, dtype='uint8')
        gray_im_inv_bi[np.where(gray_im_inv > 1)] = 255

        # Dilatation et erosion
        kernel = np.ones((9,9), np.uint8)
        img_dilation = cv2.dilate(gray_im_inv_bi, kernel, iterations=1)
        img_erode = cv2.erode(img_dilation,kernel, iterations=1)
        # clean all noise after dilatation and erosion
        img_erode = cv2.medianBlur(img_erode, 9)

        #img_erode[np.where(img_erode > 0)] = 1
        #img_erode[np.where(img_erode == 0)] = 0

        maskarr = np.zeros(gray_im.shape, dtype='uint8')
        maskarr[np.where(img_erode > 0)] = 1

        image = sitk.GetImageFromArray(gray_im)
        # image = sitk.ReadImage(imgpathoriginal)
        image_mask = sitk.GetImageFromArray(maskarr)
        #print("gray_im", image_mask)
        image_mask.CopyInformation(image)
        #sitk.WriteImage(image_mask, os.path.join(outputdir, ispleura, bfile+".nrrd"), True)  # True specifies it can use compression

        cv2.imwrite(os.path.join(outputdir, ispleura, bfile+".png"), img_erode)


    @staticmethod
    def make_nrrd_from_dir(inputdir, outputdir):
        SplitPleura.makedir(os.path.join(outputdir))
        SplitPleura.makedir(os.path.join(outputdir, "pleura"))
        SplitPleura.makedir(os.path.join(outputdir, "nonpleura"))
        for imgfile in os.listdir(os.path.join(inputdir, "pleura")):
            print("imgfile", "pleura", imgfile)
            filename = os.path.basename(imgfile)
            bfile = os.path.splitext(filename)
            SplitPleura.make_nrrd(os.path.join(inputdir, "pleura", imgfile), outputdir, "pleura", bfile[0])
        
        for imgfile in os.listdir(os.path.join(inputdir, "nonpleura")):
            print("imgfile", "nonpleura", imgfile)
            filename = os.path.basename(imgfile)
            bfile = os.path.splitext(filename)
            SplitPleura.make_nrrd(os.path.join(inputdir, "nonpleura", imgfile), outputdir, "nonpleura", bfile[0])



    @staticmethod
    def split_train_test(inputdir, limittest, limittrain):

        idx = [i for i in range(1,319)]
        random.seed(1042)
        random.shuffle(idx)

        n = len(idx)
        ncut = int((10/100.0)*n)
        ncut = 30

        group = []
        for i in range(0, ncut*10, ncut):

            #l = i+10
            g = {"test":[], "train":[]}
            g["test"] = idx[i:i+ncut]
            g["train"] = idx[0:i]+idx[i+ncut:]
            group.append(g)
        
        dsi = 0
        for row in group:
            dsi += 1
            testx, trainx = row["test"], row["train"]


            #n = len(idx)
            #ncut = int((10/100.0)*n)
            #testx = idx[:ncut]
            #trainx = idx[ncut:]


            
            SplitPleura.makedir(os.path.join(inputdir, str(dsi)))
            SplitPleura.makedir(os.path.join(inputdir, str(dsi), "test"))
            outputdir_test = os.path.join(inputdir, str(dsi), "test")
            SplitPleura.makedir(os.path.join(outputdir_test, "pleura"))
            SplitPleura.makedir(os.path.join(outputdir_test, "nonpleura"))

            SplitPleura.makedir(os.path.join(inputdir, str(dsi)))
            SplitPleura.makedir(os.path.join(inputdir, str(dsi), "train"))
            outputdir_train = os.path.join(inputdir, str(dsi), "train")
            SplitPleura.makedir(os.path.join(outputdir_train, "pleura"))
            SplitPleura.makedir(os.path.join(outputdir_train, "nonpleura"))

            SplitPleura.makedir(os.path.join(inputdir, str(dsi)))
            SplitPleura.makedir(os.path.join(inputdir, str(dsi), "vali"))
            outputdir_vali = os.path.join(inputdir, str(dsi), "vali")
            SplitPleura.makedir(os.path.join(outputdir_vali, "pleura"))
            SplitPleura.makedir(os.path.join(outputdir_vali, "nonpleura"))


            #TEST
            l_test_pleura, l_test_nonpleura = SplitPleura.readdirectory(testx,inputdir)
            random.shuffle(l_test_pleura)
            random.shuffle(l_test_nonpleura)
            tmx = SplitPleura.major(l_test_pleura, l_test_nonpleura)
            #balancing
            l_test_pleura = l_test_pleura[:tmx]
            l_test_nonpleura = l_test_nonpleura[:tmx]
            #sampling            
            if len(l_test_pleura)>limittest:
                l_test_pleura = l_test_pleura[:limittest]
                l_test_nonpleura = l_test_nonpleura[:limittest]

            SplitPleura.copyfi((l_test_pleura, l_test_nonpleura), inputdir, outputdir_test)


            #TRAIN AND VALIDATION
            l_train_pleura, l_train_nonpleura, = SplitPleura.readdirectory(trainx,inputdir)
            l_vali_pleura, l_vali_nonpleura = [], []
            random.shuffle(l_train_pleura)
            random.shuffle(l_train_nonpleura)
            tmx = SplitPleura.major(l_train_pleura, l_train_nonpleura)
            #balancing
            l_train_pleura = l_train_pleura[:tmx]
            l_train_nonpleura = l_train_nonpleura[:tmx]

            # #split validation
            # l_vali[0] = l_train[0][:len(l_test[0])]
            # l_vali[1] = l_train[1][:len(l_test[0])]

            # l_train[0] = l_train[0][len(l_test[0]):]
            # l_train[1] = l_train[1][len(l_test[0]):]

            #sampling            
            if len(l_train_pleura)>limittrain:
                l_train_pleura = l_train_pleura[:limittrain]
                l_train_nonpleura = l_train_nonpleura[:limittrain]

            # SplitPleura.copyfi(l_vali, inputdir, outputdir_vali)
            SplitPleura.copyfi((l_train_pleura, l_train_nonpleura), inputdir, outputdir_train)



    @staticmethod
    def execute(args):
        SplitPleura.ARGS = args
        
        start_time = time.time()

        pleuraPathMask =  args["pleuraPathMask"]
        nonpleuraPathMask =  args["nonpleuraPathMask"]
        pathgray =  args["pathgray"]
        pathrgb =  args["pathrgb"]
        
        outputdir =  args["outputdir"]
        tilesize = args["tilesize"]
        tilepercentage = args["tilepercentage"]
        #limitsize = args["limitsize"]
        isRGB = args["isRGB"]
        isResize = args["isResize"]
        limittrain = args["limitSampleTrain"]
        limittest = args["limitSampleTest"]

        SplitPleura.ARGS["outputdir"] = os.path.join(outputdir, SplitPleura.now())
        outputdir = SplitPleura.ARGS["outputdir"]

        SplitPleura.makedir(os.path.join(outputdir))
        SplitPleura.makedir(os.path.join(outputdir,"tiles"))

        rs_tp, rs_fp = [], []
        #rs_po, fprs_po = [],[]
        rss = []
        if args["type"] == "superpixels":
            for img in args["inputdir"]:
                pass
                """ 
                #set black backgroun                
                imgsps = SNIC.execute(img, s)
                rs.append(imgsps)
                rsid.append(index_roids) """

        elif args["type"] == "tiles":
            #c = 0
            aux = []
            for imgfile in os.listdir(pathgray):
                filename = os.path.basename(imgfile)
                bfile = os.path.splitext(filename)
                #print(c, "filename", filename)
                if filename.endswith(".tiff"):
                    print("filename", bfile)
                    aux.append({"filename":filename})

                    ndir = os.path.join(outputdir, "tiles", bfile[0])
                    SplitPleura.makedir(os.path.join(ndir,"pleura"))
                    SplitPleura.makedir(os.path.join(ndir,"nonpleura"))
            
            pool = Pool(processes=int(cpu_count()-3))
            pool.map(SplitPleura.process, aux)
            pool.close()


        SplitPleura.split_train_test(outputdir, limittest, limittrain)

        """
        idf_tp, idf_fp = [], []
        for row in rss:
            tp, fp, idfi = row[0], row[1], row[2]
            for tpi in tp:
                rs_tp.append(tpi)
                idf_tp.append(idfi)
            for fpi in fp:
                rs_fp.append(fpi)
                idf_fp.append(idfi)

        ndir = os.path.join(outputdir, SplitPleura.now())
        SplitPleura.makedir(os.path.join(ndir, "pleura"))
        SplitPleura.makedir(os.path.join(ndir, "nonpleura"))
        
        SplitPleura.makedir(os.path.join(ndir, "train"))
        SplitPleura.makedir(os.path.join(ndir, "vali"))
        SplitPleura.makedir(os.path.join(ndir, "test"))
        
        SplitPleura.makedir(os.path.join(ndir, "train", "pleura"))
        SplitPleura.makedir(os.path.join(ndir, "train", "nonpleura"))
        
        SplitPleura.makedir(os.path.join(ndir, "vali", "pleura"))
        SplitPleura.makedir(os.path.join(ndir, "vali", "nonpleura"))

        SplitPleura.makedir(os.path.join(ndir, "test", "pleura"))
        SplitPleura.makedir(os.path.join(ndir, "test", "nonpleura"))

        rs_tp = np.array(rs_tp, dtype=np.uint8)
        rs_fp = np.array(rs_fp, dtype=np.uint8)


        #whole
        rs_tp, rs_fp = SplitPleura.randomchosse(rs_tp, rs_fp)
        #print("x_pleura", "x_nonpleura", len(rs_tp), len(rs_fp))

        SplitPleura.saveimgarry(rs_tp, os.path.join(ndir, "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(rs_fp, os.path.join(ndir, "nonpleura"), isRGB, isResize)


        #train test
        train_pleura, train_nonpleura, vali_pleura, vali_nonpleura, test_pleura, test_nonpleura = SplitPleura.split_train_test(rs_tp, rs_fp, idf_tp, idf_fp)
        SplitPleura.saveimgarry(train_pleura, os.path.join(ndir, "train", "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(train_nonpleura, os.path.join(ndir, "train", "nonpleura"), isRGB, isResize)
        SplitPleura.saveimgarry(vali_pleura, os.path.join(ndir, "vali", "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(vali_nonpleura, os.path.join(ndir, "vali", "nonpleura"), isRGB, isResize)
        SplitPleura.saveimgarry(test_pleura, os.path.join(ndir, "test", "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(test_nonpleura, os.path.join(ndir, "test", "nonpleura"), isRGB, isResize)

        """
        
    @staticmethod
    def write(file, obj):
        with open(file, "w") as filef:
            filef.write(ujson.dumps(obj))


    @staticmethod
    def saveimgarry(arr, path, isRGB, isResize=None):
        # print("arr.shape", arr.shape)
        ii=0
        for tpi in arr:            
            if isResize != None:
                #print("tpi, isResize", tpi.shape, tpi, isResize)
                #tpi = tf.image.resize(tpi, (86, 240))
                tpi = SplitPleura.toResize(tpi, isResize)
                
            """ if isRGB:
                tpi = SplitPleura.toRGB(tpi) """

            if isRGB:
                cv2.imwrite(os.path.join(path, str(ii)+".png"), tpi)
            else:
                cv2.imwrite(os.path.join(path, str(ii)+".jpg"), tpi)
            ii+=1

            del tpi

    @staticmethod
    def makedir(ndir):
        if not os.path.exists(ndir):
            os.makedirs(ndir)

    @staticmethod
    def now():
        return datetime.now().strftime("%Y%m%d%H%M%S")

    @staticmethod
    def Xy(path):
        pleura = np.load(os.path.join(path, "pleura.npy"))
        nonpleura = np.load(os.path.join(path, "nonpleura.npy"))
        
        zpleura = pleura.shape[0]
        znonpleura = nonpleura.shape[0]

        idx = [i for i in range(znonpleura)]
        random.seed(42)
        random.shuffle(idx)
    
        X = np.concatenate((pleura, nonpleura[idx[:zpleura]]), axis=0)
        y = np.array([1 for i in range(zpleura)] + [0 for i in range(zpleura)])

        np.save(os.path.join(path, 'X.npy'), X)
        np.save(os.path.join(path, 'y.npy'), y)

        #Xtest, Xtest, ytrain, ytest = train_test_split(X, y, test_size=(20.0/10.0), random_state=42, stratify=yw)
        

    @staticmethod
    def train_test(path):
        X = np.load(os.path.join(path, "X.npy"))
        y = np.load(os.path.join(path, "y.npy"))

        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=(10.0/100.0), random_state=42, stratify=y)
        print("Xtrain, Xtest, ytrain, ytest",Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

        np.save(os.path.join(path, 'Xtrain.npy'), Xtrain)
        np.save(os.path.join(path, 'Xtest.npy'), Xtest)
        np.save(os.path.join(path, 'ytrain.npy'), ytrain)
        np.save(os.path.join(path, 'ytest.npy'), ytest)


        
    @staticmethod
    def toRGB(img):
        img = np.zeros([img.shape[0], img.shape[1], 3], dtype=np.uint8)
        return cv2.merge([img, img, img])
        
    @staticmethod
    def toResize(img, resize):
        return cv2.resize(img, resize, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def resizeRGBprocess(img, resize):
        img = SplitPleura.toResize(img, resize)
        return SplitPleura.toRGB(img)

    @staticmethod
    def resizeRGBprocessList(X, resize):
        Xr = []
        for i in range(len(X)):
            print("X[i]", i, X[i].shape)
            Xr.append(SplitPleura.resizeRGBprocess(X[i], resize))
        return np.array(Xr, dtype=np.uint8)

    @staticmethod
    def resizeRGB(path, resize):
        X = np.load(os.path.join(path, "X.npy"))
        y = np.load(os.path.join(path, "y.npy"))

        X = SplitPleura.resizeRGBprocessList(X, resize)

        print("len(X), len(y)", len(X), len(y))

        
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=(10.0/100.0), random_state=42, stratify=y)
        """ Xtrain = SplitPleura.resizeRGBprocessList(Xtrain, resize)
        Xtest = SplitPleura.resizeRGBprocessList(Xtest, resize) """
        print("Xtrain, Xtest, ytrain, ytest",Xtrain.shape, Xtest.shape, ytrain.shape, ytest.shape)

        np.save(os.path.join(path, 'XtrainRGB.npy'), Xtrain)
        np.save(os.path.join(path, 'XtestRGB.npy'), Xtest)
        np.save(os.path.join(path, 'ytrainRGB.npy'), ytrain)
        np.save(os.path.join(path, 'ytestRGB.npy'), ytest)

        #tile = SplitPleura.resize(tile, resize)

    @staticmethod
    def randomchosse(imglist1, imglist2):
        n1 = len(imglist1)
        n2 = len(imglist2)

        isoe = True
        minn = n1
        if n2<minn:
            minn = n2
            isoe = False

        idx = [i for i in range(minn)]
        random.seed(42)
        random.shuffle(idx)

        if isoe:
            #imglist1 = imglist1[idx[:minn]]
            imglist2 = imglist2[idx[:minn]]
        else:
            imglist1 = imglist1[idx[:minn]]
            #imglist2 = imglist2[idx[:minn]]

        return imglist1, imglist2
         
    @staticmethod
    def readTilesFrom_nrrd(pathimg, tiesize):
        tiesize = str(tiesize)
        SplitPleura.makedir(os.path.join(pathimg, tiesize))
        SplitPleura.makedir(os.path.join(pathimg, tiesize, "roids"))
        
        image_gray = cv2.imread(os.path.join(pathimg, "preprocessing.tiff"), cv2.IMREAD_GRAYSCALE)
        image_mask = sitk.ReadImage(os.path.join(pathimg, "roids"+tiesize+".nrrd"))
        mask = sitk.GetArrayFromImage(image_mask)

        lsif = sitk.LabelShapeStatisticsImageFilter()
        lsif.Execute(image_mask)
        labels = lsif.GetLabels()

        im_size = np.array(image_mask.GetSize())[::-1]
        #tiles = []
        ii = 0
        for label in labels:
            img = np.zeros(im_size, dtype=np.uint8)
            img.fill(255)
            idx = np.where(mask == label)
            r1, r2 = np.min(idx[0]), np.max(idx[0])
            c1, c2 = np.min(idx[1]), np.max(idx[1])
            img[idx] = image_gray[idx]
            imgc = 255-img[r1:r2, c1:c2]
            #imgc = np.hstack([imgc, np.zeros([500, 500])])
            imgcc = np.zeros((500,500))
            rr, cc = imgc.shape[0], imgc.shape[1]
            #print("rr, cc", rr, cc)
            imgcc[0:rr, 0:cc] = imgc
            #imgc = np.concatenate((imgc,np.zeros((500,500))), axis=1)
            print("imgcc.shape", imgcc.shape)

            cv2.imwrite(os.path.join(pathimg, tiesize, str(ii)+".jpg"), imgcc)
            #tiles.append(imgcc)
            del img
            ii += 1
        del image_mask

        #tiles = np.array(tiles)
        #print(tiles, tiles)
        #np.save(os.path.join(pathimg, 'X.npy'), tiles)
        #return tiles
        

if __name__ == "__main__":

    das = [
            # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/500/20230825140850",
            # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/400/20230826121633",
            # "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/300/20230826124152",
            "/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/200/20230826124856",

    ]
    for mp in das:
        for i in range(1,5+1):
            outputdir = mp+"/"+str(i)+"/mask"
            inputdir = mp+"/"+str(i)+"/test"
            SplitPleura.make_nrrd_from_dir(inputdir, outputdir)
            inputdir = mp+"/"+str(i)+"/train"
            SplitPleura.make_nrrd_from_dir(inputdir, outputdir)

    
    """    

    slide = [
                #{"tilesize":50, "isResize":None, "tilepercentage":0.1},
                {"tilesize":100, "isResize":None, "tilepercentage":0.1},
                #{"tilesize":200, "isResize":None, "tilepercentage":0.1},
                #{"tilesize":300, "isResize":None, "tilepercentage":0.1},
                #{"tilesize":400, "isResize":None, "tilepercentage":0.1},
                #{"tilesize":500, "isResize":None, "tilepercentage":0.1},
            ]
    for slidei in slide:
        tilesize, isResize, tilepercentage = slidei["tilesize"], slidei["isResize"], slidei["tilepercentage"]
        args = {
                "type":"tiles",
                "tilesize":tilesize,
                "tilepercentage":tilepercentage,
                "outputdir":"/mnt/sda6/software/frameworks/data/lha/dataset_3/DL/"+str(tilesize),
                "pathgray":"/mnt/sda6/software/frameworks/data/lha/dataset_3/grayscale/both",
                "pathrgb":"/mnt/sda6/software/frameworks/data/lha/dataset_3/images",
                "pleuraPathMask":"/mnt/sda6/software/frameworks/data/lha/dataset_3/masks/erode_radius_30/pleura",
                "nonpleuraPathMask":"/mnt/sda6/software/frameworks/data/lha/dataset_3/masks/erode_radius_30/non_pleura",
                "isblackbg":True,
                "isRGB":True,
                "isResize":isResize,
                "limitSampleTrain":10000,
                "limitSampleTest":1000
                }

        sp = SplitPleura.execute(args)
        del sp
    

    #pathimg = "/mnt/sda6/software/frameworks/data/radpleura/lung/2022/04/6262c73a3ff8bc1a5f5af31b/"
    #SplitPleura.readTilesFrom_nrrd(pathimg,500)    

    """    