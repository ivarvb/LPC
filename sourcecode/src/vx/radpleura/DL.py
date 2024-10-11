
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
                #amaskTiles = Tiles.conectedcomponent(tile)
                amaskTiles = Tiles.conectedcomponent2(tile, mTile)
                print("--- %s permor regions seconds ---" % (time.time() - start_time2))
                for mm in amaskTiles:
                    indc = np.where(mm != 0)
                    imgcc = np.zeros((tilesize,tilesize))
                    imgcc[indc] = mm[indc]

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
        
        #outputdir = SplitPleura.ARGS["outputdir"]
        tilesize = SplitPleura.ARGS["tilesize"]
        tilepercentage = SplitPleura.ARGS["tilepercentage"]
        isblackbg = SplitPleura.ARGS["isblackbg"]
        isRGB = SplitPleura.ARGS["isRGB"]
        
        #resize = SplitPleura.ARGS["resize"]
        
        
        filename = dat["filename"]

        pleuraMask = cv2.imread(os.path.join(pleuraPathMask, filename), cv2.IMREAD_GRAYSCALE) > 0
        nonpleuraMask = cv2.imread(os.path.join(nonpleuraPathMask, filename), cv2.IMREAD_GRAYSCALE) > 0
        imgGray = cv2.imread(os.path.join(pathgray, filename), cv2.IMREAD_GRAYSCALE)
        """ if isRGB:
            imgGray = cv2.imread(os.path.join(pathrgb, filename))
        else:
            imgGray = cv2.imread(os.path.join(pathgray, filename), cv2.IMREAD_GRAYSCALE) """


        tp_tiles, tp_positions = Tiles.execute(pleuraMask, imgGray, tilesize, tilepercentage, isblackbg)
        fp_tiles, fp_positions = Tiles.execute(nonpleuraMask, imgGray, tilesize, tilepercentage, isblackbg)

        """ if resize != None:
            for i in range(len(tp_tiles)):
                tp_tiles[i] = SplitPleura.resize(tp_tiles, resize)
            for i in range(len(fp_tiles)):
                fp_tiles[i] = SplitPleura.resize(fp_tiles, resize) """

        """ tp_tiles = SplitPleura.resize(tp_tiles, resize)
            fp_tiles = SplitPleura.resize(fp_tiles, resize) """

        del pleuraMask
        del nonpleuraMask
        del imgGray
        return tp_tiles, fp_tiles

    @staticmethod
    def split_train_test(pleura, nonpleura):
        print("len(pleura), len(nonpleura)", len(pleura), len(nonpleura))
        n = len(pleura)
        ntest = int((10/100.0)*n)
        print("n, ntest", n, ntest)
        idx = [i for i in range(n)]
        random.seed(42)
        random.shuffle(idx)

        testpleura = pleura[idx[:ntest]]
        testnonpleura = nonpleura[idx[:ntest]]
        
        trainpleura = pleura[idx[ntest:]]
        trainnonpleura = nonpleura[idx[ntest:]]
        
        return trainpleura, trainnonpleura, testpleura, testnonpleura


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
        limitSampleTrain = args["limitSampleTrain"]
        limitSampleTest = args["limitSampleTest"]
        
        

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
                #print(c, "filename", filename)
                if filename.endswith(".tiff"):
                    aux.append({"filename":filename})

            pool = Pool(processes=cpu_count()-1)
            rss = pool.map(SplitPleura.process, aux)
            pool.close()

        for row in rss:
            tp, fp = row[0], row[1]
            for tpi in tp:
                rs_tp.append(tpi)
            for fpi in fp:
                rs_fp.append(fpi)

        ndir = os.path.join(outputdir, SplitPleura.now())
        SplitPleura.makedir(os.path.join(ndir, "pleura"))
        SplitPleura.makedir(os.path.join(ndir, "nonpleura"))
        
        SplitPleura.makedir(os.path.join(ndir, "train"))
        SplitPleura.makedir(os.path.join(ndir, "test"))
        
        SplitPleura.makedir(os.path.join(ndir, "train", "pleura"))
        SplitPleura.makedir(os.path.join(ndir, "train", "nonpleura"))
        
        SplitPleura.makedir(os.path.join(ndir, "test", "pleura"))
        SplitPleura.makedir(os.path.join(ndir, "test", "nonpleura"))

        sampleDir = os.path.join(ndir, "sample")
        SplitPleura.makedir(sampleDir)
        SplitPleura.makedir(os.path.join(sampleDir, "train", "pleura"))
        SplitPleura.makedir(os.path.join(sampleDir, "train", "nonpleura"))
        SplitPleura.makedir(os.path.join(sampleDir, "test", "pleura"))
        SplitPleura.makedir(os.path.join(sampleDir, "test", "nonpleura"))

        rs_tp = np.array(rs_tp, dtype=np.uint8)
        rs_fp = np.array(rs_fp, dtype=np.uint8)


        #whole
        rs_tp, rs_fp = SplitPleura.randomchosse(rs_tp, rs_fp)
        print("x_pleura", "x_nonpleura", len(rs_tp), len(rs_fp))

        SplitPleura.saveimgarry(rs_tp, os.path.join(ndir, "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(rs_fp, os.path.join(ndir, "nonpleura"), isRGB, isResize)


        #train test
        train_pleura, train_nonpleura, test_pleura, test_nonpleura = SplitPleura.split_train_test(rs_tp, rs_fp)
        SplitPleura.saveimgarry(train_pleura, os.path.join(ndir, "train", "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(train_nonpleura, os.path.join(ndir, "train", "nonpleura"), isRGB, isResize)
        SplitPleura.saveimgarry(test_pleura, os.path.join(ndir, "test", "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(test_nonpleura, os.path.join(ndir, "test", "nonpleura"), isRGB, isResize)


        limitSampleTrain = args["limitSampleTrain"]
        limitSampleTest = args["limitSampleTest"]

        #samples limit
        train_pleura, train_nonpleura = train_pleura[:limitSampleTrain], train_nonpleura[:limitSampleTrain]
        test_pleura, test_nonpleura = test_pleura[:limitSampleTest], test_nonpleura[:limitSampleTest]
        SplitPleura.saveimgarry(train_pleura, os.path.join(sampleDir, "train", "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(train_nonpleura, os.path.join(sampleDir, "train", "nonpleura"), isRGB, isResize)
        SplitPleura.saveimgarry(test_pleura, os.path.join(sampleDir, "test", "pleura"), isRGB, isResize)
        SplitPleura.saveimgarry(test_nonpleura, os.path.join(sampleDir, "test", "nonpleura"), isRGB, isResize)

        #print("rs_tp.shape", rs_tp.shape)
        print("--- %s rois extraction ---" % (time.time() - start_time))
        print("pleura", "nonpleura", len(rs_tp), len(rs_fp))
        np.save(os.path.join(ndir, 'pleura.npy'), rs_tp)
        np.save(os.path.join(ndir, 'nonpleura.npy'), rs_fp)

        SplitPleura.Xy(ndir)
        SplitPleura.train_test(ndir)

        SplitPleura.write(os.path.join(ndir, 'config.json'), args)

    @staticmethod
    def write(file, obj):
        with open(file, "w") as filef:
            filef.write(ujson.dumps(obj))


    @staticmethod
    def saveimgarry(arr, path, isRGB, isResize):
        print("arr.shape", arr.shape)
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
    
    #slide = [50, 100, 200, 300, 500]
    
    """ 
    {"tilesize":100, "isResize":None, "tilepercentage":0.1},
    {"tilesize":200, "isResize":None, "tilepercentage":0.1},
    {"tilesize":300, "isResize":224, "tilepercentage":0.1},
    {"tilesize":400, "isResize":224, "tilepercentage":0.1},
    """

    """ 
                {"tilesize":100, "isResize":None, "tilepercentage":0.1},
                {"tilesize":300, "isResize":None, "tilepercentage":0.1},
                {"tilesize":300, "isResize":None, "tilepercentage":0.1},
                {"tilesize":400, "isResize":None, "tilepercentage":0.1},
                {"tilesize":500, "isResize":None, "tilepercentage":0.1}, """


    
    slide = [
                #{"tilesize":50, "isResize":None, "tilepercentage":0.1},
                #{"tilesize":100, "isResize":None, "tilepercentage":0.1},
                {"tilesize":200, "isResize":None, "tilepercentage":0.1},
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
                "limitSampleTrain":9000,
                "limitSampleTest":1000
                }

        sp = SplitPleura.execute(args)
        del sp
    

    #pathimg = "/mnt/sda6/software/frameworks/data/radpleura/lung/2022/04/6262c73a3ff8bc1a5f5af31b/"
    #SplitPleura.readTilesFrom_nrrd(pathimg,500)    

    