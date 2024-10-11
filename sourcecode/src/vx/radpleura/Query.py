#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2021
# E-mail: ivar@usp.br


import tornado.ioloop
import tornado.web
import tornado.httpserver

import urllib.request, json 
import requests

import ujson
import glob
import os
import time
import sys

import pandas as pd
import numpy as np
import os.path
import math
import uuid

import zipfile
from io import BytesIO
from datetime import datetime
import threading
import SimpleITK as sitk
from bson.objectid import ObjectId


from vx.com.py.database.MongoDB import *


from vx.radpleura.Settings import *
from vx.radpleura.BaseHandler import *
from vx.radpleura.ROI import *
from vx.radpleura.Features import *
from vx.radpleura.VSI import *
from vx.radpleura.Classification import *
from vx.radpleura.ClassificationDL import *
from vx.radpleura.SplitImage import *


class Query(BaseHandler):

    #Get RequestHandler
    def get(self):
        dat = self.get_argument('data')
        app = ujson.loads(dat)

        #app = DataTransfer()
        #app.load(dat)
        
        obj = ""

        if app["argms"]["type"]==0:
            pass;
        elif app["argms"]["type"]==1:
            obj = self.listimages();
        #elif app["argms"]["type"]==2:
        #    obj = self.listfilesdirs(app["argms"]);
        elif app["argms"]["type"]==3:
            obj = self.makeimgfromvsi(app["argms"]);
        elif app["argms"]["type"]==4:
            obj = None
        elif app["argms"]["type"]==5:
            obj = self.getregions(app["argms"]);
            # print("obj-x",obj);
        elif app["argms"]["type"]==7:
            obj = self.makeclassification(self.current_user, app["argms"]);
            # print("obj-x",obj);
        elif app["argms"]["type"]==8:
            obj = self.listprojects(self.current_user, app["argms"]);
            # print("obj-x",obj);
        elif app["argms"]["type"]==9:
            obj = self.openproject(self.current_user, app["argms"]);




        self.write(obj)
        self.finish()


    #Post RequestHandler
    def post(self):
        dat = self.get_argument('data')
        app = ujson.loads(dat)
        rs = ""
        if self.current_user:
            #print("app.argms", app, self.request.files['fileu'][0])
            if app["argms"]["type"]==6:
                rs = Query.uploadfiledata(self.current_user, self.request.files['fileu'][0]);

        self.write(rs)

        #pass




    # static query methods
    """
    def listimages():
        fileso = []
        for name in os.listdir(Settings.DATA_PATH):
            # print("name", name)
            if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"):
#                fileso.append(str(os.path.join(outdir, str(name))))
                # fileso.append({"name":Settings.IMAGE_PATH+str(name)})
                fileso.append({"name":str(name)})
        return {"response":fileso}
    """

    @staticmethod
    def openFile(pathf):
        dfile = {}
        with open(pathf,'r') as fp:
            dfile = ujson.load(fp)
        return dfile

    @staticmethod    
    def writeFile(pathf, rdata):
        with open(pathf,'w') as fp:
            ujson.dump(rdata, fp)

    @staticmethod
    def listimages():
        fileso = []
        """
        for name in os.listdir(Settings.DATA_PATH):
            if name.endswith(".png") or name.endswith(".jpg") or name.endswith(".jpeg"):
                fileso.append({"name":str(name)})
        """

        ini = 2021
        months = ["01","02","03","04","05","06","07","08","09","10","11","12"]
        now = 2021
        for y in range(ini,now+1):
            for m in months:
                folder = os.path.join(Settings.DATA_PATH,str(y),str(m))
                if os.path.exists(folder):
                    for ide in os.listdir(folder):
                        if os.path.isdir(os.path.join(folder, ide)):
                            fileobj = os.path.join(folder, ide, "db.obj")
                            if os.path.exists(fileobj):
                                dat = Query.openFile(fileobj)
                                #print("dat",dat, fileobj)
                                fileso.append(dat)
                                #fileso[ide] = {"y":y, "m":m, "data":dat}

        #fileso.sort(key=lambda item:item['date'], reverse=True)
        #fileso = sorted(fileso.items(), key=lambda x: x["date"])
        #fileso = sorted(fileso, key=lambda k: k['date']) 
        #print(fileso)
        fileso = sorted(fileso, key = lambda i: (i['date']), reverse=True)

        return {"response":fileso}


    # static query methods
    @staticmethod
    def listfilesdirs(argms):
        path = argms["path"]
        direc = argms["directory"]
        pathi = path
        if direc!="":
            pathi += "/"+direc

        result = []
        #print("path", path)
        #print("direc", direc)
        pathi = os.path.join(path,direc)
        #print("pathii", pathi)
        try:
            for fil in os.listdir(pathi):
                cc = os.path.join(pathi,fil)

                modTimesinceEpoc = os.path.getmtime(cc)
                modificationTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modTimesinceEpoc))
                #print("cc",cc)
                if os.path.isfile(cc):
                    result.append({"name":fil,"type":1,"date":modificationTime})
                else:
                    result.append({"name":fil,"type":0,"date":modificationTime})
            result = sorted(result, key=lambda k: (k['type'], k['name']))
            result = {"response":{"path":pathi,"files":result}, "error":0}

        except FileNotFoundError:
            result = {"response":"FileNotFoundError", "error":1}
        except PermissionError:
            result = {"response":"PermissionError", "error":1}
        except:
            result = {"response":"UndefinedError", "error":1}
        finally:
            #print("Done error checking")
            pass

        return result

    # static query methods
    @staticmethod
    def openproject(iduser, argms):
        idpj = ObjectId(argms["idpj"])
        print("idpj", idpj)
        res = list(MongoDB.find(DBS.DBMEDIA, "app_lung", {"_id": idpj}))        
        for rs in res:
            rs["_id"] = str(rs["_id"])
            rs["_id_user"] = str(rs["_id_user"])

            pf = os.path.join(Settings.DATA_PATH, rs["y"]+"/"+rs["m"]+"/"+rs["_id"]+"/pieces.json");
            if os.path.exists(pf):    
                #print("pahfile", pf)
                rs["pieces"] = Query.openFile(pf)
                #print("roisx", rois)
            rs["pathpieces"] = "data/"+rs["y"]+"/"+rs["m"]+"/"+rs["_id"]+"/pieces/"

        print("rs", res)
        return {"response":res, "error":0}



    # static query methods
    @staticmethod
    def listprojects(iduser, argms):
        #iduser = argms["iduser"]
        iduser = ObjectId(iduser.decode("utf-8"))
        #print("iduser xxxx", iduser)
        #rest = []
        rs = list(MongoDB.aggregate(DBS.DBMEDIA, "app_lung",
                        [
                            {"$lookup":
                                {
                                    "from": "user",
                                    "localField": "_id_user",
                                    "foreignField" : "_id",
                                    "as": "usersUnits",
                                }
                            },
                            {"$match": {
                                        "$or":  [
                                                    {"_id_user": iduser},
                                                    {"shared": 1}
                                                ]
                                        }
                            },
                            {"$project":
                                {   
                                    "_id" : 1,
                                    "_id_user": 1 ,
                                    "name": 1,
                                    "date_update" : 1,
                                    "factor" : 1,
                                    "m" : 1,
                                    "y" : 1,
                                    "shared" : 1,
                                    "status" : 1,
                                    "statusmsg" : 1,
                                    "usersUnits._id" : 1 ,
                                    "usersUnits.name" : 1 ,
                                } 
                            },
                            {
                                "$sort": {
                                    "date_update": -1
                                }
                            }
                        ]
                    ))
        #print("xxxxresx", rs)

        #{"_id_user": iduser}))
        print("zzzzzzsrs", rs)
        for i in range(len(rs)):
            rs[i]["_id"] = str(rs[i]["_id"])
            rs[i]["_id_user"] = str(rs[i]["_id_user"])
            if len(rs[i]["usersUnits"])==1:
                rs[i]["usersUnits"][0]["_id"] = str(rs[i]["usersUnits"][0]["_id"])

            
            #row["rois"] = []

        return {"response":rs, "error":0}




    @staticmethod
    def makedir(outdir):
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    @staticmethod
    def getPathSave(mainpath):
        dt_year = datetime.now().strftime("%Y")
        dt_mont = datetime.now().strftime("%m")
        mpth = os.path.join(mainpath, dt_year, dt_mont)
        Query.makedir(mpth)
        return dt_year, dt_mont, mpth
        
    # static query methods
    @staticmethod
    def makeimgfromvsi(argms):
        name = argms["name"]
        path = argms["path"]
        file = argms["file"]

        factor = argms["factor"]
        #print("CC",name, path, file, factor)
        vsifile = os.path.join(path,file)
        """ pathsave = getdiresave(Settings.DATA_PATH) """

        #convertvsi2img(vsifile, factor, Settings.DATA_PATH, "df3wfsd")
        y, m, idf, pathsave = Query.getPathSave(Settings.DATA_PATH)
        
        fileid = uuid.uuid4().hex

        t = threading.Thread(target=Query.convertvsi2img, args=(vsifile, factor, pathsave, fileid,))
        t.start()
        
        dt_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dbdat = {   
                    "y":y,
                    "m":m,
                    "id":idf,
                    "name":name,
                    "date":dt_string,
                    "image":fileid+".jpg",
                    "tumbail":fileid+".jpg",
                    "atributes":{"factor":factor,"status":0,"statusmsg":"working..."},
                    "images":[]
                }
        """
                    "images":[
                                {
                                    "name":"original",
                                    "date":dt_string,
                                    "image":fileid+".jpg",
                                    "tumbail":fileid+"_tumbail.jpg",
                                    "atributes":{},
                                }
                            ]
        """
        Query.writeFile(os.path.join(pathsave,"db.obj"), dbdat)
        #makeimage(filevis, factor, pathsave)
        
        result = {"response":"ok", "error":0}
        return result


    # get regions
    @staticmethod
    def getregions(argms):
        results = None
        pahfile = os.path.join(Settings.DATA_PATH, argms["path"], str(argms["tile"]), "contours.json")
        #print("pahfile", pahfile)

        try:
            rois = Query.openFile(pahfile)
            #print("roisx", rois)
            results = {"response":rois, "error":0}

        except FileNotFoundError:
            results = {"response":"FileNotFoundError", "error":1}
            print("error file not")
        except PermissionError:
            results = {"response":"PermissionError", "error":1}
            print("permission error")
        except:
            results = {"response":"UndefinedError", "error":1}
            print("error undefined")
        finally:
            #print("Done error checking")
            pass

        return results

    @staticmethod
    def convertvsi2img(vsifile, factor, pathout, outfile):
        outfiletiff = os.path.join(pathout,outfile+".tiff")
        outfilejpg = os.path.join(pathout,outfile+".jpg")
        outtumbailjpg = os.path.join(pathout,outfile+"_tumbail.jpg")
        
        BaseManager.register('VSI', VSI, exposed=['getAux','getnTilesX','getnTilesY'])
        manager = BaseManager()
        manager.start()
        obj = manager.VSI(vsifile, float(factor))
        #print("obj.aux", obj.getAux())

        #obj = VSI(vsifile, float(factor))
        image = VSI.makeimage(obj)
        #image = readVSI(vsifile, float(factor))
        cv2.imwrite(outfiletiff, image)
        cv2.imwrite(outfilejpg, image)


        fileobj = os.path.join(pathout, "db.obj")
        dat = Query.openFile(fileobj)
        dat["atributes"]["status"] = 1
        dat["atributes"]["statusmsg"] = ""

        Query.writeFile(fileobj, dat)



    @staticmethod    
    def uploadfiledata(iduser, file):
        r = """<script>
                parent.mwalert('','Error: upload file');
                parent.openprojects();
                </script>"""

        iduser = ObjectId(iduser.decode("utf-8"))
        
        path = Settings.DATA_PATH
        fname, ext = os.path.splitext(file['filename'])        
        ext = ext.lower()

        ye, mo, path = Query.getPathSave(Settings.DATA_PATH)

        rowdata = {  
            "_id_user":iduser,
            "name":fname,
            "y":ye,
            "m":mo,
            "date_create":Query.now(),
            "date_update":Query.now(),
            "factor":1.0,
            "rois":[],
            "shared":0,
            "status":1,
            "statusmsg":"new data lung...",
        }

        idin = None
        if True:
        #try:
            idin = MongoDB.insert(DBS.DBMEDIA, "app_lung", rowdata)
            idin = str(idin)
            idin = Query.converid(idin)
            rs = list(MongoDB.find(DBS.DBMEDIA, "app_lung", {"_id": idin}))

            for rr in rs:
                tilesize = [200,500]
                tileperce = 0.01

                path = os.path.join(path, str(idin))
                Query.makedir(path)
                Query.savefile(path, file['body'])
                
                SplitImage.execute(path, 500)

                #ROI segmentation
                ROI.execute(path, tilesize, tileperce)
                #Feature description
                for ts in tilesize:
                    Features.execute(path, ts)
                

                r = "<script>parent.openprojects();</script>"
        """
        except Exception as e:
            print("error upload file", e)
            if idin != None:
                Query.dropdataset(os.path.join(Settings.DATA_PATH, ye, mo, str(idin)), idin )
        """

        return r
    
    @staticmethod
    def dropdataset(path, idin):
        #filefe = str(idin)
        #os.system("rm -rf "+path)
        MongoDB.delete(DBS.DBMEDIA, "app_lung", {"_id": ObjectId(idin)})
        r = "<script>parent.openprojects();</script>"
        return {"response":r};



    @staticmethod
    def makeclassification(usid, argms):
        idus = usid
        idpj = argms["idpj"]
        idrois = argms["idroi"]
        idmodel = argms["idmodel"]
        tile = argms["tile"]
        typepredict = argms["typepredict"]

        """"
        datam = {   
                    "SVMC500":{"tile":500,"type":"ML"},
                    "XGBC500":{"tile":500,"type":"ML"},
                    "ResNet_odspleura200tile1black":{"tile":200,"type":"DL"},
                }
        
        dats = datam[idmodel]
        """

        print("argms classs", argms)
        parthquery = os.path.join(Settings.DATA_PATH, argms["path"], str(tile))
        #parthquery = os.path.join(Settings.DATA_PATH, argms["path"])
        
        ypred, labels = [], []        
        if typepredict=="ML":        
            ypred, labels = Classification.predict(parthquery, idmodel, idrois)
        elif typepredict=="DL":
            print("DL")
            #ypred, labels = ClassificationDL.predict(parthquery, idmodel, idrois)
            ypred, labels = Query.shiftClassificationDL(parthquery, idmodel, idrois, argms)
        
        rs = {"yp":ypred, "labels":labels}
        print("rs", rs)

        return {"statusopt":0, "statusval":"", "response":rs}
        #return {"statusopt":0, "statusval":"", "response":[]}
    
    @staticmethod
    def shiftClassificationDL(parthquery, idmodel, idrois, argms):
        ypred, labels = None, None
        if False:
            ypred, labels = ClassificationDL.predict(parthquery, idmodel, idrois)
        else:
            print("XXXXXXargms", argms)
            ddx = """{"argms":{"type":5,"path":\""""+argms['path']+"""\","tile":"""+str(argms['tile'])+"""}}"""
            data = {'argms':{'type':5,'path':argms['path'],'tile':argms['tile']}}
            #data = json.dumps(data)            
            url = "http://localhost:"+str(Settings.PORT)+"/query?data="+ddx
            print("ZZZZZZZurl", url)
            #r = requests.get(url,headers = {'Accept': 'application/json'})
            #print("ZZZZZZZr", r)
            #a = r.json()            

            import urllib.request, json 
            with urllib.request.urlopen(url) as urld:
                data = json.load(urld)
                print("data[response]", data["response"])


            
            """
            with urllib.request.urlopen("http://localhost:{}/{}".format(Settings.PORT, data)) as url:
                data = json.load(url)
                print(data)
            """

            ypred, labels = [],{"0":"nonpleura", "1":"pleura"}
        return ypred, labels

    @staticmethod
    def savefile(path, data):
        pfiletiff = os.path.join(path, "original.tiff")
        pfilejpg = os.path.join(path, "original.jpg")
        #create directory
        output_file = open(pfiletiff, mode="wb")
        output_file.write(data)
        output_file.close()

        image = sitk.ReadImage(pfiletiff)
        sitk.WriteImage(image, pfilejpg)        

    @staticmethod
    def now():
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    @staticmethod
    def converid(idin):
        """ 
        #idin = file
        if Settings.MULIUSER == 1:
            #idin = idin.decode("utf-8");
            idin = ObjectId(idin) """
        idin = ObjectId(idin)
        return idin

    @staticmethod
    def getIdClassifier(namec, tilesize, tiletype):
        MLlist = ["SVMC","XGBC"]
        DLCNNlist = ["ResNet", "MobilNet","DenseNet500","Xception"]
        if namec in MLlist:
            typeModel = "ML"
        elif namec in DLCNNlist:
            typeModel = "CNN"

        return namec+"_"+tiletype+"_"+tilesize+"_"+typeModel
