#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2021
# E-mail: ivar@usp.br


import tornado.ioloop
import tornado.web
import tornado.httpserver

import ujson
import glob
import os
import os.path

from vx.radpleura.Settings import *
from vx.radpleura.BaseHandler import *
from vx.radpleura.ClassificationDL import *
class QueryDL(BaseHandler):
    #Get RequestHandler
    def get(self):
        dat = self.get_argument('data')
        print("dat", dat)
        app = ujson.loads(dat)
       
        obj = ""

        idrois = app["idroi"]
        idmodel = app["idmodel"]
        parthquery = os.path.join(Settings.DATA_PATH, app["path"])

        if app["argms"]["type"]==7:
            dlo = ClassificationDL()
            ypred, labels = dlo.predict(parthquery, idmodel, idrois)
            rs = {"yp":ypred, "labels":labels}
            print("rs", rs)
            obj = {"statusopt":0, "statusval":"", "response":rs}

        self.write(obj)
        self.finish()

    #Post RequestHandler
    def post(self):
        pass
        """
        dat = self.get_argument('data')
        app = ujson.loads(dat)
        rs = ""
        if self.current_user:
            if app["argms"]["type"]==6:
                rs = Query.uploadfiledata(self.current_user, self.request.files['fileu'][0]);

        self.write(rs)
        """
