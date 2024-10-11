#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2020
# E-mail: ivar@usp.br

import tornado.ioloop
import tornado.web
import tornado.httpserver
import ujson
import bcrypt


from vx.com.py.database.MongoDB import *


from vx.radpleura.Settings import *
from vx.radpleura.BaseHandler import *
from vx.radpleura.User import *


class Login(BaseHandler):
    def get(self):
        if Settings.MULIUSER==1:
            self.render("login.html")
        else:
            self.redirect("./")
        return

    def post(self):
        op = int(self.get_argument('option'))

        re = User.login(    self.get_argument('user'),
                            self.get_argument('password') );
        if len(re)==1:
            for r in re:
                uid = str(r['_id'])
                #uid = ""+uid+"".decode("utf-8")
                self.set_secure_cookie("user", uid)
                self.set_secure_cookie("email", r['email'])
                #print("r['adminid']",r['adminid']);
                self.set_secure_cookie("adminid", str(r['adminid']))

                #self.set_secure_cookie("user", uid, expires_days=1)
                #self.set_secure_cookie("email", r['email'], expires_days=1)
            self.redirect("./")
            return
        else:
            self.redirect("./login")
            return

        return


class Logout(BaseHandler):
    def get(self):
        self.clear_cookie('user')
        self.clear_cookie('email')
        self.redirect("./")



class Index(BaseHandler):
    def get(self):
        if not self.current_user:
            self.redirect("./login")
            return
        else:
            #print("self.get_current_email()", self.get_current_email())
            #self.render("index.html",email=self.get_current_email(), pathroot=Settings.PATHROOT)
            self.redirect("./lung")


class Lung(BaseHandler):
    def get(self):
        if not self.current_user:
            self.redirect("./login")
            return
        else:
            self.render("lung.html",email=self.get_current_email())
            #self.render("index.html")


class LungPage(BaseHandler):
    def get(self, lungid):
        self.render("lungpage.html",lungid=lungid, email=self.get_current_email())


class Simil(BaseHandler):
    def get(self):
        if not self.current_user:
            self.redirect("./login")
            return
        else:
            self.render("simil.html",email=self.get_current_email(), pathroot=Settings.PATHROOT)
            #self.render("index.html")

