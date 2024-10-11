#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2023
# E-mail: ivar@usp.br

import os
   
class Settings:
    VERSION = 3.0
    DBACCESS = 0 # for MongoDB configuration without user and password
    #DBACCESS = 1 # for MongoDB configuration with user and password
    ROOTID = 0
    DEBUG = True
    MULIUSER = 1
    DIRNAME = os.path.dirname(__file__)
    IMPORT_PATH = os.path.join(DIRNAME, './import')
    DATA_PATH = os.path.join(DIRNAME, '../../../../data/radpleura/lung/')

    COOKIE_SECRET = 'L8LwECiNRxdq2W3NW1a0N2eGxx9MZlrpmu2MEdimlydNX/vt1LM='

    HOST = 'localhost'
    PORT = 8777

    PATHROOT = "http://localhost:"+str(PORT)+"/"

class DBS:
    # Change USER_HERE, PASSWORD_HERE or DB_HERE

     # configure without password
    DBMEDIA = {
        "host":"127.0.0.1",
        "port":"27017",
        "username":"",#USER_HERE
        "password":"",#PASSWORD_HERE
        "database":"radpleura",#DB_HERE
    };

    # # configure with password
    # DBMEDIA = {
    #     "host":"127.0.0.1",
    #     "port":"27017",
    #     "password":"azul",#PASSWORD_HERE
    #     "database":"radpleura",#DB_HERE
    #     "audb":"admin",
    # };
