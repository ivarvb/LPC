#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Ivar Vargas Belizario
# Copyright (c) 2020
# E-mail: ivar@usp.br


import sys

from vx.radpleura.Server import *

import psutil, os
import signal
import tornado

def signal_handler(signum, frame):
    tornado.ioloop.IOLoop.instance().stop()

""" 
def kill_proc_tree(pid, including_parent=True):    
    parent = psutil.Process(pid)
    for child in parent.get_children(recursive=True):
        child.kill()
    if including_parent:
        parent.kill()
"""

if __name__ == "__main__":
    pid = os.getpid()
    print("pid", pid)
    try:
        Server.execute()
    except KeyboardInterrupt:
        # signal.signal(signal.SIGINT, signal_handler)
        tornado.ioloop.IOLoop.instance().stop()

        """
        pid = pid
        p = psutil.Process(pid)
        p.kill()
        print("me1", pid)

        pid = os.getpid()
        print("me", pid)
        p = psutil.Process(pid)
        p.kill()  #or p.kill()
        #kill_proc_tree(me)
        # quit  
        sys.exit()    
        """


