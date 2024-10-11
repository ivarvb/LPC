#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""

import os
import ujson
from PIL import Image

class SplitImage:
    @staticmethod
    def execute(pathin, pathout, chopsize=500):
        infile = os.path.join(pathin, "original.jpg")
        img = Image.open(infile)
        width, height = img.size
        # Save Chops of original image
        boxes = {
            "chopsize":chopsize,
            "width":width,
            "height":height,
            "boxes":[],
        }
        pathpieces = os.path.join(pathin, "pieces")
        SplitImage.makedir(pathpieces)

        it = 0
        for x0 in range(0, width, chopsize):
            for y0 in range(0, height, chopsize):
                x1 = x0+chopsize if x0+chopsize < width else width
                y1 = y0+chopsize if y0+chopsize < height else height
                key = str(it)+".jpg"
                box = {"image":key, "x":x0, "y":y0, "width":(x1-x0), "height":(y1-y0)}
                boxes["boxes"].append(box)
                print('%s %s' % (infile, box))
                img.crop( (x0, y0, x1, y1) ).save(os.path.join(pathpieces, key), quality=95)
                it += 1
        SplitImage.write(os.path.join(pathin,"pieces.json"), boxes)

    @staticmethod
    def write(file, obj):
        """
        write json obj
        file: write file path
        obj: json obj
        """
        with open(file, "w") as filef:
            filef.write(ujson.dumps(obj))

    @staticmethod
    def read(file):
        data = {}
        with open(file,"r") as filef:
            data = (ujson.load(filef))
        return data

    @staticmethod
    def makedir(ndir):
        if not os.path.exists(ndir):
            os.makedirs(ndir)


if __name__ == "__main__":
    pathin = "/mnt/sda6/software/frameworks/data/media/lung/2022/03/622f1d77a0fb50f52fa93790/"
    SplitImage.execute(pathin, 500)
