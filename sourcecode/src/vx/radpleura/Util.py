#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Ivar
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ujson
import copy

from datetime import datetime

class Util:
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

    @staticmethod
    def now():
        return datetime.now().strftime("%Y%m%d%H%M%S")

    @staticmethod
    def makedir(ndir):
        if not os.path.exists(ndir):
            os.makedirs(ndir)

    @staticmethod
    def makeheatmap(id, inputDir, outputDir, fileclass):
        classdata = Util.read(inputDir+"/"+fileclass)
        index = []
        #columns = ["KNN","SVCGRID","SVCLINEAR","SVC","DTC","RFC","MLPC","ADBC","GNBC"]
        columns = []
        data = []
        for row in classdata:
            r = []
            columns = []
            #print("row[evals]", row["evals"])
            for k, v in row["evals"].items():
                r.append(v["metrics"]["f1"])
                columns.append(k)
            data.append(r)
            name = row["parameters"]["train"]["label"] +" "+ row["name"]
            
            if row["norm"]=="None":
                index.append(name+" (0)")
            elif row["norm"]=="std":
                index.append(name+" (1)")
            elif row["norm"]=="minmax":
                index.append(name+" (2)")

        df = pd.DataFrame(data, index=index, columns=columns)

        plt.subplots(figsize=(8,15))
        color_map = plt.cm.get_cmap('YlOrBr_r')
        #color_map = color_map.reversed()

        ax = sns.heatmap(df, cmap=color_map, square=True, annot=True, annot_kws={"size":6})

        for item in ax.get_yticklabels():
            item.set_rotation(0)

        for item in ax.get_xticklabels():
            item.set_rotation(90)
        fileout = f'classification_{id}.pdf'
        plt.savefig(outputDir+'/'+fileout, dpi=100, bbox_inches='tight')
        plt.close("all")

    @staticmethod
    def makebar(id, inputDir, outputDir, fileclass):
        classdata = Util.read(inputDir+"/"+fileclass)
        
        index = []
        #columns = ["KNN","SVCGRID","SVCLINEAR","SVC","DTC","RFC","MLPC","ADBC","GNBC"]
        data = []
        for row in classdata:
            for k, v in row["evals"].items():
                ac = v["metrics"]["f1"]
                #ac = row["evals"][c]["acc"]
                data.append(ac)
                name = row["parameters"]["train"]["label"] +" "+ row["name"]
                if row["norm"]=="None":
                    name += " (0)"
                elif row["norm"]=="std":
                    name += " (1)"
                elif row["norm"]=="minmax":
                    name += " (2)"
                name = k+" "+name
                index.append(name)

        data, index = zip(*sorted(zip(data, index), reverse=True))
        plt.subplots(figsize=(5,60))
        df = pd.DataFrame({"lab":index,"val":data})

        ax = sns.barplot(x = 'val', y = 'lab', data = df,  color='#0091eb')

        for x_ticks in ax.get_xticklabels():
            x_ticks.set_rotation(90)

        i = 0
        for p in ax.patches:
            ax.annotate(format(data[i], '.2f'), 
                        (p.get_x() + p.get_width(), p.get_y()+1), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 5), 
                        textcoords = 'offset points')
            i+=1
        fileout = f'bars_{id}.pdf'
        plt.savefig(outputDir+'/'+fileout, dpi=100, bbox_inches='tight')
        plt.close("all")


    @staticmethod
    def XXsplitImage(image, tileSize):
        height, width = image.shape
        # print(image.shape)

        tiles = []
        positions = []
        maxMultHeight = height - (height % tileSize)
        maxMultWidth = width - (width % tileSize)
        # print(maxMultHeight, maxMultWidth)
        for i in range(0, maxMultHeight, tileSize):
            for j in range(0, maxMultWidth, tileSize):
                # yield image[i:i+tileSize, j:j+tileSize]
                positions.append(np.asarray((i, i + tileSize, j, j + tileSize)))
                tiles.append(image[i:i + tileSize, j:j + tileSize])
                # print(image[i:i+tileSize, j:j+tileSize])

        lastTile = image[maxMultHeight:height, maxMultWidth:width]
        if lastTile.shape[0] > 0 and lastTile.shape[1] > 0:
            tiles.append(lastTile)
            positions.append(np.asarray((maxMultHeight, height, maxMultWidth, width)))
        #print(tiles)
        return tiles, positions

    def splitImage(image, tileSize):
        height, width = image.shape
        # print(image.shape)

        tiles = []
        positions = []
        maxMultHeight = height - (height % tileSize)
        maxMultWidth = width - (width % tileSize)
        # print(maxMultHeight, maxMultWidth)
        for i in range(0, height, tileSize):
            for j in range(0, width, tileSize):
                # yield image[i:i+tileSize, j:j+tileSize]
                aux_i = i + tileSize
                ls_i = aux_i if aux_i<(height-1) else height-1
                
                aux_j = j + tileSize
                ls_j = aux_j if aux_j<(width-1) else width-1
                
                positions.append(np.asarray((i, ls_i, j, ls_j)))
                tiles.append(image[i:ls_i, j:ls_j])
                # print(image[i:i+tileSize, j:j+tileSize])

        #lastTile = image[maxMultHeight:height, maxMultWidth:width]
        #if lastTile.shape[0] > 0 and lastTile.shape[1] > 0:
        #    tiles.append(lastTile)
        #    positions.append(np.asarray((maxMultHeight, height, maxMultWidth, width)))

        return tiles, positions

    @staticmethod
    def getFileName(arg):
        #print(arg["targetSet"],arg["boundaryDataSet"],arg["name"], arg["label"])
        #return arg["targetSet"]+"_"+arg["boundaryDataSet"]+"_"+arg["name"]+"_"+arg["label"]+".csv"


        return arg["targetSet"]+"_"+arg["name"]+"_"+Util.getLabel(arg["parameters"])+".csv"

        #return arg["targetSet"]+"_"+arg["name"]+"_"+arg["boundaryDataSet_id"]+str(arg["parameters"]["tile_size"])+"_"+arg["label"]+".csv"

    @staticmethod
    def getLabel(arg):
        d = []
        for k, v in arg.items():
            if type(v) == list:
                s = [str(a) for a in v]
                s = ",".join(s)
                d.append(str(k)+":["+str(s)+"]")
            else:
                d.append(str(k)+"_"+str(v))

        #d = "{"+" ".join(d)+"}"
        d = "_".join(d)
        return d


    @staticmethod
    def curvePlot(dat):
        Util.makedir(dat["outputdir"])

        #dat["inputdir"]
        #dat["outputdir"]
        #dat["files"]
        #dat["metric"]
        dato = {}
        dato_aux = []
        #filres = []
        for name, fil in dat["files"].items():            
            #filres.append(Util.read(fil))
            dato[name] = {}
            dato_aux
            obj = Util.read(dat["inputdir"]+"/"+fil)
            #print("obj",obj)
            for row in obj:
                #print("EE",row["evals"])
                for k, v in row["evals"].items():
                    #print("row", v["metrics"][dat["metric"]])
                    dato[name][row["xval"]] = v["metrics"][dat["metric"]]
                    
                    metrics = {}
                    metrics["name"] = name
                    metrics["xval"] = row["xval"]
                    for kk, vv in v["metrics"].items():
                        metrics[kk] = vv

                    dato_aux.append(metrics) 

        print(dato_aux)
        df_aux = pd.DataFrame(dato_aux)
        df_aux.to_csv(dat["outputdir"]+"/"+dat["filename"]+"_info.csv")

        index = []
        curves = {}
        for k, v in dato.items():
            curves[k] = []
            index = v.keys()
            for kv in index:
                curves[k].append(v[kv]) 

        #print(obj["evals"])
        #print(curves)
        #print(index)

        df = pd.DataFrame(curves, index=index)
        df.to_csv(dat["outputdir"]+"/"+dat["filename"]+".csv")

        lines = df.plot.line(figsize=[5,3])
        plt.xlabel(dat["xlabel"])
        plt.ylabel(dat["ylabel"])
        if len(dat["ylim"])==2:
            plt.ylim(dat["ylim"])
        plt.xticks([i for i in index],[str(i) for i in index])
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--')

        fig = lines.get_figure()
        fig.savefig(dat["outputdir"]+"/"+dat["filename"]+".pdf", dpi=300, bbox_inches='tight')


    @staticmethod
    def curvePlotFromCSV(dat):
        dfin = pd.read_csv(dat["inputdir"]+"/"+dat["file"]) 
        print("dfin", dfin)
        index = dfin["ID"].tolist()
        dfin = dfin.drop(["ID"], axis=1)

        lines = dfin.plot.line(figsize=[5,3])

        plt.xlabel(dat["xlabel"])
        plt.ylabel(dat["ylabel"])
        
        if len(dat["ylim"])==2:
            plt.ylim(dat["ylim"])
        plt.xticks([i for i in range(len(index))],[str(i) for i in index])
        if "legendloc" in dat:
            plt.legend(loc=dat["legendloc"])
        else:
            plt.legend(loc='lower right')
        plt.grid(True, linestyle='--')

        if "islogy" in dat and dat["islogy"]==True:
            plt.yscale('log')

        fig = lines.get_figure()
        fig.savefig(dat["outputdir"]+"/"+dat["filename"]+".pdf", dpi=300, bbox_inches='tight')


    @staticmethod
    def curvePlotFromDIR(dat):
        Util.makedir(dat["outputdir"])
        dato = {}
        dato_aux = []
        for name, fil in dat["files"].items():            
            dato[name] = {}
            for di in range(dat["from"], dat["to"]+1, dat["increment"]):
                obj = Util.read(dat["inputdir"]+"/"+str(di)+"/"+fil)
                for row in obj:
                    for k, v in row["evals"].items():
                        xxx = di
                        dato[name][xxx] = v["metrics"][dat["metric"]]
                        
                        metrics = {}
                        metrics["name"] = name
                        metrics["xval"] = xxx
                        for kk, vv in v["metrics"].items():
                            metrics[kk] = vv

                        dato_aux.append(metrics) 

        print(dato_aux)
        df_aux = pd.DataFrame(dato_aux)
        df_aux.to_csv(dat["outputdir"]+"/"+dat["filename"]+"_info.csv")

        index = []
        curves = {}
        for k, v in dato.items():
            curves[k] = []
            index = v.keys()
            for kv in index:
                curves[k].append(v[kv]) 

        print("curves", curves)
        df = pd.DataFrame(curves, index=index)
        df.to_csv(dat["outputdir"]+"/"+dat["filename"]+".csv")

        lines = df.plot.line(figsize=[5,3])
        if "weight" in dat and "height" in dat:
            lines = df.plot.line(figsize=[dat["weight"], dat["height"]])

        plt.xlabel(dat["xlabel"])
        plt.ylabel(dat["ylabel"])
        if len(dat["ylim"])==2:
            plt.ylim(dat["ylim"])
        
        plt.xticks([i for i in index],[str(i) for i in index])
        if "xinterval" in dat:
            plt.xticks(np.arange(min(index), max(index)+1, dat["xinterval"]))

        if "xrotation" in dat:
            plt.xticks(rotation=dat["xrotation"])
        if "xticksfontsize" in dat:
            plt.xticks(fontsize=dat["xticksfontsize"])

        
            
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--')

        fig = lines.get_figure()
        fig.savefig(dat["outputdir"]+"/"+dat["filename"]+".pdf", dpi=300, bbox_inches='tight')

        

    @staticmethod
    def makeConfigureFormUtil(dat):
        #dat = dat[0]
        #print(dat)
        dao = []
        a = dat["fromUtil"]["limits"][0]
        b = dat["fromUtil"]["limits"][1]
        c = dat["fromUtil"]["limits"][2]
        
        for idx in range(a, b+c, c):
            dat_copy = copy.deepcopy(dat)
            dat_copy["outputdir"] = dat_copy["outputdir"]+"/"+str(idx)
            dat_copy["featureselection"]["n_features"] = idx
            
            dao.append(dat_copy)
        
        return dao 