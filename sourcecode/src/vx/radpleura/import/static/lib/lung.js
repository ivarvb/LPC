/*
# Author: Ivar Vargas Belizario
# Copyright (c) 2021
# E-mail: ivar@usp.br
*/


function showloading() {
    gelem("idloading").style.display = "block";
}
function hideloading() {
    gelem('idloading').style.display = "none";
}



function ServiceData(pname) {
    var self = this;
    this.in = {"argms": {"type": 0}};
    this.ou = '';
    this.event = function () { };
    this.start = function () {
        var ps = MOPRO.pushprocess(pname);
        try {
            var url = "./query?data=" + JSON.stringify(self.in);
            d3.json(url).then(function(data){
                self.ou = data;
                self.event();
                MOPRO.popprocess(ps);
            });
        }
        catch (err) {
            MOPRO.popprocess(ps);
            console.log(err);
        }
    };
}









function lung() {
    var self = this;

    this.mw = new ModalWindow();
    //console.log("this.mw", self.mw);
    this.isnamedbedited = true;
    this.homepath = "/";
    this.path = "";
    this.file = "";
    this.stack = Array(100).fill("");
    this.stacki = 0;
    this.width = 100;
    this.height = 100;
    this.data = [];
    this.contours = [];
    this.selectroids = [];
    this.pjpath = null;
    this.idpj = null;
    this.labels = null;
    this.ypredicted = null;
    //this.ypredicted = null;
    //this.ypredicted = null;
    this.models = {   
        "SVMC500":{"tile":500,"type":"ML"},
        "XGBC500":{"tile":500,"type":"ML"},
        "ResNet_odspleura200tile1black":{"tile":200,"type":"DL"},
    }

    this.setToolpiltex = function (ex, ey, txt) {
        gelem("toolpiltex").innerHTML = "";
        gelem('toolpiltex').style.left = (ex + 2) + "px";
        gelem('toolpiltex').style.top = (ey - 23) + "px";
        gelem('toolpiltex').style.display = "block";
        gelem("toolpiltex").innerHTML = txt;
    };

    this.hideToolpiltex = function () {
        gelem("toolpiltex").style.display = "none";
        gelem("toolpiltex").innerHTML = "";
    };

    this.opengallery = function(){
        gelem("lyimgpanel").style.display = "none";
        gelem("lylistpanel").style.display = "block";

        gelem("lylistview").innerHTML = "";
        var ob = new ServiceData("load gallery");
        ob.in.argms["type"] = 8;
        ob.event = function () {
            self.data = this.ou["response"];
            console.log("rexxxxr", self.data);
            self.listimagestable();
            //self.statusthread();
        };
        ob.start();
    };
    
    this.listimagestable = function(){
        var txtx = "";
        txtx = `
        <div class="tableFixHead">
        <table class="tablex2 table-striped table-sm btn-table" style="background-color: #fff; width: 100%;">
            <thead>
            <tr>
                <th style="text-align:left">Name</th>
                <th style="width: 70px;">Status</th>
                <th style="width: 120px;">Update</th>
                <th style="width: 120px;">Actions</th>
            <tr>
            </thead>
            <tbody>
        `;
        for(var i in self.data){
            re = self.data[i];
            console.log("rex3r", re);
            projectid = re["y"]+"/"+re["m"]+"/"+re["_id"];
            //txtx += `<img src="data/`+re[i]["name"]+`" style="height: 80px; margin: 5px;"
            //            onclick="SCB.chosseimage('data/`+re[i]["name"]+`')";
            //        >`;
            txtx += `
            <tr onclick="
                SCB.openproject('`+projectid+`','`+re["_id"]+`');
                "

                title = "`+projectid+`"
            >
                <td class="align-middle"  style="text-align:left">
                    `+re["name"]+`
                </td>
                <td class="align-middle">`;

            if (re["status"]==1){
                txtx += `<div>
                <i class="fa fa-check-circle" style="color: #0059b3"></i>
                </div>`;
            }
            else{
                txtx += `<div>
                <i class="fas fa-cog fa-spin" style="color: #ff0000"></i>
                </div>`;
            }   
            txtx += `
                </td>
                <td class="align-middle">
                    `+re["date_update"]+`
                </td>
                <td class="align-middle">
                    <a href="#" class="btn btn-light" style="padding: 2px;" title="Downlod dataset" 
                        onclick="
                            //xxx.downloaddataset('5f52b46265aed74204758191');
                        "
                    >
                        <i class="fa fa-download fa-lg" style="color: #f5a742;"></i>
                    </a>
                    <a href="#" class="btn btn-light" style="padding: 2px;" title="Drop dataset"
                        onclick="
                            //xxx.dropdataset('5f52b46265aed74204758191');
                        "
                    >
                        <i class="fa fa-trash fa-lg" style="color: #ff0000;"></i>
                    </a>
                </td>
            </tr>
            `;
        }
        txtx += `
        </tbody>
        </table>
        </div>
        <div style="height: 65px;"></div>
        `;
        gelem("lylistview").innerHTML = txtx;            
        gelem("idimginfo").innerHTML = "";
    };

    this.opendirectory = function(pathin, direcin){
/*         gelem("lyimgpanel").style.display = "none";
        gelem("lylistpanel").style.display = "block"; */

        gelem("lypaneldirsfiles").innerHTML = "";
        
        var ob = new ServiceData("load gallery");
        ob.in.argms["type"] = 2;
        ob.in.argms["path"] = pathin;
        ob.in.argms["directory"] = direcin;

        ob.event = function () {
            var re = this.ou["response"];
            var err = this.ou["error"];
            //console.log("ss",err);
            if (err==1){
                alert(re);
                return 0;
            }
            path = re["path"];
            files = re["files"];

            self.stack[self.stacki] = path;
            self.stacki++;
    
            gelem("inputpathdir").value = path;

            txtx = `
            <div class="tableFixHead">
            <table class="table table-striped table-sm btn-table" style="background-color: #fff; width: 100%;">
                <thead>
                <tr>
                    <th style="text-align:left">Name</th>
                    <th style="width: 120px;">Modified</th>
                <tr>
                </thead>
                <tbody>
            `;
            for (i in files){
                fi = files[i]
                if(fi["type"]==1){
                    txtx += `
                    <tr 
                        ondblclick="
                            gelem('inputFilevsival').value = '`+fi["name"]+`';
                            SCB.showlayout('frmvsi');
                        "
                    >
                    <td class="align-middle"  style="text-align:left">
                        <i class="fas fa-file-alt" style="color: #333;"></i>
                        &nbsp;`+fi["name"]+`</div>
                    </td>
                    <td class="align-middle"  style="text-align:right">
                        `+fi["date"]+`
                    </td>
                    </tr>`;
                }
                else{
                    txtx += `
                    <tr 
                        ondblclick="
                            SCB.opendirectory('`+path+`','`+fi["name"]+`');
                        "
                        title="`+path+`"
                    >
                    <td class="align-middle"  style="text-align:left">
                        <i class="fa fa-folder" style="color: #256cb8;"></i>
                        &nbsp;`+fi["name"]+`
                    </td>
                    <td class="align-middle"  style="text-align:right">
                        `+fi["date"]+`
                    </td>
                    </tr>`;
                }                
            }
            txtx += "</tbody></table></div>"
            gelem("lypaneldirsfiles").innerHTML = txtx;
        };
        ob.start();
    };

    this.goBack = function(){
        self.stacki = self.stacki-2;
        if (self.stacki< 0){
            self.stacki = 0;
            self.stack[self.stacki] = self.homepath;
        }
        //console.log("EEE",self.stack[self.stacki], self.stacki, self.stack);
        //self.opendirectory(self.stack[self.stacki],'');
    };

    this.showlayout = function(op){
        gelem("lyfrminputpanel").style.display = "none";
        gelem("lyfrmuploadpanel").style.display = "none";
        gelem("lyfileopen").style.display = "none";
        gelem("lyimgpanel").style.display = "none";
        gelem("lylistpanel").style.display = "none";
        gelem("lyhelppanel").style.display = "none";

        if (op=="gallerylist"){
            gelem("lylistpanel").style.display = "block";
        }
        else if (op=="galleryimg"){
            gelem("lylistpanel").style.display = "block";
        }
        else if (op=="frmvsi"){
            gelem("lyfrminputpanel").style.display = "block";
        }
        else if (op=="frmupload"){
            gelem("lyfrmuploadpanel").style.display = "block";
        }
        else if (op=="openfile"){
            gelem("lyfileopen").style.display = "block";
        }
        else if (op=="lyhelppanel"){
            gelem("lyhelppanel").style.display = "block";
        }
        

    };

    /*  
    this.opendirectory = function(pathin, direcin){

        gelem("lypaneldirsfiles").innerHTML = "";
        
        var ob = new ServiceData("load gallery");
        ob.in.argms["type"] = 2;
        ob.in.argms["path"] = pathin;
        ob.in.argms["directory"] = direcin;

        ob.event = function () {
            var re = this.ou["response"];
            var err = this.ou["error"];


            console.log("ss xxx ",re);
            if (err==1){
                alert(re);
                return 0;
            }
            path = re["path"];
            files = re["files"];

            self.stack[self.stacki] = path;
            self.stacki++;
    
            gelem("inputpathdir").value = path;

            txtx = `
            <div class="tableFixHead">
            <table class="table table-striped table-sm btn-table" style="background-color: #fff; width: 100%;">
                <thead>
                <tr>
                    <th style="text-align:left">Name</th>
                    <th style="width: 120px;">Modified</th>
                <tr>
                </thead>
                <tbody>
            `;
            for (i in files){
                fi = files[i]
                if(fi["type"]==1){
                    txtx += `
                    <tr 
                        ondblclick="
                            SCB.setChooseFile(\``+path+`\`,\``+fi["name"]+`\`);
                        "
                    >
                    <td class="align-middle"  style="text-align:left">
                        <i class="fas fa-file-alt" style="color: #333;"></i>
                        &nbsp;`+fi["name"]+`</div>
                    </td>
                    <td class="align-middle"  style="text-align:right">
                        `+fi["date"]+`
                    </td>
                    </tr>`;
                }
                else{
                    txtx += `
                    <tr 
                        ondblclick="
                            SCB.opendirectory(\``+path+`\`,\``+fi["name"]+`\`);
                        "
                    >
                    <td class="align-middle"  style="text-align:left">
                        <i class="fa fa-folder" style="color: #256cb8;"></i>
                        &nbsp;`+fi["name"]+`
                    </td>
                    <td class="align-middle"  style="text-align:right">
                        `+fi["date"]+`
                    </td>
                    </tr>`;
                }                
            }

            txtx += "</tbody></table></div>"
            gelem("lypaneldirsfiles").innerHTML = txtx;
        };
        ob.start();
    };
     */

    this.setChooseFile = function(path,file){
        self.path = path;
        self.file = file;
        gelem('idinputFilevsival').value = file;
        //console.log(self.path, self.file);
        self.showlayout('frmvsi');
    };
    this.createimgpfromvsi = function(){
        var name = trim(gelem('idFileName').value);
        var factor = (gelem('idnumberfactor').value);

        if (name==""){
            ffocus('idFileName');
            return;
        }
        if (self.path=="" || self.file==""){
            ffocus('idinputFilevsival');
            return;
        }
/*         if (factor>0){
            ffocus('idnumberfactor');
            return;
        } */
            
        var ob = new ServiceData("load gallery");
        ob.in.argms["type"] = 3;
        ob.in.argms["name"] = name;
        ob.in.argms["path"] = self.path;
        ob.in.argms["file"] = self.file;
        ob.in.argms["factor"] = factor;

        ob.event = function () {
            var re = this.ou["response"];
            //console.log("re",re);
            self.opengallery();
            self.showlayout('gallerylist');
        };
        ob.start();
    };

    this.createimgpfromupload = function(){
        gelem("lypaneldirsfiles").innerHTML = "";
                
        var ob = new ServiceData("load gallery");
        ob.in.argms["type"] = 4;
        ob.in.argms["name"] = pathin;
        ob.in.argms["vsifile"] = pathin;
        ob.in.argms["factor"] = direcin;

        ob.event = function () {
            var re = this.ou["response"];

            gelem("lypaneldirsfiles").innerHTML = txtx;
        };
        ob.start();
    };

    // xxxxxxxxxxxx xxxxxxxxxxxxxx
    this.load_roids_as_polygons = function(path){
        console.log("print roid howss")
        modelsel = self.models[gelem('idmodel').value]

        var ob = new ServiceData("load roids");
        ob.in.argms["type"] = 5;
        ob.in.argms["path"] = path;
        ob.in.argms["tile"] = modelsel["tile"];

        ob.event = function () {
            //var re = this.ou["response"];
            if (this.ou["error"]==0){
                console.log("poligon....",this.ou["response"]);
                self.selectroids = {};
                self.contours = this.ou["response"];
                //DRW.setContours(this.ou["response"]);
                DRW.drawsegmentation();
            }
        };
        ob.start();
    };

    /* 
    this.statusthread = function(){
        let ob = new ServiceData("load gallery validation");
        ob.in.argms["type"] = 1;
        ob.in["lo"] = false;
        ob.event = function () {
            datax = this.ou["response"];
            console.log("hoxxlax", this.in["lo"]);
            out = false;
            for(i in datax){
                self.data[i] = datax[i];
                if (self.data[i]["atributes"]["status"]==0){
                    out = true;
                }
            }
            if(out){
                setTimeout(function () { self.statusthread(); }, 4000);
            }
            else{
                //self.data = datax;
                self.listimagestable();
            }
        };
        ob.start();        
    };
    */

    this.fullScreen = function () {
        var element = document.documentElement;

        if (element.requestFullscreen) {
            element.requestFullscreen();
        } else if (element.mozRequestFullScreen) {
            element.mozRequestFullScreen();
        } else if (element.webkitRequestFullscreen) {
            element.webkitRequestFullscreen();
        } else if (element.msRequestFullscreen) {
            element.msRequestFullscreen();
        }
    };

    this.uploadfiledata = function () {
        var fi = document.getElementById('fileu');
        var file = fi.value;
        var reg = /(.*?)\.(TIFF|tiff)$/;

        //console.log("self.mv", self.mw);
        if (file.match(reg)) {
            var fsize = fi.files.item(0).size;
            var z = Math.round((fsize / 1024));
            if (z <= 71680) {
                let ob = new ServiceData("Upload image");
                ob.in.argms["type"] = 6;
        
                gelem('datafromupload').value = JSON.stringify(ob.in);

                //showloading
                MOPRO.show("Uploading file");
                
                gelem('formupload').submit();        
                gelem('fileu').value = "";        
            }
            else{
                self.mw.alert("","please use files up to 70MB");
            }
        }
        else{
            self.mw.alert("","Please use only .tiff files");
        }
        fi.value = "";
    };

    this.fullScreen = function () {
        var isInFullScreen = (document.fullscreenElement && document.fullscreenElement !== null) ||
        (document.webkitFullscreenElement && document.webkitFullscreenElement !== null) ||
        (document.mozFullScreenElement && document.mozFullScreenElement !== null) ||
        (document.msFullscreenElement && document.msFullscreenElement !== null);

        if (!isInFullScreen) {
            self.setfullScreen();
        } else {
            self.exitfullScreen();
        }
    };

    this.setfullScreen = function () {
        var docElm = document.documentElement;
        if (docElm.requestFullscreen) {
            docElm.requestFullscreen();
        } else if (docElm.mozRequestFullScreen) {
            docElm.mozRequestFullScreen();
        } else if (docElm.webkitRequestFullScreen) {
            docElm.webkitRequestFullScreen();
        } else if (docElm.msRequestFullscreen) {
            docElm.msRequestFullscreen();
        }
    };
    this.exitfullScreen = function () {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.mozCancelFullScreen) {
            document.mozCancelFullScreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
    };

    this.updatedatasetname = function (e, newname) {
        newname = trim(newname);
        if (e.keyCode === 13) {

            if (self.isnamedbedited){
                var txtnamedb = newname;
                txtnamedb += `&nbsp;<i class="fas fa-edit fa-sm"></i>`;
                gelem('iddatasettitle').innerHTML = txtnamedb;
                gelem('layoutchangenametxtdb').style.display = 'none';
                gelem('iddatasettitle').style.display = 'block';
    
            }

/*             if (self.datafileselected != "" && newname != "") {
                var ob = new ServiceData("update dataset name");
                ob.in.argms["type"] = 6;
                ob.in.argms["file"] = self.datafileselected;
                ob.in.argms["newname"] = newname;
                ob.event = function () {
                    response = this.ou;
                    self.getdatasetname();
                };
                ob.start();
            } */
        }
    };

    this.makeclassification = function () {
        console.log("self.selectroids.length", self.selectroids);
        if(Object.keys(self.selectroids).length>0){
            var ob = new ServiceData("make classification");
            modelsel = self.models[gelem('idmodel').value]

            ob.in.argms["type"] = 7;
            ob.in.argms["idpj"] = self.idpj;
            ob.in.argms["path"] = self.pjpath;
            ob.in.argms["idroi"] = self.selectroids;
            //ob.in.argms["idmodelversion"] = "001";
            ob.in.argms["idmodel"] = gelem('idmodel').value;
            ob.in.argms["tile"] = modelsel["tile"];
            ob.in.argms["typepredict"] = modelsel["type"];
            

            
            //console.log("self.selectroids", self.selectroids);
            ob.event = function () {
                console.log("this.ou[response]", this.ou["response"]);
                //DRW.setlabel(this.ou["response"]);
                self.labels = this.ou["response"]["labels"];
                self.labels = {"0":"non-pleura", "1":"pleura"};
                self.ypredicted = this.ou["response"]["yp"];
                colors = ["#ff0000", "#00ff00"]
                for (i in self.selectroids){
                    console.log("i", i);
                    idreg = self.selectroids[i];

                    lab = self.ypredicted[i];
                    console.log("lasdfasdççç",self.labels[lab], colors[lab]);
                    DRW.highlight_id(idreg,colors[lab]);
                }


            };
            ob.start();
        }
    };


    this.openproject = function (path,idpj) {
        console.log('ope projettttttttt', path, idpj);
        self.pjpath = path;
        self.idpj = idpj;
        var ob = new ServiceData("make classification");
        ob.in.argms["type"] = 9;
        ob.in.argms["idpj"] = idpj;

        ob.event = function () {
            console.log("EEXX",this.ou);
            //DRW.setlabel(this.ou["response"]);
            dpj = this.ou["response"][0];
            console.log("dpjdpjdpjdpj",dpj, dpj["name"],"XX");
            //gelem("idnameproject").innerHTML = dpj["name"];
            txtnamedb = dpj["name"]+`&nbsp;<i class="fas fa-edit fa-sm"></i>`;
            gelem('iddatasettitle').style.display = "block";
            gelem('iddatasettitle').innerHTML = txtnamedb;
            gelem('idtxtnameproject').value = dpj["name"]            
            console.log("selflung.data", dpj);            

            DRW.chosseimage(path, dpj);

            // LOAD ROIDS
            //console.log('DRW.homepath',path);
            self.load_roids_as_polygons(path);
        };
        ob.start();
    };
    
    this.setToolpiltexHelp = function(ex, ey){
        console.log("ex, ey", ex, ey);
        htmlx = gelem("lyhelppanel").innerHTML;
        self.setToolpiltex(ex, ey, htmlx);
    };
    


    // this.chosseimage("../data/input/oso.jpg");
    // this.chosseimage("../data/input/llama2.jpg");
    this.opengallery();
    this.showlayout("gallerylist");
    //this.opendirectory(this.homepath,'');

    
}

function openprojects() {
    //self.fullScreen();
    gelem('iddatasettitle').style.display = "none";
    gelem('layoutchangenametxtdb').style.display = "none";
    
    SCB.opengallery();
    SCB.showlayout('gallerylist');

}
function mwalert(title, txtbody) {
    //self.fullScreen();
    SCB.mw.alert(title, txtbody);
}

