/**
 * Class: Monitor Process
 */
 function MonitorProcess(selff) {
    var self = this;

    this.auto = false;
    //this.auto = true;
    this.idcont = "idloading";
    this.idtxt = "idloadingtxt";
    this.idstatus = "idloadingload";


    this.show = function(text) {
        gelem(self.idcont).style.display = "block";
        gelem(self.idtxt).innerHTML = text+"";
    }
    this.status = function(load) {
        gelem(self.idstatus).style.width = load+"%";
    }
    this.hide = function () {
        gelem(self.idcont).style.display = "none";
        //gelem(self.idstatus).style.width = "0%";
    }    




    this.processkey = 0;
    this.process = {};
    this.toolstatus = [100, 95, 70, 40, 20, 10, 1];

    this.pushprocess = function(pname) {
        self.processkey++;
        var ps = self.processkey;

        self.process[ps] = pname;

        self.show(pname);
        if (self.auto){
            self.status(self.computestatus());
        }
        //console.log("push: ", ps);

        return ps;
    };

    this.popprocess = function(p) {
        delete self.process[p];
        //console.log("delete: ", p, Object.keys(self.process).length, self.process);
        
        if (self.auto){
            self.status(self.computestatus());
        }

        if (Object.keys(self.process).length == 0) {
            self.hide();
        }
    };

    this.computestatus = function() {
        var size = Object.keys(self.process).length;
        var load = 1;
        if (size<=7){
            load = self.toolstatus[size];
        } 
        return load;
    };


    this.main = function () {
        var win = `
            <div>
                <div style="position: absolute; 
                    width: 360px;
                    min-width:360px;
                    max-width: 360px;
                    top: 40%;
                    left: 50%;
                    /*margin-top: -40px;*/
                    margin-left: -180px;
                    z-index: 20;
                    text-align: center;
                    "
                >

                <div style="
                        background-color: #494882;
                        padding: 6px; 
                        border: 1px solid #453a5e;
                        border-radius: 2px;
                        "
                    >
                        <div id="`+self.idtxt+`"
                            style="
                                z-index: 23;
                                color:#fff;
                                text-align: center;
                                font-size: 12px;
                                vertical-align: bottom;
                            "
                        >
                            Loading ...
                        </div>
                        <div style="height:3px"></div>
                        <div class="progress">
                            <div id="`+self.idstatus+`" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" aria-valuenow="100" aria-valuemin="100" aria-valuemax="100" style="width: 100%">
                            </div>
                        </div>
                    </div>
                </div>
            </div>        
        `;

        var container = document.createElement("DIV");
        container.setAttribute("id",self.idcont);
        container.setAttribute("style", `display: block;
                                    position: fixed;
                                    width: 100%;
                                    height: 100%;
                                    left: 30px;
                                    top: 0;
                                    background: rgba(21, 22, 40, 0.9);
                                    z-index: 10;`);
        container.innerHTML = win;
        
        //removechilds("idmodelcontent");
        //gelem("idmodelcontent").appendChild(container); 
        document.body.appendChild(container);
        self.hide();
    };

    //this.main();
}
