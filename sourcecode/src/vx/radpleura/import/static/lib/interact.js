/*
# Author: Ivar Vargas Belizario
# Copyright (c) 2021
# E-mail: ivar@usp.br
*/




/**
 * Fuanção para selecionar as regioes que o 
 * usuario por meio da interação de um polygon
 */
function chooseRegioes(dat) {
    regs_selected = []
    
    //var self = this;
    regions = dat["regions"];
    regions_map = dat["regions_map"];
    regions_map_sel = dat["regions_map_selected"];
    lassoPolygon = dat["user_polygon"];
    //explorar os clusters de regioes selecionadas pelo usuario
    for(cluser in regions_map_sel){
        //para cada cluster
        for( r of regions_map[cluser]){
            //para cada regiao do cluster r

            reg = regions[r];
            coutp = reg["points"].length;
            coutv = 0;
            //para ponto da regiao expresada como poligono
            for( pxy of reg["points"]){
                /* 
                var xx = (parseFloat(pxy.x) * self.transform.k) + self.transform.x;
                var yy = (parseFloat(pxy.y) * self.transform.k) + self.transform.y; 
                */    
                var xx = parseFloat(pxy[0]);
                var yy = parseFloat(pxy[1]);
                point = [xx, yy];
                //verifcar se os pontos das regioes estao dentro do polygono que crio o usuario
                if (pointInPolygon(point, lassoPolygon)) {
                    coutv++;
                }
            }
            //se todos os ponto da região estão dentro do polygono do usuario
            //marcar como selecionado
            if (coutp==coutv){
                regs_selected.push(reg["id"]);
            }
        }
    }
    return regs_selected;
}

regions =   [
                {"id":0,"class":0,"points":[[0,1],[10,10],[20,25],[10,35]]},
                {"id":1,"class":0,"points":[[0,10],[10,100],[20,250],[10,350]]}
            ];
regions_map = [[0],[1]];
regions_map_selected = [0,1];
user_polygon = [[0.0,0.0],[100.0,0.0],[100.0,200.0],[0,200.0]];

dat = {
        "regions":regions,
        "regions_map":regions_map,
        "regions_map_selected":regions_map_selected,
        "user_polygon":user_polygon
    };

restult = chooseRegioes(dat);
console.log("Holax",restult);
