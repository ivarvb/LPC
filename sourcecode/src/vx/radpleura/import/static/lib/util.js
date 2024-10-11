/*
# Author: Ivar Vargas Belizario
# Copyright (c) 2021
# E-mail: ivar@usp.br
*/
/* function addslashes(txt) {
    return txt.replace(/(["'])/g, "\\$1");
} */
function gelem(id) {
    return document.getElementById(id);
}
function gvalue(id) {
    return document.getElementById(id).value;
}
function trim(str) {
    return str.replace(/^\s+|\s+$/g, "");
}
function ffocus(v) {
    //gelem(v).focus();
}
function polygonToPath(polygon) {
    return ("M" + (polygon.map(function (d) { return d.join(','); }).join('L')));
}



// from https://github.com/substack/point-in-polygon
function pointInPolygon(point, vs) {
    var xi, xj, i, intersect,
        x = point[0],
        y = point[1],
        inside = false;
    for (var i = 0, j = vs.length - 1; i < vs.length; j = i++) {
        xi = vs[i][0],
            yi = vs[i][1],
            xj = vs[j][0],
            yj = vs[j][1],
            intersect = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}



