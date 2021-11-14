function changeFun(evtId, mod){
    var targetId = "#" + evtId + mod
    console.log("changelinks init:" + targetId);
    var lst = document.querySelectorAll(targetId); 
    lst.forEach(function(item) {
        if(item.style.visibility == "visible"){
            item.style.visibility = "hidden"
        }else{
            item.style.visibility = "visible"
        }
    }); 
}
function changeLinks(evtId){
    //called when a variant ellipse is clicked.
    changeFun(evtId, "_1mu")
    lst = document.querySelectorAll("#" + evtId); 
    lst.forEach(function(item) {
        if(item.hasAttribute("data-sequencetag")){
            var vis; 
            if(item.style.visibility == "visible"){
                item.style.visibility = "hidden"
                vis = false
            }else{
                item.style.visibility = "visible"
                vis = true
            }
            seq = item.getAttribute("data-sequencetag")
            var ellipses = document.querySelectorAll("ellipse");
            for (var i = 0; i < ellipses.length; i++) {
                var el = ellipses[i]; 
                id = el.getAttribute("id")
                if(/_enabled/.test(id)){
                    groups = id.match(/(\d+)(\w)_en/)
                    if(seq.charAt(groups[1]-1) == groups[2]){
                        if(vis){
                            el.style.visibility = "visible"
                        }else{
                            el.style.visibility = "hidden"
                        }
                    }
                }
            }
        }
    }); 
}
function changeLineage(evtId, lst){
    changeLinks(evtId)
    console.log("changeLineage " + lst)
    lst.forEach(function(edge) {
        var item = document.querySelector("#lineage" + edge);
        if(item.style.visibility == "visible"){
            item.style.visibility = "hidden"
        }else{
            item.style.visibility = "visible"
        }
    }); 
}
function changeLabel2(evtId, extraLabel, vis){
    var targetId = "#" + evtId + extraLabel
    console.log("ChangeLabel2 " + evtId + extraLabel)
    lst = document.querySelectorAll(targetId); 
    lst.forEach(function(item) {
        item.style.visibility = vis;
    }); 
    lst = document.querySelectorAll(targetId + "_background"); 
    lst.forEach(function(item) {
        item.style.visibility = vis;
    });
}
function showLabel(evtId){
    changeLabel2(evtId, "_label", "visible")
}
function hideLabel(evtId){
    changeLabel2(evtId, "_label", "hidden")
}
function crossReferenceEffects(mutationTag, vis){
    // show/hide labels for the effects, on right side.
    groups = mutationTag.matchAll(/(\w)(\d+)(\w)/gi)
    for(let result of groups){
        //console.log("showMutation:" + result[1] + "," + result[2] + "," + result[3] )
        var targetId = "#effect" + result[2] + result[3] + "_label"; 
        lst = document.querySelectorAll(targetId); 
        lst.forEach(function(item) {
            item.style.visibility = vis;
        }); 
        lst = document.querySelectorAll(targetId + "_background"); 
        lst.forEach(function(item) {
            item.style.visibility = vis;
        });
    }
}
function showLabel2(evtId, labno, mutationTag){
    changeLabel2(evtId, "_label" + labno, "visible")
    crossReferenceEffects(mutationTag, "visible")
}
function hideLabel2(evtId, labno, mutationTag){
    changeLabel2(evtId, "_label" + labno, "hidden")
    crossReferenceEffects(mutationTag, "hidden")
}


function showMutation(evtId, mutationTag){
    console.log("showMutation:" + evtId + " " + mutationTag);
    var paths = document.querySelectorAll("path");
    for (var i = 0; i < paths.length; i++) {
        var el = paths[i]; 
        if(el.hasAttribute("data-mutationtag")){
            //console.log("showMutation:found one! " + evtId + " " + mutationTag);
            mt = el.getAttribute("data-mutationtag")
            id = el.getAttribute("id")
            if(mt.localeCompare(mutationTag) == 0 && id != evtId){
                if(el.style.visibility == "visible"){
                    el.style.visibility = "hidden"
                }else{
                    el.style.visibility = "visible"
                }
            }
        }
    }
    // now need to parse out "what the muation means" so we can match and highlight the nodes
    // (obviously, the endpoints of the labeled mutations have it, but some others do as well)
    var ellipses = document.querySelectorAll("ellipse");
    for (var i = 0; i < ellipses.length; i++) {
        var el = ellipses[i];
        if(el.hasAttribute("data-sequencetag")){
            seq = el.getAttribute("data-sequencetag")
            groups = mutationTag.matchAll(/(\w)(\d+)(\w)/gi)
            var matchAll = true
            for(let result of groups){
                //console.log("showMutation:" + result[1] + "," + result[2] + "," + result[3] )
                if(seq.charAt(result[2]-1) != result[3]){
                    matchAll = false
                }
            }
            if(matchAll){
                if(el.style.visibility == "visible"){
                    el.style.visibility = "hidden"
                }else{
                    el.style.visibility = "visible"
                }
            }
        }
    }
}

function changeSubs(evtId){
    //called when an effect ellipse is clicked
    console.log("changeSubs:" + evtId);
    var lst = document.querySelectorAll("#" + evtId + "_enabled")
    lst.forEach(function(item) {
        console.log("changeSubs: toggling one");
        if(item.style.visibility == "visible"){
            item.style.visibility = "hidden"
        }else{
            item.style.visibility = "visible"
        }
    }); 
    // now, redo the rest of the graphics .. 
    var ellipses = document.querySelectorAll("ellipse");
    var subStr = "" // looks like the typical mutation list: 23D, 121W etc
    for (var i = 0; i < ellipses.length; i++) {
        var el = ellipses[i]; 
        id = el.getAttribute("id")
        if(/_enabled/.test(id)){
            if(el.style.visibility == "visible"){
                groups = id.match(/(\d+\w)(_en)/)
                console.log("changeSubs: found sub " + groups[1]);
                if(!subStr.includes(groups[1])){
                    subStr = subStr + " " + groups[1];
                }
            }
        }
    }
    console.log("changeSubs: subStr " + subStr);
    var patt = /\d+/g;
    if(patt.test(subStr)){
        var ellipses = document.querySelectorAll("ellipse");
        for (var i = 0; i < ellipses.length; i++) {
            var el = ellipses[i]; 
            if(el.hasAttribute("data-sequencetag")){
                seq = el.getAttribute("data-sequencetag")
                groups = subStr.matchAll(/(\d+)(\w)/gi)
                var matchAll = true
                for(let result of groups){
                    console.log("changeSubs:" + result[1] + result[2])
                    var t = seq.charAt(result[1]-1)
                    if(t != result[2]){
                        matchAll = false
                    }
                }
                if(matchAll){
                    el.style.visibility = "visible"
                } else {
                    el.style.visibility = "hidden"
                }
            }
        }
        var paths = document.querySelectorAll("path");
        for (var i = 0; i < paths.length; i++) {
            var el = paths[i]; 
            if(el.hasAttribute("data-mutationtag")){
                mt = el.getAttribute("data-mutationtag")
                var matchAll = true
                var groups = subStr.matchAll(/(\d+\w)/gi)
                for(let result of groups){
                    console.log("showMutation: " + result[1] + " " + mt);
                    var reg = new RegExp("[^\\d]" + result[1])
                    if(!reg.test(mt)){
                        matchAll = false
                    }
                }
                if(matchAll){
                    el.style.visibility = "visible"
                } else {
                    el.style.visibility = "hidden"
                }
            }
        }
    } else {
        // turn all relevant elements off
        var ellipses = document.querySelectorAll("ellipse");
        for (var i = 0; i < ellipses.length; i++) {
            var el = ellipses[i]; 
            if(el.hasAttribute("data-sequencetag")){
                el.style.visibility = "hidden"
            }
        }
        var paths = document.querySelectorAll("path");
        for (var i = 0; i < paths.length; i++) {
            var el = paths[i]; 
            if(el.hasAttribute("data-mutationtag")){
                el.style.visibility = "hidden"
            }
        }
    }
}

function insertAfter(el, referenceNode) {
    referenceNode.parentNode.insertBefore(el, referenceNode.nextSibling);
}
function makeBG(elem) {
  var svgns = "http://www.w3.org/2000/svg"
  var bounds = elem.getBBox()
  var bg = document.createElementNS(svgns, "rect")
  var style = getComputedStyle(elem)
  var padding_top = parseInt(style["padding-top"])
  var padding_left = parseInt(style["padding-left"])
  var padding_right = parseInt(style["padding-right"])
  var padding_bottom = parseInt(style["padding-bottom"])
  bg.setAttribute("x", bounds.x - parseInt(style["padding-left"]))
  bg.setAttribute("y", bounds.y - parseInt(style["padding-top"]))
  bg.setAttribute("width", bounds.width + padding_left + padding_right)
  bg.setAttribute("height", bounds.height + padding_top + padding_bottom)
  bg.setAttribute("fill", "#f7ff85")
  bg.setAttribute("stroke-width", style["border-top-width"])
  bg.setAttribute("stroke", style["border-top-color"])
  if (elem.hasAttribute("transform")) {
    bg.setAttribute("transform", elem.getAttribute("transform"))
  }
  bg.setAttribute("visibility", "hidden")
  bg.setAttribute("id", elem.getAttribute("id") + "_background")
  elem.parentNode.insertBefore(bg, elem)
  //above needs a redraw event to work properly.  
  // fortunately, that occurs when unhiding. 
}
function redoBG(){
    //first remove all the old background rects. 
    var rects = document.querySelectorAll("rect");
    for (var i = 0; i < rects.length; i++) {
        el = rects[i]; 
        if(el.hasAttribute("id")){
            var id = el.getAttribute("id")
            if(/background/.test(id)){
                //console.log("redoGB removing " + id)
                el.parentNode.removeChild(el); 
            }
        }
    };
    var texts = document.querySelectorAll("text");
    for (var i = 0; i < texts.length; i++) {
        makeBG(texts[i]);
    };
}
function redoMutPaths(){
    // and all the mutation paths. 
    var paths = document.querySelectorAll("path");
    for (var i = 0; i < paths.length; i++) {
        var el = paths[i]; 
        if(el.hasAttribute("data-xone"+g_xaxis)){
            var x1 = parseFloat(el.getAttribute("data-xone"+g_xaxis))
            var x2 = parseFloat(el.getAttribute("data-xtwo"+g_xaxis))
            var y1 = parseFloat(el.getAttribute("data-yone"+g_yaxis))
            var y2 = parseFloat(el.getAttribute("data-ytwo"+g_yaxis))
            el.setAttribute("d", "M "+x1+","+y1+" "+((x1+x2)/2)+","+((y1+y2)/2)+" "+x2+","+y2)
            //need to move the label too. 
            var id = el.getAttribute("id")
            var muno = el.getAttribute("data-muno")
            var targetId = "#" + id + "_label" + muno
            lst = document.querySelectorAll(targetId); 
            lst.forEach(function(item) {
                var xpos = (x1+x2)/2; 
                var ypos = (y1+y2)/2 - 0.25; 
                item.setAttribute("x", xpos)
                item.setAttribute("y", ypos)
            });
            lab = el.parentNode; 
            
        }
    }
}
function enableXAxis(axis){
    console.log("enableXAxis " + axis)
    g_xaxis = axis;
    for(ax = 0; ax < g_nAxes; ax++){
        var el = document.querySelector("#axisX" + ax);
        if(el){
            if(ax == axis){
                el.style.visibility = "visible"
            } else {
                el.style.visibility = "hidden"
            }
        }
        var el = document.querySelector("#axisXlabel" + ax);
        if(el){
            if(ax == axis){
                el.style.opacity = "1.0"
            } else {
                el.style.opacity = "0.4"
            }
        }
    }
    // now, move all the points. 
    var ellipses = document.querySelectorAll("ellipse");
    for (var i = 0; i < ellipses.length; i++) {
        var el = ellipses[i]; 
        if(el.hasAttribute("data-xone"+axis)){
            var xpos = el.getAttribute("data-xone"+axis)//this is a string, not float
            // find all other variants that need to be moved
            var id = el.getAttribute("id")
            var variants = document.querySelectorAll("#" + id);
            for( var j=0; j<variants.length; j++){
                el2 = variants[j]; 
                el2.setAttribute("cx", xpos)
            }
            //and the label
            var labels = document.querySelectorAll("#" + id + "_label");
            for( var j=0; j<labels.length; j++){
                el2 = labels[j]; 
                el2.setAttribute("x", xpos)
            }
        }
    }
    redoMutPaths();
    redoBG(); 
}
function enableYAxis(axis){
    console.log("enableYAxis " + axis)
    g_yaxis = axis;
    for(ax = 0; ax < g_nAxes; ax++){
        var el = document.querySelector("#axisY" + ax);
        if(el){
            if(ax == axis){
                el.style.visibility = "visible"
            } else {
                el.style.visibility = "hidden"
            }
        }
        var el = document.querySelector("#axisYlabel" + ax);
        if(el){
            if(ax == axis){
                el.style.opacity = "1.0"
            } else {
                el.style.opacity = "0.4"
            }
        }
    }
    // now, move all the points. 
    var ellipses = document.querySelectorAll("ellipse");
    for (var i = 0; i < ellipses.length; i++) {
        var el = ellipses[i]; 
        if(el.hasAttribute("data-yone"+axis)){
            var ypos = el.getAttribute("data-yone"+axis)
            // find all other variants that need to be moved
            var id = el.getAttribute("id")
            var variants = document.querySelectorAll("#" + id);
            for( var j=0; j<variants.length; j++){
                el2 = variants[j]; 
                el2.setAttribute("cy", ypos)
            }
            //and the label
            var labels = document.querySelectorAll("#" + id + "_label");
            for( var j=0; j<labels.length; j++){
                el2 = labels[j]; 
                el2.setAttribute("y", ypos-0.75)
            }
        }
    }
    redoMutPaths();
    redoBG(); 
}
