
var lab=["l1","l2","l3","l4"];
var i=0;var myVar;

function preview_file(image){
    
    //document.getElementById(image).src=document.getElementById("vupload").value
    //alert(document.getElementById("vupload").value.split("\\").slice(-1)[0]);
    document.getElementById("img_file").value= "-->  "+document.getElementById("vupload").value.split("\\").slice(-1)[0];
    document.getElementById("img_file").style.display="block"
    document.getElementById("p1").style.display="none";

}
function loadTimer(){
    i+=1;
    console.log("hello");
    console.log("print-->",i);
    if(i>3)
        {clearInterval(myVar);
            //alert("done"+String(i))
        }
    else
        showdiv(lab[i],lab[i-1]);
}

function checkImage(i_id) {
    
    
    var v=document.getElementById(i_id).value.split("\\").slice(-1)[0].split(".").slice(-1)[0];
    //console.log(v=="mp4");
    if(v=="mp4")
        { 
            showdiv(lab[i],"img_file")
            myVar=setInterval(() => {
                i+=1;
                console.log("hello");
                console.log("print-->",i);
                if(i>3)
                    {clearInterval(myVar);
                        //alert("done"+String(i))
                    }
                else
                    showdiv(lab[i],lab[i-1]);
                
            }, 3000);
            return true;    
        }
    document.getElementById("p1").style.display="block";
    return false    
}
function showdiv(ids,ids1) {
    document.getElementById(ids).style.display='block';
    document.getElementById(ids1).style.display='none';
    
}
