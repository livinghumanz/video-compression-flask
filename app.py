#from fastai1 import *
from flask import Flask,render_template,request,flash,url_for,redirect
from werkzeug.utils import secure_filename
#import json
import CompressVdo
import os
import time
import tablib

###############################################
#l=learner.load_learner("./models/level1.pth") #
###############################################

app=Flask(__name__)
app.secret_key = 'h432hi5ohi3h5i5hi3o2hi'

#create a route
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/Compression',methods=['GET','POST'])
def result():
    if request.method == 'POST':
        
        if request.files:
            video=request.files["vupload"]
            #flash(" ".join(request.form.keys()))
            flash(video.filename)
        '''for i in range(5):
            flash("hello "+str(i))
            time.sleep(2)'''
        infile="cvideo.mp4"
        path1=os.getcwd()+"/static/video/"
        path=path1+infile
        video.save(path)
        vd=CompressVdo.compressVideo(path)
        #for o in vd:
        #    flash(o)
        #time.sleep(60)
        f1=round((os.stat(path).st_size)/1024,2)
        f2=round((os.stat(path1+"cvideo1.mp4").st_size)/1024,2)
        percent=round((f2*100)/f1,2)
        vd.extend([f1,f2,percent])

        
        
        #print(video,os.getcwd()+"/static/video/cvideo.mp4")
        #f=request.form['img_file'].split("/")
        #-----------------------------------#
        #result=jsonify(l.predict(f))        #
        #json.dump(result,"testfile.json")   #
        #-----------------------------------#
        '''with open("testfile.json") as jfile:
            dicl=json.load(jfile)
        ifile=f[len(f)-1]
        if ifile in dicl.keys():
            result=dicl[ifile]
        furl="/test_images/"+f[len(f)-1]'''
        return render_template('index.html',isindex=True,video_info=vd)#,imagef=str(url_for("static",filename=furl)),result=result)
    else:
        return redirect(url_for('home'))
@app.route('/model')
def model():
    return render_template('model.html')

####################################
