from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


MODEL_PATH = 'model.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()   
print('Model loaded. Check http://127.0.0.1:5000/')

def model_predict(img_path, model):
    img = keras.utils.load_img(img_path, target_size=(224, 224))
    x = keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def splash():
    return render_template('splash.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
     if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'images', secure_filename(f.filename))
        f.save(file_path)


        preds = model_predict(file_path, model)
        accuracy=float(np.max(preds,axis=1)[0]) * 100
        result=np.argmax(preds,axis=1)[0]
        labels=np.array(preds)
        labels[labels>=0.6]=1
        labels[labels<0.6]=0
        final=np.array(labels)
        print(preds,result,accuracy)
        gt = {'preds': {'result': 'Glioma Tumor','accuracy': accuracy}}
        mt = {'preds': {'result': 'Meningioma Tumor','accuracy': accuracy}}
        nt = {'preds': {'result': 'Tidak ada tumor','accuracy': accuracy}}
        pt = {'preds': {'result': 'pituitary tumor','accuracy': accuracy}}
        

        if final[0][0]==1:
            return(gt)                                                       
        elif final[0][1]==1:
            return(mt)                                                 
        elif final[0][2]==1:
            return(nt)                                                   
        elif final[0][3]==1:                                                       
            return(pt)                                                    
        else:
            return('Tidak Tahu')
        

        


if __name__ == "__main__":
    app.run(debug=True)