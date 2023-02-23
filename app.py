import os
import base64
import numpy as np
import pandas as pd
import io
import tensorflow as tf
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from pathlib import Path
from keras.models import load_model

Path("./static").mkdir(parents=True, exist_ok=True)
UPLOAD_FOLDER = './static'
ALLOWED_EXTENSIONS = {'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app._static_folder = 'static'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
model = './models/EmailToFAQ.h5'

@app.route('/', methods = ['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'input.csv'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filepath) 
            file.save(filepath)
        
        transfer_model = load_model(model)
        data = pd.read_csv('./static/input.csv')
        
        pred = transfer_model.predict(data)
        print(pred)
        
        return render_template('result.html', pred = pred, len = len(pred))
    return render_template('index.html')


if __name__ == '__main__':
    app.run(port=5000, debug=True)