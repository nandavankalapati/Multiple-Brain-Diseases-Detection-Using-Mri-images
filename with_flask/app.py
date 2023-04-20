from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf

# Define a flask app
app = Flask(__name__)

# Load your trained model
model = load_model('Brain_Tumor_Image_Classification_Model.h5')  
print('Model loaded.')

def model_predict(img_path,model):
    #img = image.load_img(img_path, target_size=(150, 150))
    #img_array = image.img_to_array(img)
    #img_batch = np.expand_dims(img_array, axis=0) / 255.
    img = cv2.imread(img_path)
    img = cv2.resize(img, (150, 150))
    img_array = np.array(img)
    img_array.shape
    img_array = img_array.reshape(1,150,150,3)
    img_array.shape

    #a = model.predict(img_batch)
    a = model.predict(img_array)

    class_names = ['Alzheimer-MildDemented', 'Alzheimer-ModerateDemented', 'Alzheimer-NonDemented', 'Alzheimer-VeryMildDemented', 'Brain_Tumor-glioma_tumor', 'Brain_Tumor-glioma_tumor-meningioma_tumor', 'Brain_Tumor-glioma_tumor-pituitary_tumor', 'Healthy_Brain', 'Multiple Sclerosis-Control-Axial', 'Multiple Sclerosis-Control-Sagittal', 'Multiple Sclerosis-MS-Axial', 'Multiple Sclerosis-MS-Sagittal']
    #predicted_class_index = np.argmax(prediction)
    #predicted_class = class_names[predicted_class_index]
    predicted_class = class_names[a.argmax()]

    return predicted_class


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path,model)
        #pred_class=decode_predictions(preds,top=1)
        #prediction=str(preds[0][0][1])
        return preds        
    return None

if __name__ == '__main__':
    app.run(debug=True)
