from os import sep
from flask import Flask, render_template, request

import keras

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import skimage
from skimage import transform
import pandas as pd


DEFAULT_IMAGE_SIZE = tuple((256, 256))
labels = pd.read_csv("labels.txt", sep='/n').values

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])

def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    imgFile = request.files['imgFile']
    image_path = './images/' + imgFile.filename
    imgFile.save(image_path)

    new_model = keras.models.load_model('model_new.h5')

    img1 = cv2.imread(image_path)
    np_image = Image.open(imgFile)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)

    CATEGORIES = ["Brown Spot", "Common Rust","Healthy", "Northern Leaf Blight"]
    prediction = new_model.predict(np_image)
    prediction = np.argmax(prediction,axis=1)

    result = CATEGORIES[prediction[0]]
    toReturn = [image_path, result]

    print(image_path)
    print(result)
    print(prediction)
    


    return render_template('index.html', data=result) 




if __name__ == '__main__':
    app.run(port=3000, debug=True)
