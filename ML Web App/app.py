from posixpath import join
from flask import Flask, render_template, request
import keras
import numpy as np
from PIL import Image
import skimage
from skimage import transform


app = Flask(__name__)

toReturn = ['']
showResults = False

@app.route('/', methods=['GET', 'POST'])

def home_page():
    
    return render_template('index.html', data="toReturn")

@app.route('/predict', methods=['POST'])

def predict():
    imgFile = request.files['imgFile']
    image_path = './static/images/' + imgFile.filename
    imgFile.save(image_path)

    new_model = keras.models.load_model('model_new.h5')

    np_image = Image.open(imgFile)
    np_image = np.array(np_image).astype('float32')/255
    np_image = transform.resize(np_image, (256, 256, 3))
    np_image = np.expand_dims(np_image, axis=0)

    CATEGORIES = ["Brown Spot", "Common Rust","Healthy", "Northern Leaf Blight"]
    prediction = new_model.predict(np_image)
    prediction = np.argmax(prediction,axis=1)

    result = CATEGORIES[prediction[0]]

    toReturn = [image_path, result]

    showResults = True

    print(image_path)
    print(toReturn)
    print(prediction)
    


    return render_template('index.html', data=toReturn) 




if __name__ == '__main__':
    app.run(port=3000, debug=True)
