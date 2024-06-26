import base64
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO

# Load your trained models here
cnn_model_path = r'C:\Users\student\Desktop\Hand Written Digit Classification\CNN_model.h5'
mlp_model_path = r'C:\Users\student\Desktop\Hand Written Digit Classification\MLP_model.h5'
lenet_model_path = r'C:\Users\student\Desktop\Hand Written Digit Classification\LeNet5_model.h5'

app = Flask(__name__)

def preprocess_image(image):
    image = image.resize((28, 28)).convert('L')
    image = np.array(image)
    image = 255 - image
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    model_type = data['model']
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    processed_image = preprocess_image(image)
    
    results = {}
    
    if model_type == 'cnn' or model_type == 'all':
        cnn_pred = cnn_model.predict(processed_image).argmax()
        results['cnn'] = int(cnn_pred)
    
    if model_type == 'mlp' or model_type == 'all':
        mlp_pred = mlp_model.predict(processed_image.reshape(1, 28 * 28)).argmax()
        results['mlp'] = int(mlp_pred)
    
    if model_type == 'lenet' or model_type == 'all':
        lenet_pred = lenet_model.predict(processed_image).argmax()
        results['lenet'] = int(lenet_pred)

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)


