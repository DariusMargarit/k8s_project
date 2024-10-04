from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import torch
import torch.nn as nn
import os
import numpy as np
import logging
from skimage import color, transform
from PIL import Image
import io
import base64
import re
from typing import List

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the MLP model class
class MLP(nn.Module):
    def __init__(self, input_size: int = 784, output_size: int = 10, hidden_layer: List[int] = [100], activations: List[nn.Module] = [nn.ReLU]):
        super().__init__()

        layers = []
        if len(hidden_layer) != len(activations):
            raise ValueError("The length of the hidden_layer and activations lists must match.")

        prev_layer_size = input_size
        for layer_size, activation_func in zip(hidden_layer, activations):
            layers.append(nn.Linear(prev_layer_size, layer_size))
            layers.append(activation_func())
            prev_layer_size = layer_size

        layers.append(nn.Linear(prev_layer_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Initialize the model
model = MLP(input_size=784, output_size=10, hidden_layer=[300, 100, 50], activations=[nn.ReLU, nn.ReLU, nn.ReLU])

# Load the model state_dict
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist_model.pth')
try:
    device = torch.device('cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
except FileNotFoundError:
    print(f"Model file not found at {model_path}. Please check the path and try again.")

class ImageTransformer:
    def transform(self, image):
        img_array = np.array(image)
        if img_array.shape[2] == 4:
            img_array = color.rgba2rgb(img_array)
        grayscale_image = color.rgb2gray(img_array)
        resized_image = transform.resize(grayscale_image, (28, 28), anti_aliasing=True)
        inverted_image = 1.0 - resized_image
        thresholded_image = np.clip(inverted_image, 0, 1)
        tensor = torch.tensor(thresholded_image, dtype=torch.float32).view(1, 784)
        return tensor.numpy()

pipeline = ImageTransformer()

@app.route('/')
def index():
    return 'Model is running!'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['imageData']
    image_data = base64.b64decode(re.sub('^data:image/.+;base64,', '', data))
    image = Image.open(io.BytesIO(image_data))
    transformed_image = pipeline.transform(image)
    tensor = torch.tensor(transformed_image, dtype=torch.float32)
    output = model(tensor)
    predicted_digit = torch.argmax(output, dim=1).item()
    resp = make_response(jsonify(predicted_digit=predicted_digit))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    logger.info('Model 1')
    return resp

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
