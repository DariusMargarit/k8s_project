import re
import base64
import io
import numpy as np
import logging
from PIL import Image
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from skimage import color, transform
from transformers import CLIPProcessor, CLIPModel

class ImageTransformer:
    def transform(self, image):
        img_array = np.array(image)
        if img_array.shape[2] == 4:
            img_array = color.rgba2rgb(img_array)
        grayscale_image = color.rgb2gray(img_array)
        resized_image = transform.resize(grayscale_image, (28, 28), anti_aliasing=True)
        inverted_image = 1.0 - resized_image
        thresholded_image = np.clip(inverted_image, 0, 1)

        # Convert to 3-channel RGB image
        rgb_image = np.stack([thresholded_image] * 3, axis=-1)
        pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8))

        return pil_image

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

pipeline = ImageTransformer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    return 'Model is running!'


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['imageData']
    image_data = base64.b64decode(re.sub('^data:image/.+;base64,', '', data))
    image = Image.open(io.BytesIO(image_data))
    transformed_image = pipeline.transform(image)

    inputs = processor(text=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                       images=transformed_image, return_tensors="pt", padding=True)
    outputs = model(**inputs)

    logits = outputs.logits_per_image
    predicted_digit = logits.softmax(dim=1).detach().numpy()
    predicted_digit = np.argmax(predicted_digit, axis=1)[0]
    logger.info('Model 2')

    resp = make_response(jsonify(predicted_digit=predicted_digit.astype('str')))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
