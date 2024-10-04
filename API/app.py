from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_SERVICE_URL = os.getenv('MODEL_SERVICE_URL')

@app.route('/')
def index():
    return "Flask API is running"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('imageData')  # Assuming JSON payload
    if not data:
        return jsonify(error='Missing imageData in request body'), 400

    try:
        response = requests.post(f"{MODEL_SERVICE_URL}/predict", json={'imageData': data})
        if response.status_code == 200:
            return jsonify(predicted_digit=response.json().get('predicted_digit')), 200
        else:
            return jsonify(error='Error in model service'), response.status_code
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
