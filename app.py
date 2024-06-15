from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from PIL import Image
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the model
try:
    model = load_model('models/model.h5')
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

def preprocess_frame(frame, target_size):
    try:
        image = Image.fromarray(frame)
        if image.mode != "L":
            image = image.convert("L")
        image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)
        image = np.array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        logging.error(f"Error preprocessing frame: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if 'frame' not in data:
            logging.error("No frame provided in the request.")
            return jsonify({'error': 'No frame provided'}), 400

        frame_data = base64.b64decode(data['frame'])
        frame_array = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

        if frame is None:
            logging.error("Failed to decode frame.")
            return jsonify({'error': 'Failed to decode frame'}), 400

        processed_frame = preprocess_frame(frame, target_size=(48, 225))
        if processed_frame is None:
            logging.error("Error processing frame.")
            return jsonify({'error': 'Error processing frame'}), 500

        predictions = model.predict(processed_frame)
        predicted_class = np.argmax(predictions, axis=1)
        response = {'predicted_class': int(predicted_class[0])}

        return jsonify(response)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': f"An error occurred: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)