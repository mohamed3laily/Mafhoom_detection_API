from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import base64
from PIL import Image
import threading
from flask_cors import CORS  

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
# Load the model
model = load_model('model.h5')

# Function to preprocess the frame
def preprocess_frame(frame, target_size):
    image = Image.fromarray(frame)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((target_size[1], target_size[0]), Image.LANCZOS)  
    image = np.array(image)
    image = image / 255.0 
    image = np.expand_dims(image, axis=0)  
    return image

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    if 'frame' not in data:
        return jsonify({'error': 'No frame provided'}), 400

    frame_data = base64.b64decode(data['frame'])
    frame_array = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Failed to decode frame'}), 400

    processed_frame = preprocess_frame(frame, target_size=(48, 225))
    predictions = model.predict(processed_frame)
    
    predicted_class = np.argmax(predictions, axis=1)
    response = {'predicted_class': int(predicted_class[0])}  

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
