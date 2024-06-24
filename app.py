from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import base64
from flask_cors import CORS
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout
import logging
from dotenv import load_dotenv
import os

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Define actions
actions = ['computers', 'faculty', 'Hello', 'I am', 'information', 'student', 'university', 'Mansoura', 'in', 'and']

# Define a custom Transformer Encoder layer
class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.mha = MultiHeadAttention(head_size=embed_dim, num_heads=num_heads)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, x, training):
        attn_output = self.mha([x, x, x])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

app = Flask(__name__)
CORS(app)

# Load the model
model_path = "models/model.h5"
with custom_object_scope({'TransformerEncoder': TransformerEncoder}):
    model = load_model(model_path, compile=False)

# Recompile the model with a new optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose = np.zeros(132)
    face = np.zeros(1404)
    lh = np.zeros(21*3)
    rh = np.zeros(21*3)
    
    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()

    return np.concatenate([pose, face, lh, rh])

@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    try:
        data = request.json
        frames = data.get('frames', [])
        logging.info(f"Received {len(frames)} frames")

        if not frames:
            return jsonify({"error": "No frames received"}), 400

        sequence = []
        sentence = []
        predictions = []
        threshold = 0.5

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for i, frame_data in enumerate(frames):
                try:
                    # Decode base64 image
                    img_data = base64.b64decode(frame_data.split(',')[1])
                    nparr = np.frombuffer(img_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        logging.error(f"Frame {i} is None after decoding")
                        continue

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Extract keypoints
                    keypoints = extract_keypoints(results)

                    # Append keypoints to sequence
                    sequence.append(keypoints)
                    if len(sequence) > 30:
                        sequence.pop(0)

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0))[0]
                        predicted_action_index = np.argmax(res)
                        predictions.append(predicted_action_index)

                        if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == predicted_action_index:
                            if res[predicted_action_index] > threshold:
                                predicted_action = actions[predicted_action_index]
                                if not sentence or predicted_action != sentence[-1]:
                                    sentence.append(predicted_action)

                        if len(sentence) > 5:
                            sentence = sentence[-5:]
                except Exception as frame_error:
                    logging.error(f"Error processing frame {i}: {str(frame_error)}")
                    continue

        return jsonify({
            "predicted_sentence": ' '.join(sentence),
            "predictions": [actions[pred] for pred in predictions if pred < len(actions)]
        }), 200

    except Exception as e:
        error_message = f"Exception during prediction: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
