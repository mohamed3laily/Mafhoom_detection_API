from flask import Flask, request, jsonify
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
import base64
from flask_cors import CORS
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.utils import custom_object_scope
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Dropout, GlobalAveragePooling1D
import os
import tempfile
import logging
from NLP_pipeline import process_arabic_text , words



# Define actions
actions2 = ['computers', 'student', 'information', 'Hello', 'i love you', 'thanks']
actions8 = ['faculty','Hello','I am','iloveyou','in','enemy','student','thanks']
actions7 = ['computers','Hello','iloveyou','in','screen', 'student','thanks']
actions10 = ['computers','faculty','Hello','I am','iloveyou','in','information','enemy','student','thanks']
actions17 = ['computers','enemy','faculty','family','father','friend','Hello','I am','iloveyou','in','information','job','Mansoura','mother','screen','student','thanks']
actions = ['computers','enemy','faculty','Hello','I am','iloveyou','in','information','job','screen', 'student','thanks','family']

# Define TransformerEncoder class (unchanged)
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

# Model configuration
embed_dim = 64
num_heads = 4
ff_dim = 128

# Build the model
def build_model():
    input_shape = (30, 1662)
    inputs = Input(shape=input_shape)
    x = inputs
    x = Dense(embed_dim)(x)
    for _ in range(2):
        x = TransformerEncoder(embed_dim, num_heads, ff_dim)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(len(actions), activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

# Load the model
model_path = "models/final13words_100seq.h5"
model = build_model()
model.load_weights(model_path)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])



# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic

def extract_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        print(f"Video has fewer than {num_frames} frames. Using all available frames.")
        num_frames = total_frames
    
    step = total_frames // num_frames
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        if len(frames) == num_frames:
            break
    
    cap.release()
    return frames

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
        video_base64 = data.get('video', '')
        
        if not video_base64:
            return jsonify({"error": "No video received"}), 400

        # Decode base64 video and save to a temporary file
        video_data = base64.b64decode(video_base64)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video:
            temp_video.write(video_data)
            temp_video_path = temp_video.name

        # Extract frames from the video
        frames = extract_frames(temp_video_path)
        logging.info(f"Extracted {len(frames)} frames from video")

        sequence = []
        sentence = []
        predictions = []
        confidence_values = []
        threshold = 0.5

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            for i, frame in enumerate(frames):
                try:
                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)
                    
                    # Extract keypoints
                    keypoints = extract_keypoints(results)

                    # Append keypoints to sequence
                    sequence.append(keypoints)

                except Exception as frame_error:
                    logging.error(f"Error processing frame {i}: {str(frame_error)}")
                    continue

            # Ensure we have exactly 30 frames
            sequence = sequence[:30]
            if len(sequence) < 30:
                logging.warning(f"Only {len(sequence)} frames were processed. Padding with zeros.")
                sequence += [np.zeros(1662)] * (30 - len(sequence))

            input_data = np.expand_dims(sequence, axis=0)
            logging.info(f"Input shape: {input_data.shape}")
            res = model.predict(input_data)[0]
            logging.info(f"Prediction result: {res}")
            predicted_action_index = np.argmax(res)
            
            predictions.append(actions[predicted_action_index])
            confidence_value = res[predicted_action_index]
            confidence_values.append(confidence_value)

            if confidence_value > threshold:
                sentence.append(actions[predicted_action_index])

        # Clean up the temporary video file
        os.unlink(temp_video_path)

        logging.info(f"Predicted sentence: {' '.join(sentence)}")
        logging.info(f"Predictions: {predictions}")
        logging.info(f"Confidence values: {confidence_values}")

        return jsonify({
            "predicted_sentence": ' '.join(sentence),
            "predictions": predictions,
            "confidence": float(confidence_value)
        }), 200

    except Exception as e:
        logging.error(f"Error during prediction sequence: {str(e)}")
        return jsonify({"error": f"Failed to predict sequence: {str(e)}"}), 500

@app.route('/process_text', methods=['POST'])
def process_arabic_endpoint():
    data = request.json
    if 'text' not in data:
        return jsonify({'error': 'Missing text parameter'}), 400

    text = data['text']
    result = process_arabic_text(text, words)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)