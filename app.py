from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input, LayerNormalization, Dropout
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from flask_cors import CORS
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.utils import custom_object_scope
import logging

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

# Flask app setup
app = Flask(__name__)
CORS(app)

# Model path
model_path = "models/model.h5"

# Load the model within custom object scope
with custom_object_scope({'TransformerEncoder': TransformerEncoder}):
    model = load_model(model_path)

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0
    image = image.flatten()[:1662]
    return image

def preprocess_sequence(frames, target_size):
    processed_frames = []
    for frame in frames:
        image_data = frame.split(",")[1]
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        processed_image = preprocess_image(image, target_size)
        processed_frames.append(processed_image)
    return np.array(processed_frames)

@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    try:
        data = request.json
        frames = data['frames']

        logging.info(f"Received {len(frames)} frames")

        if len(frames) != 30:
            raise ValueError(f"Expected 30 frames, but got {len(frames)} frames")

        processed_frames = preprocess_sequence(frames, target_size=(640, 480))

        if processed_frames.shape != (30, 1662):
            raise ValueError(f"Processed frames have incorrect shape {processed_frames.shape}, expected (30, 1662)")

        processed_frames = np.expand_dims(processed_frames, axis=0)
        predictions = model.predict(processed_frames)
        max_confidence = np.max(predictions)

        predicted_class = np.argmax(predictions, axis=1)
        predicted_action = actions[int(predicted_class)]

        return jsonify({
            "predicted_action": predicted_action,
            "confidence": float(max_confidence),
            "predictions": predictions.tolist()
        }), 200

    except KeyError as e:
        error_message = f"KeyError: Missing key in JSON request: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 400

    except ValueError as e:
        error_message = f"ValueError: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 400

    except Exception as e:
        error_message = f"Exception during prediction: {str(e)}"
        logging.error(error_message)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)