from flask import Flask, request, jsonify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np
from PIL import Image
import io
import base64
from flask_cors import CORS

actions = ['computers', 'faculty', 'Hello', 'I am', 'information', 'student', 'university', 'Mansoura', 'in', 'and']

app = Flask(__name__)
CORS(app)

model = None  

def load_my_model():
    global model
    input_shape = (30, 1662)  
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

model = load_my_model()

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
        image.close()  
    return np.array(processed_frames)

@app.route('/predict_sequence', methods=['POST'])
def predict_sequence():
    try:
        data = request.json
        frames = data['frames']
        
        processed_frames = preprocess_sequence(frames, target_size=(640, 480))
        
        if processed_frames.shape != (30, 1662):
            raise ValueError(f"Reshaped frames have incorrect shape {processed_frames.shape}, expected (30, 1662)")
        processed_frames = np.expand_dims(processed_frames, axis=0)
        predictions = model.predict(processed_frames)
        max_confidence = np.max(predictions)
        
        if max_confidence < 0.3:  
            print("Confidence level below threshold, not responding")
            return jsonify({"predicted_action": "None", "confidence": float(max_confidence)}), 200
        
        predicted_class = np.argmax(predictions, axis=1)
        predicted_action = actions[int(predicted_class)]
        
        print("Predicted action:", predicted_action)
        print("Predictions dictionary:", {
            "predicted_action": predicted_action,
            "confidence": float(max_confidence),
            "predictions": predictions.tolist()  
        })
        
        return jsonify({
            "predicted_action": predicted_action,
            "confidence": float(max_confidence),
            "predictions": predictions.tolist()
        }), 200
    
    except KeyError as e:
        error_message = f"KeyError: Missing key in JSON request: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 400
    
    except ValueError as e:
        error_message = f"ValueError: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 400
    
    except Exception as e:
        error_message = f"Exception during prediction: {str(e)}"
        print(error_message)
        return jsonify({"error": error_message}), 500

if __name__ == '__main__':
    app.run(debug=True)