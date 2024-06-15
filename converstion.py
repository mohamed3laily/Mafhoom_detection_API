from tensorflow.keras.models import load_model

model = load_model('model.keras')
model.save('model.h5')
