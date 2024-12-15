from flask import Flask, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import gensim.downloader as api

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained GloVe model, scaler, and trained classifier model
glove_model = api.load("glove-wiki-gigaword-200")
scaler_path = 'models/fcdnn/spe/scaler.pkl'
model_path = 'models/fcdnn/spe/fcdnn_model.keras'

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load the trained FCDNN model
model = load_model(model_path)

# Helper function to preprocess text using GloVe embeddings
def get_sentence_embedding_glove(sentence, glove_model):
    words = sentence.split()
    word_vecs = [glove_model[word] for word in words if word in glove_model]
    if len(word_vecs) == 0:
        return np.zeros(200)  # GloVe is 200-dimensional
    return np.mean(word_vecs, axis=0)

# Define the API route for prediction
@app.route('/predict/triage/level', methods=['POST'])
def predict_triage_level():
    data = request.get_json()

    # Extract input fields from the JSON data
    chief_complaint = data.get('chief_complaint')
    systolic_bp = data.get('systolic_bp')
    spo2 = data.get('spo2')
    pulse_rate = data.get('pulse_rate')

    # Preprocess the chief complaint
    text_embedding = get_sentence_embedding_glove(chief_complaint, glove_model)

    # Combine text embedding with the vitals
    vital_signs = np.array([spo2, pulse_rate, systolic_bp])
    combined_input = np.hstack((text_embedding, vital_signs))

    # Scale the input
    combined_input_scaled = scaler.transform([combined_input])

    # Make the prediction
    prediction = model.predict(combined_input_scaled)
    predicted_class = np.argmax(prediction, axis=1)[0] + 1  # Shift back to 1-5
    confidence_score = float(np.max(prediction, axis=1)[0])*100

    # Convert predicted_class (NumPy int64) to Python int
    predicted_class = int(predicted_class)

    # Create response
    response = {
        'triage_level': predicted_class,
        'confidence_score': confidence_score
    }

    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5003)
