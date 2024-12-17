from flask import Flask, request, jsonify
import pickle
import numpy as np
from gensim.models import Word2Vec
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import logging
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load pre-trained models and scalers
try:
    with open('models/logistic-regression/spe/logistic_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    word2vec_model = Word2Vec.load('models/logistic-regression/spe/word2vec.model')

    with open('models/logistic-regression/spe/scaler-embeddings.pkl', 'rb') as f:
        scaler_embeddings = pickle.load(f)

    with open('models/logistic-regression/spe/scaler-vitals.pkl', 'rb') as f:
        scaler_vitals = pickle.load(f)
except Exception as e:
    logging.error(f"Error loading model or scaler: {e}")

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing function
stop_words = set(stopwords.words('english')) - {"no", "not", "wasn't", "was not", "isn't", "is not"}
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    try:
        text = text.lower()  # Lowercasing
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
        words = word_tokenize(text)  # Tokenization
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatization
        return ' '.join(words)
    except Exception as e:
        logging.error(f"Error in text preprocessing: {e}")
        return ""


# Function to convert text to embedding
def get_sentence_embedding(sentence, model):
    words = sentence.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        logging.warning(f"No word vectors found for the sentence: {sentence}")
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)


# API route for prediction
@app.route('/predict/triage/level', methods=['POST'])
def predict_triage_level():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate input
        if not all(key in data for key in ('chief_complaint', 'spo2', 'pulse_rate', 'systolic_bp')):
            return jsonify({"error": "Missing required fields"}), 400

        # Ensure vital signs are numeric
        try:
            spo2 = float(data['spo2'])
            pulse_rate = float(data['pulse_rate'])
            systolic_bp = float(data['systolic_bp'])
        except ValueError:
            return jsonify({"error": "Vital signs must be numeric"}), 400

        # Validate ranges for all vital signs
        if not (0 <= spo2 <= 100):
            return jsonify({"error": "SpO2 must be between 0 and 100"}), 400
        if not (30 <= pulse_rate <= 200):
            return jsonify({"error": "Pulse rate must be between 30 and 200"}), 400
        if not (50 <= systolic_bp <= 300):
            return jsonify({"error": "Systolic BP must be between 50 and 300"}), 400

        chief_complaint = data['chief_complaint']

        # Preprocess the text (chief complaint)
        preprocessed_complaint = preprocess_text(chief_complaint)

        # Get the text embedding for chief complaint
        text_embedding = get_sentence_embedding(preprocessed_complaint, word2vec_model)

        # Scale the embeddings and vital signs
        text_embedding_scaled = scaler_embeddings.transform([text_embedding])
        vital_signs_scaled = scaler_vitals.transform([[spo2, pulse_rate, systolic_bp]])

        # Combine scaled embeddings and vital signs
        X_combined = np.hstack((text_embedding_scaled, vital_signs_scaled))

        # Make prediction using the model
        predicted_triage = model.predict(X_combined)[0]  # Get the predicted triage level

        # Calculate confidence score
        probabilities = model.predict_proba(X_combined)[0]  # Get probabilities for all classes
        confidence_score = round(max(probabilities) * 100, 2)  # Confidence as a percentage

        # Log the prediction
        logging.info(f"Predicted triage level: {predicted_triage} for complaint: {chief_complaint}")

        # Create response
        response = {
            'triage_level': int(predicted_triage),
            'confidence_score': confidence_score
        }

        # Return the result as JSON (only the triage level)
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=5001)
