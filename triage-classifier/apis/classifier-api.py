from flask import Flask, request, jsonify
import numpy as np
import pickle
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import shap
import os
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load necessary resources (ensure these are pre-downloaded for production)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

# Load pre-trained models and scalers
word2vec_model_path = "models/classifier/word2vec.model"
word2vec_model = Word2Vec.load(word2vec_model_path)

scaler_embeddings_path = "models/classifier/scaler-embeddings.pkl"
with open(scaler_embeddings_path, "rb") as f:
    scaler_embeddings = pickle.load(f)

scaler_vitals_path = "models/classifier/scaler-vitals.pkl"
with open(scaler_vitals_path, "rb") as f:
    scaler_vitals = pickle.load(f)

classifier_model_path = "models/classifier/classifier3.pkl"
with open(classifier_model_path, "rb") as f:
    classifier_model = pickle.load(f)

# Initialize SHAP explainer for the Random Forest model
try:
    explainer_rf = shap.TreeExplainer(classifier_model)
except Exception as e:
    print(f"Error initializing SHAP explainer: {e}")
    explainer_rf = None

# Define preprocessing components
stop_words = set(stopwords.words('english'))
stop_words -= {"no", "not", "wasn't", "was not", "isn't", "is not"}
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Preprocess the input text by removing punctuation, stopwords, and applying lemmatization."""
    if not text:
        return []
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    return [lemmatizer.lemmatize(word) for word in words]


def get_word_embeddings(sentence_tokens, model):
    """Generate sentence embedding by averaging word embeddings."""
    word_vecs = [model.wv[word] for word in sentence_tokens if word in model.wv]
    if not word_vecs:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)


@app.route('/predict/triage/range', methods=['POST'])
def predict():
    data = request.get_json()

    # Validate input data
    chief_complaint = data.get('chief_complaint')
    systolic_bp = data.get('systolic_bp')
    spo2 = data.get('spo2')
    pulse_rate = data.get('pulse_rate')

    if not isinstance(systolic_bp, (int, float)):
        return jsonify({"error": "Invalid systolic_bp value. Must be a number."}), 400
    if not isinstance(spo2, (int, float)):
        return jsonify({"error": "Invalid spo2 value. Must be a number."}), 400
    if not isinstance(pulse_rate, (int, float)):
        return jsonify({"error": "Invalid pulse_rate value. Must be a number."}), 400

    # Preprocess the chief complaint and get embeddings
    words = preprocess_text(chief_complaint)
    word_embeddings = get_word_embeddings(words, word2vec_model)

    # Scale embeddings if valid, otherwise use zero vector
    if word_embeddings.any():
        word_embeddings_scaled = scaler_embeddings.transform(word_embeddings.reshape(1, -1))
    else:
        word_embeddings_scaled = np.zeros((1, scaler_embeddings.n_features_in_))

    # Scale vital signs
    vital_signs = np.array([[spo2, pulse_rate, systolic_bp]])
    vital_signs_scaled = scaler_vitals.transform(vital_signs)

    # Combine features
    input_features = np.hstack((word_embeddings_scaled.flatten(), vital_signs_scaled.flatten())).reshape(1, -1)

    # Make prediction
    triage_prediction = classifier_model.predict(input_features)[0]
    classifier_probabilities = classifier_model.predict_proba(input_features)[0]

    # Generate SHAP explanation only for vital signs
    shap_explanation = {}
    if explainer_rf:
        shap_values = explainer_rf.shap_values(input_features)
        try:
            # Extract SHAP values only for vital signs
            vital_signs_shap_values = shap_values[0][0][-vital_signs_scaled.shape[1]:]
            shap_explanation = {
                "spo2_importance": {
                    "impact": "positive" if vital_signs_shap_values[0] > 0 else "negative",
                    "value": round(vital_signs_shap_values[0], 4)
                },
                "pulse_rate_importance": {
                    "impact": "positive" if vital_signs_shap_values[1] > 0 else "negative",
                    "value": round(vital_signs_shap_values[1], 4)
                },
                "systolic_bp_importance": {
                    "impact": "positive" if vital_signs_shap_values[2] > 0 else "negative",
                    "value": round(vital_signs_shap_values[2], 4)
                }
            }
        except IndexError:
            shap_explanation["error"] = "Unable to calculate SHAP values for vital signs."

    # Confidence score
    confidence_score = max(classifier_probabilities) * 100

    # Map classifier prediction to triage ranges
    if triage_prediction == "Range 1":
        triage_range = "Range 1"
        api_url = "http://127.0.0.1:5003/predict/triage/level"
    elif triage_prediction == "Range 2":
        triage_range = "Range 2"
        api_url = "http://127.0.0.1:5002/predict/triage/level"
    else:
        triage_range = "Range 3"
        api_url = "http://127.0.0.1:5001/predict/triage/level"

    # Prepare response
    response = {
        "triage_range": triage_range,
        "redirect_api": api_url,
        "confidence_score": confidence_score,
        "classifier_probabilities": classifier_probabilities.tolist(),
        "shap_explanation": shap_explanation
    }

    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(debug=False, port=port)
