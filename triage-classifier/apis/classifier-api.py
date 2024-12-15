from flask import Flask, request, jsonify
import numpy as np
import pickle
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Initialize Flask app
app = Flask(__name__)

# Load necessary resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Load pre-trained models and scalers
word2vec_model_path = "models/classifier/word2vec.model"
word2vec_model = Word2Vec.load(word2vec_model_path)

scaler_embeddings_path = "models/classifier/scaler-embeddings.pkl"
with open(scaler_embeddings_path, "rb") as f:
    scaler_embeddings = pickle.load(f)

scaler_vitals_path = "models/classifier/scaler-vitals.pkl"
with open(scaler_vitals_path, "rb") as f:
    scaler_vitals = pickle.load(f)

classifier_model_path = "models/classifier/classifier3.pkl"  # Replace with the actual path to your classifier
with open(classifier_model_path, "rb") as f:
    classifier_model = pickle.load(f)

# Define preprocessing components
stop_words = set(stopwords.words('english')) - {"no", "not", "wasn't", "was not", "isn't", "is not"}
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Preprocess the input text by removing punctuation, stopwords, and applying lemmatization."""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


def get_sentence_embedding(sentence, model):
    """Generate sentence embedding by averaging word embeddings."""
    words = sentence.split()
    word_vecs = [model.wv[word] for word in words if word in model.wv]
    if len(word_vecs) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vecs, axis=0)

# API route for prediction
@app.route('/predict/triage/range', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract input data
    chief_complaint = data.get('chief_complaint', "")
    systolic_bp = data.get('systolic_bp', 0)
    spo2 = data.get('spo2', 0)
    pulse_rate = data.get('pulse_rate', 0)

    # Preprocess the chief complaint
    processed_complaint = preprocess_text(chief_complaint)

    # Generate text embedding
    text_embedding = get_sentence_embedding(processed_complaint, word2vec_model)
    text_embedding_scaled = scaler_embeddings.transform([text_embedding])

    # Scale vital signs
    vital_signs = np.array([[spo2, pulse_rate, systolic_bp]])
    vital_signs_scaled = scaler_vitals.transform(vital_signs)

    # Combine features
    combined_features = np.hstack((text_embedding_scaled, vital_signs_scaled))

    # Predict using the classifier model
    classifier_prediction = classifier_model.predict(combined_features)[0]
    classifier_probabilities = classifier_model.predict_proba(combined_features)[0]

    # Calculate confidence score
    confidence_score = max(classifier_probabilities)*100

    # Map classifier prediction to triage ranges
    if classifier_prediction == "Range 1":
        triage_range = "Range 1"
        api_url = "http://127.0.0.1:5003/predict/triage/level"
    elif classifier_prediction == "Range 2":
        triage_range = "Range 2"
        api_url = "http://127.0.0.1:5002/predict/triage/level"
    else:
        triage_range = "Range 3"
        api_url = "http://127.0.0.1:5001/predict/triage/level"

    # Prepare the response
    response = {
        "triage_range": triage_range,
        "redirect_api": api_url,
        "confidence_score": confidence_score,
        "classifier_probabilities": classifier_probabilities.tolist()
    }

    # Return the prediction
    return jsonify(response)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, port=8000)
