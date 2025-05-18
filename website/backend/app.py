from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os
import logging

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)

def load_model():
    try:
        model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        model = joblib.load(model_path)
        logging.info("Model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Model loading error: {e}")
        raise

model = load_model()

FEATURES = [
    'Irregular / Missed periods', 'Cramping', 'Menstrual clots', 'Infertility',
    'Pain / Chronic pain', 'Diarrhea', 'Long menstruation', 'Vomiting / constant vomiting',
    'Migraines', 'Extreme Bloating', 'Leg pain', 'Depression', 'Fertility Issues',
    'Ovarian cysts', 'Painful urination', 'Pain after Intercourse',
    'Digestive / GI problems', 'Anaemia / Iron deficiency', 'Hip pain',
    'Vaginal Pain/Pressure', 'Cysts (unspecified)', 'Abnormal uterine bleeding',
    'Hormonal problems', 'Feeling sick', 'Abdominal Cramps during Intercourse',
    'Insomnia / Sleeplessness', 'Loss of appetite'
]

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = [int(data.get(feature, 0)) for feature in FEATURES]
        input_array = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_array)[0]
        proba = model.predict_proba(input_array)[0][prediction]
        return jsonify({
            'prediction': int(prediction),
            'confidence': float(proba),
            'diagnosis': 'Endometriosis' if prediction == 1 else 'No Endometriosis'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/')
def serve_index():
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')
    if not os.path.exists(frontend_dir):
        logging.error(f"Frontend directory does not exist: {frontend_dir}")
        return "Error: Frontend directory does not exist.", 500
    return send_from_directory(frontend_dir, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    frontend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../frontend')
    return send_from_directory(frontend_dir, filename)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
