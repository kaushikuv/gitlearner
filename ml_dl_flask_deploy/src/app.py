from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = r"C:\Users\LENOVO\ml_dl_flask_deploy\models\rf_model.pkl"
SCALER_PATH = r"C:\Users\LENOVO\ml_dl_flask_deploy\models\scaler.pkl"


# Get the directory where app.py is located
#BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load model & scaler
rf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['features']])
    scaled_features = scaler.transform(features)
    prediction = rf.predict(scaled_features)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)