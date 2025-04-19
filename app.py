from flask import Flask, request, jsonify
import mlflow.sklearn
import json
import os
from loguru import logger

app = Flask(__name__)

# Cấu hình model name và stage (có thể lấy từ biến môi trường)
MODEL_NAME = os.getenv("MODEL_NAME", "BestClassifierModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "latest")

# Load model từ MLflow Model Registry
logger.info(f"Loading model: {MODEL_NAME} ({MODEL_STAGE})...")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Load metadata (file JSON các kết quả model)
try:
    with open("model_comparison_results.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    logger.warning("model_comparison_results.json not found.")
    model_results = []

@app.route("/")
def home():
    return jsonify(message="Welcome to MLflow Classifier API")

@app.route("/models", methods=["GET"])
def get_all_models():
    """Return all trained models and their accuracy/params"""
    return jsonify(model_results)

@app.route("/predict", methods=["POST"])
def predict():
    """
    Dự đoán dựa trên dữ liệu đầu vào.
    JSON: { "features": [float, float, ..., float] }
    """
    data = request.get_json()
    features = data.get("features")

    if not features or not isinstance(features, list):
        return jsonify({"error": "Invalid input format. Provide 'features': [list of values]"}), 400

    try:
        prediction = model.predict([features])[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)