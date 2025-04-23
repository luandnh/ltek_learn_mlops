from flask import Flask, request, Response
from flask_restx import Api, Resource, fields
import mlflow.sklearn
import os
import json
from loguru import logger


# Env config
try:
    with open("/app/VERSION", "r") as f:
        APP_VERSION = f.read().strip()
except FileNotFoundError:
    APP_VERSION = "unknown"

MODEL_NAME = os.getenv("MODEL_NAME", "BestClassifierModel")
MODEL_STAGE = os.getenv("MODEL_STAGE", "latest")

logger.info(f"Starting MLflow Classifier API - Version: {APP_VERSION}")
logger.info(f"Loading model: {MODEL_NAME} ({MODEL_STAGE})...")
model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")

# Initialize app + API
app = Flask(__name__)
api = Api(
    app,
    title="MLflow Classifier API",
    version=APP_VERSION,
    doc=False,
    prefix="/api",
    description=f"MLflow Classifier API - Version: {APP_VERSION}",
)

ns = api.namespace(
    "api",
    description="Model endpoints",
)

# Load model comparison results
try:
    with open("model_comparison_results.json", "r") as f:
        model_results = json.load(f)
except FileNotFoundError:
    logger.warning("model_comparison_results.json not found.")
    model_results = []

# Define input/output model
predict_input = api.model(
    "PredictInput",
    {"features": fields.List(fields.Float, required=True, description="Feature list")},
)

predict_output = api.model("PredictOutput", {"prediction": fields.Integer})


@app.route("/")
def scalar_index():
    html_content = f"""<!doctype html>
    <html>
    <head>
        <title>MLflow Classifier API</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
    </head>
    <body>
        <script id="api-reference" data-url="/openapi.json"></script>
        <script src="https://cdn.jsdelivr.net/npm/@scalar/api-reference"></script>
    </body>
    </html>"""
    return Response(html_content, mimetype="text/html")


info_model = api.model(
    "Info",
    {
        "message": fields.String(
            description="Welcome message", example="Welcome to MLflow Classifier API"
        ),
        "version": fields.String(
            description="Current version of the API", example=APP_VERSION
        ),
    },
)


@api.route("/info")
class Index(Resource):
    @api.doc(
        description="Provides general information about the API service and its version.",
        tags=["info"],
    )
    @api.marshal_with(info_model)
    def get(self):
        """MLflow Classifier API info"""
        return {"message": "Welcome to MLflow Classifier API", "version": APP_VERSION}


# Override default swagger.json to openapi.json
@app.route("/openapi.json")
def openapi_json():
    return api.__schema__


model_metadata = api.model(
    "ModelMetadata",
    {
        "model_name": fields.String(description="Name of the model"),
        "accuracy": fields.Float(description="Accuracy score"),
        "params": fields.Raw(description="Model parameters (varies by model)"),
    },
)

model_list = api.model(
    "ModelList",
    {
        "models": fields.List(
            fields.Nested(model_metadata), description="List of trained models"
        )
    },
)


@ns.route("/models")
class AllModels(Resource):
    @api.doc(
        description="Returns metadata for all trained models including name, accuracy, and parameters.",
        tags=["model"],
    )
    @api.marshal_list_with(model_metadata)
    def get(self):
        """
        Get a list of all trained models and their evaluation results.
        """
        return model_results


@ns.route("/predict")
class Predict(Resource):
    @api.doc(
        description="Run prediction on a list of numerical features using the active MLflow model.",
        tags=["inference"],
    )
    @api.expect(predict_input, validate=True)
    @api.response(200, "Successful prediction", model=predict_output)
    @api.response(400, "Missing or invalid 'features'")
    @api.response(500, "Internal model prediction error")
    @api.marshal_with(predict_output)
    def post(self):
        """
        Accepts input features and returns a model prediction.

        Example input:
        {
            "features": [0.1, 0.5, 0.9, ...]
        }

        Response:
        {
            "prediction": 1
        }
        """
        data = request.get_json()
        features = data.get("features")

        if not features or not isinstance(features, list):
            api.abort(400, "Missing or invalid 'features'")

        try:
            prediction = model.predict([features])[0]
            return {"prediction": int(prediction)}
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            api.abort(500, str(e))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
