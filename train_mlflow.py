from loguru import logger
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd
import json
import warnings
import os
from datetime import datetime

# Load environment variable to determine the working environment (DEV/STAGING/PRODUCTION)
ENVIRONMENT = os.getenv("ENVIRONMENT", "DEV").upper()
logger.info(f"Running in environment: {ENVIRONMENT}")

# Define the experiment name and artifact storage location
EXPERIMENT_NAME = "mlflow_full_model_comparison"
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "file:/app/mlruns")

# Setup logging configurations
warnings.filterwarnings("ignore")  # Ignore warnings to keep logs clean
logger.add("training.log", rotation="1 MB", level="DEBUG")

# 1. Generate synthetic classification dataset
logger.info("Generating synthetic classification dataset...")
X, y = make_classification(
    n_samples=1000,  # number of samples
    n_features=20,  # number of features
    n_informative=15,  # number of informative features
    n_redundant=5,  # number of redundant features
    n_classes=2,  # number of classes
    random_state=42,  # random seed
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define model configurations and hyperparameter grids for different environments
logger.info("Defining model configs and hyperparameter grids...")

# Config for DEV environment (only LogisticRegression)
dev_model_configs = {
    "LogisticRegression": {
        "model_class": LogisticRegression,
        "param_grid": {
            "penalty": ["l2"],
            "C": [0.1, 1.0],
            "solver": ["liblinear"],
        },
    }
}

# Config for STAGING environment (LogisticRegression + RandomForest)
staging_model_configs = {
    "LogisticRegression": dev_model_configs["LogisticRegression"],
    "RandomForest": {
        "model_class": RandomForestClassifier,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap": [True],
        },
    },
}

# Config for PRODUCTION environment (adds XGBoost model)
production_model_configs = {
    **staging_model_configs,
    "XGBoost": {
        "model_class": XGBClassifier,
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.01, 0.1],
            "subsample": [0.7, 1.0],
            "colsample_bytree": [0.7, 1.0],
            "gamma": [0, 1],
            "reg_lambda": [0, 1],
            "reg_alpha": [0, 1],
        },
    },
}

# Select model configuration based on the environment
if ENVIRONMENT == "PRODUCTION":
    model_configs = production_model_configs
elif ENVIRONMENT == "STAGING":
    model_configs = staging_model_configs
else:
    model_configs = dev_model_configs

# 3. Setup MLflow experiment (create if not existing)
client = MlflowClient()
try:
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = client.create_experiment(
            EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT
        )
    else:
        experiment_id = experiment.experiment_id
except Exception as e:
    logger.error(f"Error setting up experiment: {e}")
    raise e

mlflow.set_experiment(EXPERIMENT_NAME)

# Variables to keep track of the best model
best_acc = 0.0
best_run_id = None
best_model_name = ""
best_params = None
results = []  # List to store results of each training
train_time = datetime.now().isoformat()  # Timestamp for training session

# 4. Perform grid search + training + logging
logger.info("Start model training and logging to MLflow...")
for model_name, config in model_configs.items():
    ModelClass = config["model_class"]
    param_grid = config["param_grid"]

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=model_name):
            logger.debug(f"Training {model_name} with params: {params}")
            try:
                # Special case handling for specific models
                if model_name == "XGBoost":
                    model = ModelClass(
                        **params, use_label_encoder=False, eval_metric="logloss"
                    )
                elif model_name == "LogisticRegression":
                    model = ModelClass(**params, max_iter=1000)
                else:
                    model = ModelClass(**params)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
    
                # Generate input example and model signature for better model reproducibility
                input_example = X_test[:1]
                signature = infer_signature(X_test, y_pred)

                # Log to MLflow
                mlflow.set_tag("train_time", train_time)
                mlflow.log_param("model_type", model_name)
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )

                # Save the result for reporting
                results.append({"model": model_name, "params": params, "accuracy": acc})

                logger.info(f"{model_name} - Accuracy: {acc:.4f}")

                # Update best model if current model is better
                if acc > best_acc:
                    best_acc = acc
                    best_run_id = mlflow.active_run().info.run_id
                    best_model_name = model_name
                    best_params = params

            except Exception as e:
                logger.error(f"{model_name} failed with params {params} → {e}")

# 5. Register the best model to the Model Registry
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"

    client = MlflowClient()
    artifacts = client.list_artifacts(best_run_id)

    # Check if model artifact exists
    has_model = any(a.path == "model" for a in artifacts)
    if has_model:
        logger.info("Registering best model to Model Registry...")

        result = mlflow.register_model(model_uri, "BestClassifierModel")
        version = result.version

        logger.success(f"Registered BestClassifierModel (version {version})")
        logger.success(f"Model will be available as models:/BestClassifierModel/latest")
        logger.success(f"Best model: {best_model_name} with acc={best_acc:.4f}")
        logger.success(f"Best params: {best_params}")
    else:
        logger.error(f"Model artifact 'model' not found in run {best_run_id}")
else:
    logger.warning("No valid run to register.")

# 6. Save all training results to JSON and CSV for easy reporting and review
with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
    logger.info("Saved results to model_comparison_results.json")

pd.DataFrame(results).to_csv("model_comparison_results.csv", index=False)
logger.info("Saved results to model_comparison_results.csv")
