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

ENVIRONMENT = os.getenv("ENVIRONMENT", "DEV").upper()
logger.info(f"Running in environment: {ENVIRONMENT}")

# Setup logging
warnings.filterwarnings("ignore")
logger.add("training.log", rotation="1 MB", level="DEBUG")

# 1. Generate classification data
logger.info("Generating synthetic classification dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    n_classes=2,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Define model configs and tuning hyperparameters
logger.info("Defining model configs and hyperparameter grids...")

# Model configs for DEV
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

# Model configs for STAGING
staging_model_configs = {
    "LogisticRegression": {
        "model_class": LogisticRegression,
        "param_grid": {
            "penalty": ["l2"],
            "C": [0.1, 1.0],
            "solver": ["liblinear"],
        },
    },
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
# Model configs for PRODUCTION
production_model_configs = {
    "LogisticRegression": {
        "model_class": LogisticRegression,
        "param_grid": {
            "penalty": ["l2"],
            "C": [0.1, 1.0],
            "solver": ["liblinear"],
        },
    },
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

if ENVIRONMENT == "PRODUCTION":
    model_configs = production_model_configs
elif ENVIRONMENT == "STAGING":
    model_configs = staging_model_configs
else:
    model_configs = dev_model_configs

# 3. Set MLflow experiment
mlflow.set_experiment("mlflow_full_model_comparison")

best_acc = 0.0
best_run_id = None
best_model_name = ""
best_params = None
results = []

# 4. Grid search + logging
logger.info("Start model training and logging to MLflow...")
for model_name, config in model_configs.items():
    ModelClass = config["model_class"]
    param_grid = config["param_grid"]

    for params in ParameterGrid(param_grid):
        with mlflow.start_run(run_name=model_name):
            logger.debug(f"Training {model_name} with params: {params}")
            try:
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

                # Generate input example and infer signature
                input_example = X_test[:1]
                signature = infer_signature(X_test, y_pred)

                # Logging to MLflow
                mlflow.log_param("model_type", model_name)
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", acc)
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    signature=signature,
                    input_example=input_example,
                )

                # Save result
                results.append({"model": model_name, "params": params, "accuracy": acc})

                logger.info(f"{model_name} - Accuracy: {acc:.4f}")

                if acc > best_acc:
                    best_acc = acc
                    best_run_id = mlflow.active_run().info.run_id
                    best_model_name = model_name
                    best_params = params

            except Exception as e:
                logger.error(f"{model_name} failed with params {params} → {e}")

# 5. Register best model (accessible via 'latest')
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"

    # Use MLflowClient to check if model path exists
    client = MlflowClient()
    artifacts = client.list_artifacts(best_run_id)

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

# 6. Save all results to JSON and CSV
with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
    logger.info("Saved results to model_comparison_results.json")

pd.DataFrame(results).to_csv("model_comparison_results.csv", index=False)
logger.info("Saved results to model_comparison_results.csv")
