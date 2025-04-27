from loguru import logger
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from scipy.stats import uniform
from xgboost import XGBClassifier
import warnings
import os
import numpy as np
from datetime import datetime
import json
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

# Define the experiment name and artifact storage location
EXPERIMENT_NAME = "mlflow_mlops_demo"
ARTIFACT_ROOT = os.getenv("MLFLOW_ARTIFACT_ROOT", "file:/app/mlruns")

# Setup logging configurations
warnings.filterwarnings("ignore")  # Ignore warnings to keep logs clean
logger.add("training.log", rotation="1 MB", level="DEBUG")

# Setup MLflow experiment (create if not existing)
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


# 1. Generate synthetic classification dataset
logger.info("Generating synthetic classification dataset...")
X, y = make_classification(
    n_samples=1000,  # number of samples
    n_features=20,  # number of features
    n_informative=8,  # number of informative features
    n_redundant=5,  # number of redundant features
    n_classes=2,  # number of classes
    random_state=42,  # random seed
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Train Basic Logistic Regression model
logger.info("Training Basic Logistic Regression model...")
basic_model = LogisticRegression(random_state=42)
basic_model.fit(X_train, y_train)

# 2.1 Evaluate Basic Logistic Regression model
logger.info("Evaluating Basic Logistic Regression model...")
basic_predictions = basic_model.predict(X_test)
basic_accuracy = accuracy_score(y_test, basic_predictions)
basic_classification_rep = classification_report(y_test, basic_predictions)

logger.info("Basic Logistic Regression - Accuracy: {:.4f}".format(basic_accuracy))
logger.info(
    "Basic Logistic Regression - Classification Report:\n{}".format(
        basic_classification_rep
    )
)

# 2.2 Report Basic Logistic Regression model to MLflow
# Accuracy: 0.6633
# Classification Report:
#       precision    recall  f1-score   support
#    0       0.63      0.67      0.65       140
#    1       0.70      0.66      0.68       160
#
# Class 0:
# F1-score: 0.65
# Class 1:
# F1-score: 0.68
# Nhận xét:
# - Mô hình cân bằng giữa hai lớp, nhưng độ chính xác còn thấp.
# - Precision của lớp 1 cao hơn lớp 0 → mô hình dự đoán lớp 1 "chắc chắn" hơn.
# - Khoảng 1/3 mẫu bị dự đoán sai → cần cải thiện.
# Đề xuất:
# Tuning siêu tham số (C, solver, max_iter).
# Tăng kích thước dữ liệu, thêm nhiễu có kiểm soát.

# 2.3 Save Basic Logistic Regression model to MLflow
with mlflow.start_run(run_name="Basic Logistic Regression"):
    mlflow.set_tag("train_time", train_time)
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_metric("accuracy", basic_accuracy)
    signature = infer_signature(X_train, basic_model.predict(X_train))
    input_example = X_train[:1]
    mlflow.sklearn.log_model(
        basic_model, "model", signature=signature, input_example=input_example
    )
    if basic_accuracy > best_acc:
        best_acc = basic_accuracy
        best_run_id = mlflow.active_run().info.run_id
        best_model_name = "Basic Logistic Regression"

results.append(
    {
        "model": "Basic Logistic Regression",
        "accuracy": basic_accuracy,
    }
)

# 3. Tunning Logistic Regression hyperparameters with RandomizedSearchCV
logger.info("Tunning Logistic Regression hyperparameters with RandomizedSearchCV...")
param_dist_lr = {
    "C": uniform(0.01, 100),  # Regularization strength
    "solver": ["liblinear", "lbfgs"],  # Different solvers
    "max_iter": [100, 200, 500],  # Iteration limits
}

random_search_lr = RandomizedSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_distributions=param_dist_lr,
    n_iter=10,  # Number of hyperparameter combinations to try
    scoring="accuracy",  # Use accuracy as the evaluation metric
    cv=5,  # 5-fold cross-validation
    verbose=0,  # Suppress verbose output
    random_state=42,  # For reproducibility
)

random_search_lr.fit(X_train, y_train)

# # Retrieve best model and parameters
tuned_best_model = random_search_lr.best_estimator_
tuned_best_params = random_search_lr.best_params_
tuned_best_accuracy = random_search_lr.best_score_


# 3.1 Evaluate Tuned Logistic Regression model
tuned_predictions = tuned_best_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, tuned_predictions)
tuned_report = classification_report(y_test, tuned_predictions)

logger.info("Tuned Logistic Regression - Best Parameters: {}".format(tuned_best_params))

logger.info(
    "Best Model: {} - Best Accuracy: {:.4f}".format(
        tuned_best_model, tuned_best_accuracy
    )
)

logger.info(
    "Tuned Logistic Regression - Test Set Accuracy: {:.4f}".format(tuned_accuracy)
)
logger.info(
    "Tuned Logistic Regression - Classification Report:\n{}".format(tuned_report)
)


# 3.2 Report Tuned Logistic Regression model
# Best Parameters: {'C': 37.464011884736244, 'max_iter': 100, 'solver': 'liblinear'}
# Accuracy: 0.6633
# Classification Report:
#       precision    recall  f1-score   support
#    0       0.63      0.67      0.65       140
#    1       0.70      0.66      0.68       160
#
# Class 0:
# F1-score: 0.65
# Class 1:
# F1-score: 0.68

# Nhận xét:
# F1-score (và Accuracy) giữa bước 2 (Basic Logistic Regression) và bước 3 (Tuned Logistic Regression) hầu như không thay đổi.
# Lý do:
# - Dữ liệu đã được tạo synthetic (make_classification) khá đơn giản, ít nhiễu, số feature cũng nhỏ (10 features).
# - Logistic Regression là mô hình rất đơn giản, và nó ít bị ảnh hưởng bởi việc tuning các hyperparameters nhẹ như C, solver, max_iter.
# - RandomizedSearchCV hay GridSearchCV đang tuning trong khoảng nhỏ (C từ 0.01 đến 100) mà bài toán không đủ phức tạp -> không tạo ra cải thiện đáng kể.

# Đề xuất:
# - Tăng n_informative và n_redundant chênh lệch hơn
# - Thêm noise mạnh hơn vào features
# - Thử C từ 1e-4 tới 1e4 rộng hơn
# - Thêm tham số penalty (l1, l2, elasticnet) nếu solver hỗ trợ

# 3.3 Save Tuned Logistic Regression model to MLflow
with mlflow.start_run(run_name="Tuned Logistic Regression"):
    mlflow.set_tag("train_time", train_time)
    mlflow.log_params(random_search_lr.best_params_)
    mlflow.log_metric("accuracy", tuned_accuracy)
    signature = infer_signature(X_train, tuned_best_model.predict(X_train))
    input_example = X_train[:1]
    mlflow.sklearn.log_model(
        tuned_best_model, "model", signature=signature, input_example=input_example
    )
    if tuned_accuracy > best_acc:
        best_acc = tuned_accuracy
        best_run_id = mlflow.active_run().info.run_id
        best_model_name = "Tuned Logistic Regression"

results.append(
    {
        "model": "Tuned Logistic Regression",
        "accuracy": tuned_accuracy,
    }
)

# 4. Data Augmentation + Noising
logger.info("Data Augmentation + Noising...")
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=16,
    n_redundant=4,
    n_classes=2,
    random_state=42,
)

noise = np.random.normal(0, 0.5, X.shape)
X = X + noise

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

param_dist_lr = {
    "C": uniform(1e-4, 1e4),
    "solver": ["liblinear", "saga", "lbfgs"],
    "penalty": ["l1", "l2"],
    "max_iter": [100, 300, 500],
}

aug_random_search_lr = RandomizedSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_distributions=param_dist_lr,
    n_iter=10,  # Number of hyperparameter combinations to try
    scoring="accuracy",  # Use accuracy as the evaluation metric
    cv=5,  # 5-fold cross-validation
    verbose=0,  # Suppress verbose output
    random_state=42,  # For reproducibility
)


aug_random_search_lr.fit(X_train, y_train)

# # Retrieve best model and parameters
aug_best_model = aug_random_search_lr.best_estimator_
aug_best_params = aug_random_search_lr.best_params_
aug_best_accuracy = aug_random_search_lr.best_score_


# 4.1 Evaluate Tuned Logistic Regression model
aug_predictions = aug_best_model.predict(X_test)
aug_accuracy = accuracy_score(y_test, aug_predictions)
aug_report = classification_report(y_test, aug_predictions)

logger.info(
    "Tuned + Data Augmentation + Noising - Best Parameters: {}".format(aug_best_params)
)

logger.info(
    "Best Model: {} - Best Accuracy: {:.4f}".format(aug_best_model, aug_best_accuracy)
)

logger.info(
    "Tuned + Data Augmentation + Noising - Test Set Accuracy: {:.4f}".format(
        aug_accuracy
    )
)
logger.info(
    "Tuned + Data Augmentation + Noising - Classification Report:\n{}".format(
        aug_report
    )
)

# 4.2 Report Tuned Logistic Regression model + Data Augmentation + Noising
# Best Parameters: {'C': 6011.1502174320885, 'max_iter': 500, 'penalty': 'l2', 'solver': 'liblinear'}
# Accuracy: 0.7700
# Classification Report:
#       precision    recall  f1-score   support
#    0       0.76      0.78      0.77       150
#    1       0.78      0.76      0.77       150
#
# Class 0:
# F1-score: 0.77
# Class 1:
# F1-score: 0.77

# Nhận xét:
# - Cải thiện rõ rệt so với Logistic Regression ban đầu, accuracy tăng từ mức khoảng 66% lên 77%, F1-score giữa hai lớp rất cân bằng (77%-77%).
# - C value lớn (~6000) => cho thấy regularization rất yếu, model gần như trở thành Linear Regression thuần, ít ràng buộc độ lớn trọng số.

# Đề xuất:
# - Sử dựng các mô hình phi tuyến mạnh hơn như RandomForest hoặc XGBoost.

# 4.3 Save Tuned Logistic Regression model + Data Augmentation + Noising to MLflow
with mlflow.start_run(run_name="Tuned Logistic Regression + Augmentation"):
    mlflow.set_tag("train_time", train_time)
    mlflow.log_params(aug_random_search_lr.best_params_)
    mlflow.log_metric("accuracy", aug_accuracy)
    signature = infer_signature(X_train, aug_best_model.predict(X_train))
    input_example = X_train[:1]
    mlflow.sklearn.log_model(
        aug_best_model, "model", signature=signature, input_example=input_example
    )
    if aug_accuracy > best_acc:
        best_acc = aug_accuracy
        best_run_id = mlflow.active_run().info.run_id
        best_model_name = "Tuned Logistic Regression + Augmentation"

results.append(
    {
        "model": "Tuned Logistic Regression + Augmentation",
        "accuracy": aug_accuracy,
    }
)

# 5. Train XGBoost
logger.info("Training XGBoost model...")
xgb_model = XGBClassifier(
    random_state=42,
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric="logloss",
    tree_method="hist",
)

# 5.1 Evaluate XGBoost model
xgb_model.fit(X_train, y_train)

xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_classification_report = classification_report(y_test, xgb_predictions)

logger.info("XGBoost - Accuracy: {:.4f}".format(xgb_accuracy))
logger.info("XGBoost - Classification Report:\n{}".format(xgb_classification_report))


# 5.2 Report XGBoost model
# Accuracy: 0.8967
# Classification Report:
#       precision    recall  f1-score   support

#    0       0.89      0.89      0.89       150
#    1       0.89      0.89      0.89       150

# Class 0:
# F1-score: 0.89
# Class 1:
# F1-score: 0.89
#
# Nhận xét:
# - So với Logistic Regression (77%) XGBoost đã đạt tới 89% accuracy.
# - Không có dấu hiệu bias nghiêng về lớp 0 hay lớp 1.

# 5.3 Save XGBoost model to MLflow
with mlflow.start_run(run_name="XGBoost"):
    mlflow.set_tag("train_time", train_time)
    mlflow.log_param("model", "XGBoost")
    mlflow.log_metric("accuracy", xgb_accuracy)
    signature = infer_signature(X_train, xgb_model.predict(X_train))
    input_example = X_train[:1]
    mlflow.sklearn.log_model(
        xgb_model, "model", signature=signature, input_example=input_example
    )
    if xgb_accuracy > best_acc:
        best_acc = xgb_accuracy
        best_run_id = mlflow.active_run().info.run_id
        best_model_name = "XGBoost"

results.append(
    {
        "model": "XGBoost",
        "accuracy": xgb_accuracy,
    }
)

# Nhận xét tổng thể:
# Qua quá trình huấn luyện và so sánh:
# - Logistic Regression cơ bản đạt Accuracy ~66%, thể hiện hạn chế khi xử lý bài toán, dù kết quả khá cân bằng giữa hai lớp.
# - Tuning hyperparameters với Logistic Regression không mang lại cải thiện đáng kể do dữ liệu synthetic đơn giản và tính chất tuyến tính của mô hình.
# - Khi áp dụng kỹ thuật Data Augmentation và thêm nhiễu có kiểm soát, Logistic Regression đã cải thiện đáng kể, nâng Accuracy lên ~77%.
# - Sử dụng XGBoost giúp đạt Accuracy vượt trội ~89%, đồng thời giữ F1-score cân bằng giữa các lớp, cho thấy mô hình mạnh mẽ và ổn định.

# Đề xuất tiếp theo:
# - Thử nghiệm thêm các mô hình mạnh khác như Random Forest, LightGBM.
# - Áp dụng kỹ thuật Cross Validation để đánh giá mô hình chính xác hơn.
# - Thực hiện thêm Hyperparameter Optimization với Optuna hoặc RandomizedSearchCV cho các mô hình nâng cao.


# 6. Register the best model to the Model Registry
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    artifacts = client.list_artifacts(best_run_id)
    has_model = any(a.path == "model" for a in artifacts)
    if has_model:
        logger.info("Registering best model to Model Registry...")
        result = mlflow.register_model(model_uri, "BestClassifierModel")
        version = result.version
        logger.success(f"Registered BestClassifierModel (version {version})")
        logger.success(
            "Best model: {} with acc={:.4f}".format(best_model_name, best_acc)
        )
        logger.success(f"Best params: {best_params}")
    else:
        logger.error(f"Model artifact 'model' not found in run {best_run_id}")
else:
    logger.warning("No valid run to register.")

# 7. Save all training results to JSON and CSV
with open("model_comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
    logger.info("Saved results to model_comparison_results.json")

pd.DataFrame(results).to_csv("model_comparison_results.csv", index=False)
logger.info("Saved results to model_comparison_results.csv")
