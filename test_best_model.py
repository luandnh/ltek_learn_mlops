import json
import argparse
import mlflow.sklearn
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Test best MLflow model from mlruns")
parser.add_argument("--mlruns", default="mlruns", help="Path to mlruns directory")
parser.add_argument("--experiment", default="0", help="Experiment ID")
parser.add_argument(
    "--input-size", type=int, default=20, help="Number of features expected"
)
args = parser.parse_args()

# Step 1: Load the results
with open("model_comparison_results.json") as f:
    results = json.load(f)

# Step 2: Find the best model (highest accuracy)
best = max(results, key=lambda r: r["accuracy"])
print("Best model info:", best)

# Step 3: Reconstruct path to model inside mlruns
mlruns_path = Path(args.mlruns) / args.experiment
best_run_id = None

for run_dir in mlruns_path.iterdir():
    if not run_dir.is_dir():
        continue
    try:
        params_path = run_dir / "params"
        run_params = {
            p.name: (params_path / p.name).read_text().strip()
            for p in params_path.iterdir()
        }
        if run_params.get("model_type") == best["model"] and all(
            str(best["params"].get(k)) == run_params.get(k) for k in best["params"]
        ):
            best_run_id = run_dir.name
            break
    except Exception:
        continue

if best_run_id is None:
    raise ValueError("Could not locate best model run in mlruns")

# Step 4: Load model from mlruns
model_path = f"{args.mlruns}/{args.experiment}/{best_run_id}/artifacts/model"
model = mlflow.sklearn.load_model(model_path)

# Step 5: Predict
sample = np.random.rand(1, args.input_size)  # assuming 20 feature input by default
pred = model.predict(sample)
print("Sample prediction:", pred)
