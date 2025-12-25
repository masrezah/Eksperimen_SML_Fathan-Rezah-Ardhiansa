import os
import sys
import numpy as np
import pandas as pd
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 1. FUNGSI UTILITAS
# ==========================================

def load_data(csv_path: str) -> pd.DataFrame:
    """Membaca dataset dengan pengecekan path yang robust."""
    print(f"🔍 Looking for dataset...")
    
    # Daftar kemungkinan lokasi file (untuk Local vs CI/CD)
    possible_paths = [
        csv_path,
        os.path.join("Workflow-CI", "MLProject", csv_path),
        os.path.join("MLProject", csv_path),
        "HousingData_clean.csv",
        "../HousingData_clean.csv"
    ]

    final_path = None
    for path in possible_paths:
        if os.path.exists(path):
            final_path = path
            print(f"✓ Found dataset at: {final_path}")
            break
    
    if not final_path:
        print(f"❌ ERROR: Dataset '{csv_path}' not found in any standard paths.")
        print(f"   Current Directory: {os.getcwd()}")
        print(f"   Directory Contents: {os.listdir('.')}")
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(final_path)
    
    # Validasi isian
    if df.empty:
        raise ValueError("Dataset is empty")
    if "MEDV" not in df.columns:
        raise ValueError("Target column 'MEDV' missing")
        
    return df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# ==========================================
# 2. TRAINING PIPELINE
# ==========================================

def main():
    print("\n" + "="*60)
    print("🏠 BOSTON HOUSING - TRAINING PIPELINE")
    print("="*60)

    # --- A. Setup Environment ---
    # Jika tracking URI tidak ada di env (Local Run), init DagsHub manual
    if not os.getenv('MLFLOW_TRACKING_URI'):
        print("⚙️  Initializing DagsHub (Local Mode)...")
        dagshub.init(repo_owner='rezahmas', repo_name='boston-housing-mlflow', mlflow=True)
    
    mlflow.set_experiment("CI_Docker_Build")

    # --- B. Load Data ---
    try:
        df = load_data('HousingData_clean.csv')
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- C. Training & Comparison ---
    print("\n🔄 Training Models...")
    
    # Model 1: Linear Regression (Pakai Pipeline untuk Scaling)
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    rmse_lr, mae_lr, r2_lr = eval_metrics(y_test, lr_pred)
    print(f"  👉 Linear Regression RMSE: {rmse_lr:.4f}")

    # Model 2: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rmse_rf, mae_rf, r2_rf = eval_metrics(y_test, rf_pred)
    print(f"  👉 Random Forest RMSE: {rmse_rf:.4f}")

    # Pilih Model Terbaik
    if rmse_lr < rmse_rf:
        best_model = lr_pipe
        best_name = "Linear_Regression"
        best_metrics = (rmse_lr, mae_lr, r2_lr)
    else:
        best_model = rf
        best_name = "Random_Forest"
        best_metrics = (rmse_rf, mae_rf, r2_rf)

    print(f"\n🏆 WINNER: {best_name}")

    # --- D. Logging ke MLflow (CRITICAL STEP) ---
    print("\n💾 Logging to MLflow...")
    
    # Matikan autolog model agar kita bisa set nama folder "model" secara manual
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run(run_name="Best_Model_Selection") as run:
        # Log Metrics
        mlflow.log_metric("rmse", best_metrics[0])
        mlflow.log_metric("mae", best_metrics[1])
        mlflow.log_metric("r2", best_metrics[2])
        
        # Log Params
        mlflow.log_param("best_model_name", best_name)
        
        # Log Model (INI BAGIAN PENTING UNTUK DOCKER)
        # artifact_path="model" adalah KUNCI agar GitHub Actions bisa menemukannya
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
            input_example=X_train.iloc[:1]
        )
        
        run_id = run.info.run_id
        print(f"✓ Model saved to: runs:/{run_id}/model")

    print("\n" + "="*60)
    print("✅ DONE")
    print("="*60)

if __name__ == "__main__":
    main()