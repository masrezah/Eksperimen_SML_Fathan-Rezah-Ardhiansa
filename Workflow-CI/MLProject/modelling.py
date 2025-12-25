import os
import sys
import time
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

def load_data(csv_path: str) -> pd.DataFrame:
    """Membaca dataset dengan pengecekan path yang robust."""
    possible_paths = [
        csv_path,
        "HousingData_clean.csv",
        "../HousingData_clean.csv",
        "Workflow-CI/MLProject/HousingData_clean.csv"
    ]
    final_path = None
    for path in possible_paths:
        if os.path.exists(path):
            final_path = path
            break
    
    if not final_path:
        raise FileNotFoundError(f"Dataset not found. Checked: {possible_paths}")

    df = pd.read_csv(final_path)
    if "MEDV" not in df.columns:
        raise ValueError("Target column 'MEDV' missing")
    return df

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    print("="*60)
    print("🚀 STARTING TRAINING PIPELINE (DEBUG MODE)")
    print("="*60)

    # 1. FORCING AUTHENTICATION
    # Ini langkah paling krusial agar tidak perlu otorisasi manual
    token = os.getenv('MLFLOW_TRACKING_PASSWORD')
    if not token:
        print("❌ ERROR: MLFLOW_TRACKING_PASSWORD not found in env!")
        sys.exit(1)

    print("🔑 Authenticating with DagsHub...")
    dagshub.auth.add_app_token(token)
    
    # Init dengan repository kamu
    dagshub.init(repo_owner='rezahmas', repo_name='boston-housing-mlflow', mlflow=True)
    
    # Cek kemana data akan dikirim
    print(f"📡 Tracking URI: {mlflow.get_tracking_uri()}")
    
    mlflow.set_experiment("CI_Docker_Build")

    # 2. LOAD DATA
    try:
        df = load_data('HousingData_clean.csv')
        print(f"✓ Dataset loaded: {len(df)} rows")
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. TRAINING
    print("\n🔄 Training Models...")
    # Model: Linear Regression Pipeline
    lr_pipe = Pipeline([('scaler', StandardScaler()), ('lr', LinearRegression())])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test, lr_pred)
    
    print(f"✓ Model Trained (RMSE: {rmse:.4f})")

    # 4. LOGGING (CRITICAL PART)
    print("\n💾 Logging to MLflow...")
    mlflow.sklearn.autolog(log_models=False) # Matikan autolog model

    with mlflow.start_run(run_name="Best_Model_Selection") as run:
        run_id = run.info.run_id
        print(f"🆔 Run ID: {run_id}")
        
        # Cek Artifact URI (Harus s3://..., bukan file://...)
        artifact_uri = mlflow.get_artifact_uri()
        print(f"📂 Artifact Storage URI: {artifact_uri}")
        
        if "file://" in artifact_uri:
            print("⚠️ WARNING: Saving to LOCAL storage, not remote!")
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("model_type", "LinearRegression_Pipeline")
        
        print("📤 Uploading model artifact...")
        mlflow.sklearn.log_model(
            sk_model=lr_pipe,
            artifact_path="model", # Nama folder wajib 'model'
            input_example=X_train.iloc[:1]
        )
        print("✓ Model upload command executed.")
        
    # 5. VERIFIKASI LANGSUNG
    # Kita tunggu dan cek apakah benar-benar terupload sebelum menutup script
    print("\n⏳ Verifying upload (Waiting 10s)...")
    time.sleep(10)
    
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    artifacts = client.list_artifacts(run_id)
    paths = [a.path for a in artifacts]
    
    print(f"📋 Found artifacts in remote: {paths}")
    
    if "model" in paths:
        print("✅ SUCCESS: 'model' folder confirmed in remote storage!")
    else:
        print("❌ FAILED: 'model' folder missing from remote storage.")
        print("   This usually means 'boto3' is missing or network failed.")
        sys.exit(1) # Paksa error biar Pipeline berhenti disini

if __name__ == "__main__":
    main()