import os
import sys
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ==========================================
# 1. FUNGSI LOAD DATA (ROBUST)
# ==========================================
def load_data(csv_filename):
    """
    Mencari file dataset di berbagai kemungkinan folder
    agar tidak error 'File Not Found' di GitHub Actions.
    """
    possible_paths = [
        csv_filename,                                   # Di folder yang sama
        os.path.join("Workflow-CI", "MLProject", csv_filename), # Dari root repo
        os.path.join("MLProject", csv_filename),        # Dari folder parent
        "../HousingData_clean.csv",                     # Di folder atas
        "HousingData_clean.csv"                         # Fallback
    ]

    final_path = None
    for path in possible_paths:
        if os.path.exists(path):
            final_path = path
            print(f"✓ Dataset ditemukan di: {final_path}")
            break
    
    if final_path is None:
        # Debugging: Tampilkan isi folder saat ini jika file tidak ketemu
        print(f"❌ Error: File {csv_filename} tidak ditemukan.")
        print(f"   Posisi Folder saat ini: {os.getcwd()}")
        print(f"   Isi Folder: {os.listdir('.')}")
        sys.exit(1)

    df = pd.read_csv(final_path)
    
    # Validasi sederhana
    if "MEDV" not in df.columns:
        print("❌ Error: Kolom target 'MEDV' tidak ada di dataset.")
        sys.exit(1)
        
    return df

# ==========================================
# 2. TRAINING & EVALUASI
# ==========================================
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def main():
    print("="*50)
    print("🚀 STARTING TRAINING PIPELINE (LOCAL MODE)")
    print("="*50)

    # --- KONFIGURASI MLFLOW (KUNCI SUKSES) ---
    # Kita gunakan SQLite lokal. Ini menjamin pipeline TIDAK AKAN GAGAL 
    # karena masalah koneksi internet atau password DagsHub yang salah.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    
    # Nama eksperimen disesuaikan dengan YAML
    experiment_name = "CI_Drive_Build"
    mlflow.set_experiment(experiment_name)
    
    print(f"📡 Tracking URI: sqlite:///mlflow.db")
    print(f"🧪 Experiment: {experiment_name}")

    # --- LOAD DATA ---
    df = load_data("HousingData_clean.csv")

    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n🔄 Sedang melatih model...")

    # --- MODEL 1: Linear Regression (dengan Scaling) ---
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    lr_pipe.fit(X_train, y_train)
    lr_pred = lr_pipe.predict(X_test)
    rmse, mae, r2 = eval_metrics(y_test, lr_pred)
    
    print(f"   👉 Model Trained (RMSE: {rmse:.4f})")

    # --- LOGGING KE MLFLOW ---
    print("\n💾 Menyimpan model ke MLflow...")
    
    # Matikan autolog model supaya kita bisa simpan manual dengan nama folder yang pas
    mlflow.sklearn.autolog(log_models=False)

    with mlflow.start_run() as run:
        # Log Metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("model_type", "LinearRegression_Pipeline")

        # Log Model (PENTING: artifact_path harus "model")
        # Ini supaya step 'mlflow models build-docker' bisa menemukannya.
        mlflow.sklearn.log_model(
            sk_model=lr_pipe,
            artifact_path="model",
            input_example=X_train.iloc[:1]
        )
        
        run_id = run.info.run_id
        print(f"✅ Model berhasil disimpan!")
        print(f"🆔 Run ID: {run_id}")
        print(f"📂 Lokasi Artifact: runs:/{run_id}/model")

if __name__ == "__main__":
    main()