import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ==========================================
# 1. Persiapan Data
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(script_dir, 'housing_preprocessing', 'clean_housing_data.csv')

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File tidak ditemukan di: {DATA_PATH}")

print("Memuat data...")
df = pd.read_csv(DATA_PATH)

X = df.drop('MEDV', axis=1)
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. Pelatihan Model
# ==========================================
print("Melatih Model...")
n_estimators = 100
max_depth = 10

model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 3. Evaluasi Model
# ==========================================
print("Evaluasi Model...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Hasil -> MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

# ==========================================
# 4. MLflow Tracking (SOLUSI BIAR GAK ERROR DI GITHUB)
# ==========================================
print("Menyimpan log ke MLflow...")

# Cek apakah sedang dijalankan oleh GitHub Actions (mlflow run)
if os.environ.get('MLFLOW_RUN_ID'):
    # JIKA DI GITHUB: Jangan set_experiment! Langsung log ke ID yang sudah disiapkan.
    print("Mode GitHub Actions terdeteksi.")
    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")
else:
    # JIKA DI LAPTOP (MANUAL): Baru bikin eksperimen sendiri.
    print("Mode Lokal terdeteksi.")
    mlflow.set_experiment("Housing_Price_Prediction")
    with mlflow.start_run(run_name="RandomForest_Local"):
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")

print("Selesai! Log tersimpan.")