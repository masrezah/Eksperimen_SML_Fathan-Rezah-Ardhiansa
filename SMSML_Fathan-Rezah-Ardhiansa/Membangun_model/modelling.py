import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# ==========================================
# 1. Persiapan Data (PATH FIX)
# ==========================================
# Ambil lokasi absolut di mana script ini (modelling.py) berada
script_dir = os.path.dirname(os.path.abspath(__file__))

# Gabungkan lokasi script dengan folder data
# Ini akan menghasilkan path: .../Membangun_model/housing_preprocessing/clean_housing_data.csv
DATA_PATH = os.path.join(script_dir, 'housing_preprocessing', 'clean_housing_data.csv')

print(f"Mencari data di: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File tidak ditemukan di: {DATA_PATH}. \nPastikan file 'clean_housing_data.csv' ada di folder 'housing_preprocessing'.")

print("Memuat data...")
df = pd.read_csv(DATA_PATH)

# Pisahkan Fitur (X) dan Target (y)
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. Pelatihan Model (Basic)
# ==========================================
print("Melatih Model...")

# Kita gunakan parameter tetap (fixed) untuk modelling.py (bukan tuning)
n_estimators = 100
max_depth = 10

# Inisialisasi Model
model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

# Training
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
# 4. MLflow Tracking (FIXED)
# ==========================================
print("Menyimpan log ke MLflow...")

# Hapus set_experiment dan run_name. 
# Biarkan mlflow mengambil active run yang dibuat oleh perintah 'mlflow run'
if mlflow.active_run():
    with mlflow.start_run():
        # Log Metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        
        # Log Parameter
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # Log Model
        mlflow.sklearn.log_model(model, "model")
else:
    # Fallback jika dijalankan manual python biasa (bukan mlflow run)
    mlflow.set_experiment("Housing_Price_Prediction_Basic")
    with mlflow.start_run(run_name="RandomForest_Basic"):
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.sklearn.log_model(model, "model")

print("Selesai! Model berhasil disimpan.")