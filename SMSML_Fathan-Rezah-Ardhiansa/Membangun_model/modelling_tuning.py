import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# 1. Load Data
# Pastikan file ini ada di folder housing_preprocessing
DATA_PATH = 'housing_preprocessing/clean_housing_data.csv'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"File {DATA_PATH} tidak ditemukan! Pastikan Anda sudah copy file dari Kriteria 1.")

print("Memuat data...")
df = pd.read_csv(DATA_PATH)

# Pisahkan Fitur dan Target (Harga Rumah / MEDV)
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Hyperparameter Tuning (Syarat Skilled)
print("Memulai Tuning Model...")
rf = RandomForestRegressor(random_state=42)

# Parameter yang akan dicoba otomatis
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, None]
}

# Mencari settingan terbaik
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_params = grid_search.best_params_
print(f"Parameter Terbaik: {best_params}")

# 3. Evaluasi
y_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Hasil -> MAE: {mae:.2f}, MSE: {mse:.2f}, R2: {r2:.2f}")

# 4. MLflow Tracking (Manual Logging)
mlflow.set_experiment("Housing_Price_Prediction_Skilled")

print("Menyimpan ke MLflow...")
with mlflow.start_run(run_name="RandomForest_Tuning"):
    # Log Parameter Terbaik
    mlflow.log_param("n_estimators", best_params['n_estimators'])
    mlflow.log_param("max_depth", best_params['max_depth'])
    
    # Log Metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    
    # Log Model (Simpan Artefak)
    mlflow.sklearn.log_model(best_model, "model")

print("Selesai! Silakan cek MLflow UI.")