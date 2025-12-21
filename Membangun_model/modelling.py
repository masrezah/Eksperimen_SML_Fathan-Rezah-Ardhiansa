# modelling.py
# Kriteria 2 - Advanced: MLflow dengan DagsHub remote tracking

import os
import numpy as np
import pandas as pd

import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data(csv_path: str) -> pd.DataFrame:
    """Membaca dataset preprocessing dari path yang diberikan."""
    df = pd.read_csv(csv_path)
    return df


def train_linear_regression(X_train, X_test, y_train, y_test):
    """Melatih Linear Regression dengan scaling dan mencetak metrik."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("=== Linear Regression ===")
    print(f"RMSE : {rmse:.3f}")
    print(f"R^2  : {r2:.3f}")

    return model, scaler, rmse, r2


def train_random_forest(X_train, X_test, y_train, y_test):
    """Melatih Random Forest dan mencetak metrik."""
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Random Forest Regressor ===")
    print(f"RMSE : {rmse:.3f}")
    print(f"R^2  : {r2:.3f}")

    return model, rmse, r2


def main():
    # 1. Load data preprocessing
    data_path = "HousingData_clean.csv"
    df = load_data(data_path)

    # 2. Pisahkan fitur dan target
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]

    # 3. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================================
    # 4. SETUP DAGSHUB (GANTI DENGAN DATA KAMU!)
    # ============================================
    dagshub.init(
        repo_owner='rezahmas',
        repo_name='boston-housing-mlflow',
        mlflow=True
    )

    mlflow.set_experiment("Boston_Housing_Experiment")

    # 5. Aktifkan autolog
    mlflow.sklearn.autolog(log_models=True)

    # --------- Run 1: Linear Regression ---------
    with mlflow.start_run(run_name="Linear_Regression"):
        model_lr, scaler_lr, rmse_lr, r2_lr = train_linear_regression(
            X_train, X_test, y_train, y_test
        )

    # --------- Run 2: Random Forest ---------
    with mlflow.start_run(run_name="Random_Forest"):
        model_rf, rmse_rf, r2_rf = train_random_forest(
            X_train, X_test, y_train, y_test
        )

    print("\n✅ Selesai! Cek hasil di DagsHub:")
    print("https://dagshub.com/USERNAME_KAMU/boston-housing-mlflow/experiments")


if __name__ == "__main__":
    main()