import os
import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_data(csv_path: str) -> pd.DataFrame:
    """Membaca dataset dari file CSV."""
    df = pd.read_csv(csv_path)
    return df

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Melatih Linear Regression dan mengembalikan metrik evaluasi."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2

def train_random_forest(X_train, X_test, y_train, y_test):
    """Melatih Random Forest dan mengembalikan metrik evaluasi."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, rmse, r2

def main():
    data_path = "HousingData_clean.csv"
    df = load_data(data_path)
    
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("Housing_Experiment")

    # Aktifkan MLflow autologging untuk model
    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="Linear_Regression"):
        model_lr, rmse_lr, r2_lr = train_linear_regression(X_train, X_test, y_train, y_test)
    
    with mlflow.start_run(run_name="Random_Forest"):
        model_rf, rmse_rf, r2_rf = train_random_forest(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()
