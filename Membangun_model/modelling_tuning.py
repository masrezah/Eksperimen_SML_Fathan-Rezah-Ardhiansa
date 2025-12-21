# modelling_tuning.py
# Kriteria 2 - Advanced: hyperparameter tuning + DagsHub remote tracking

import os
import numpy as np
import pandas as pd

import dagshub
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


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

    mlflow.set_experiment("Boston_Housing_Tuning")

    # 5. Definisikan model dasar dan grid hyperparameter
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=5,
        n_jobs=-1,
        verbose=1
    )

    # 6. Mulai run MLflow (manual logging)
    with mlflow.start_run(run_name="RandomForest_GridSearch"):
        grid_search.fit(X_train, y_train)

        best_params = grid_search.best_params_
        best_neg_mse = grid_search.best_score_
        best_rmse_cv = np.sqrt(-best_neg_mse)

        # 7. Train model terbaik
        best_rf = RandomForestRegressor(
            **best_params,
            random_state=42,
            n_jobs=-1
        )
        best_rf.fit(X_train, y_train)

        y_pred = best_rf.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
        r2_test = r2_score(y_test, y_pred)

        # 8. Manual logging
        mlflow.log_params(best_params)
        mlflow.log_metric("rmse_cv", best_rmse_cv)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_test", r2_test)

        print("Best Params  :", best_params)
        print(f"Best CV RMSE : {best_rmse_cv:.3f}")
        print(f"Test RMSE    : {rmse_test:.3f}")
        print(f"Test R^2     : {r2_test:.3f}")

        # 9. Simpan model
        mlflow.sklearn.log_model(best_rf, artifact_path="model")

    print("\n✅ Selesai! Cek hasil di DagsHub:")
    print("https://dagshub.com/USERNAME_KAMU/boston-housing-mlflow/experiments")


if __name__ == "__main__":
    main()