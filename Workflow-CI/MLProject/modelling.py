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
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def load_data(csv_path: str) -> pd.DataFrame:
    """Membaca dataset preprocessing dari path yang diberikan."""
    
    # Get absolute path
    abs_path = os.path.abspath(csv_path)
    
    # Debug information
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Looking for dataset at: {abs_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found: {csv_path}")
        print(f"Absolute path checked: {abs_path}")
        
        # Try to find the file in common locations
        possible_paths = [
            'preprocessing/HousingData_clean.csv',
            '../preprocessing/HousingData_clean.csv',
            'HousingData_clean.csv',
            os.path.join(os.path.dirname(__file__), 'preprocessing/HousingData_clean.csv'),
        ]
        
        print("\n🔍 Searching in alternative locations:")
        for path in possible_paths:
            abs_alt = os.path.abspath(path)
            exists = os.path.exists(path)
            print(f"  - {path} → {abs_alt} [{'✓ EXISTS' if exists else '✗ NOT FOUND'}]")
            
            if exists:
                print(f"\n✓ Found dataset at: {path}")
                csv_path = path
                break
        else:
            # Still not found, list directory contents
            print("\n📁 Directory contents:")
            print("Current directory:")
            for item in os.listdir('.'):
                item_type = "📁" if os.path.isdir(item) else "📄"
                print(f"  {item_type} {item}")
            
            if os.path.exists('preprocessing'):
                print("\npreprocessing/ directory:")
                for item in os.listdir('preprocessing'):
                    item_type = "📁" if os.path.isdir(os.path.join('preprocessing', item)) else "📄"
                    print(f"  {item_type} {item}")
            
            raise FileNotFoundError(
                f"Dataset file not found at any of the checked locations. "
                f"Please ensure 'HousingData_clean.csv' exists in the preprocessing folder."
            )
    
    # Load dataset
    print(f"📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Validate target column
    if "MEDV" not in df.columns:
        raise ValueError(
            f"Target column 'MEDV' not found in dataset. "
            f"Available columns: {', '.join(df.columns)}"
        )
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Warning: Dataset contains {missing_count} missing values")
    
    return df

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Melatih Linear Regression dengan scaling dan mencetak metrik."""
    print("🔄 Training Linear Regression...")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Log additional metrics manually
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    
    # Log parameters
    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("scaler", "StandardScaler")

    print("\n" + "="*50)
    print("📊 Linear Regression Results")
    print("="*50)
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print("="*50)

    return model, scaler, rmse, r2

def train_random_forest(X_train, X_test, y_train, y_test):
    """Melatih Random Forest dan mencetak metrik."""
    print("🔄 Training Random Forest...")
    
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=10
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Log additional metrics manually
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2_score", r2)
    mlflow.log_metric("mae", mae)
    
    # Log parameters
    mlflow.log_param("model_type", "Random Forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 10)

    print("\n" + "="*50)
    print("📊 Random Forest Results")
    print("="*50)
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print("="*50)

    return model, rmse, r2

def main():
    print("\n" + "="*70)
    print("🏠 BOSTON HOUSING PRICE PREDICTION - ML PIPELINE")
    print("="*70)
    
    # Path ke dataset (support environment variable)
    data_path = os.getenv('DATA_PATH', 'preprocessing/HousingData_clean.csv')
    print(f"📍 Dataset path: {data_path}")
    
    # Load data dengan error handling
    try:
        df = load_data(data_path)
    except FileNotFoundError as e:
        print(f"\n❌ FATAL ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error loading data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Pisahkan fitur dan target
    X = df.drop("MEDV", axis=1)
    y = df["MEDV"]
    
    print(f"\n📊 Dataset Information:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  Feature columns: {list(X.columns)}")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n✂️  Train-Test Split:")
    print(f"  Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"  Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # Setup DagsHub dan MLflow
    print("\n" + "="*70)
    print("🔧 Setting up MLflow tracking...")
    print("="*70)
    
    try:
        dagshub.init(
            repo_owner='rezahmas',
            repo_name='boston-housing-mlflow',
            mlflow=True
        )
        print("✓ DagsHub initialized")
        
        mlflow.set_experiment("Boston_Housing_Experiment")
        print("✓ Experiment set: Boston_Housing_Experiment")
        
        mlflow.sklearn.autolog(log_models=True)
        print("✓ MLflow autolog enabled")
        
    except Exception as e:
        print(f"⚠️  Warning: MLflow setup issue: {e}")
        print("Continuing without MLflow tracking...")

    # --------- Run 1: Linear Regression ---------
    print("\n" + "="*70)
    print("🚀 MODEL 1: LINEAR REGRESSION")
    print("="*70)
    
    try:
        with mlflow.start_run(run_name="Linear_Regression"):
            model_lr, scaler_lr, rmse_lr, r2_lr = train_linear_regression(
                X_train, X_test, y_train, y_test
            )
            print("✓ Linear Regression training completed")
    except Exception as e:
        print(f"❌ Error training Linear Regression: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --------- Run 2: Random Forest ---------
    print("\n" + "="*70)
    print("🚀 MODEL 2: RANDOM FOREST REGRESSOR")
    print("="*70)
    
    try:
        with mlflow.start_run(run_name="Random_Forest"):
            model_rf, rmse_rf, r2_rf = train_random_forest(
                X_train, X_test, y_train, y_test
            )
            print("✓ Random Forest training completed")
    except Exception as e:
        print(f"❌ Error training Random Forest: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Final Summary
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\n📊 Model Comparison:")
    print(f"  {'Model':<25} {'RMSE':<12} {'R² Score':<12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Linear Regression':<25} {rmse_lr:<12.3f} {r2_lr:<12.3f}")
    print(f"  {'Random Forest':<25} {rmse_rf:<12.3f} {r2_rf:<12.3f}")
    
    # Determine best model
    best_model = "Linear Regression" if rmse_lr < rmse_rf else "Random Forest"
    print(f"\n🏆 Best Model: {best_model}")
    
    print(f"\n🔗 View detailed results at:")
    print(f"  https://dagshub.com/rezahmas/boston-housing-mlflow/experiments")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR in main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)