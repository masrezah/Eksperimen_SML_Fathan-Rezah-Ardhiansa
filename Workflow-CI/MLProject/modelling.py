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
from sklearn.pipeline import Pipeline

def load_data(csv_path: str) -> pd.DataFrame:
    """Membaca dataset dari path yang diberikan."""
    abs_path = os.path.abspath(csv_path)
    print(f"🔍 Current working directory: {os.getcwd()}")
    print(f"🔍 Looking for dataset at: {abs_path}")
    
    if not os.path.exists(csv_path):
        print(f"❌ ERROR: File not found: {csv_path}")
        possible_paths = [
            'HousingData_clean.csv',
            'preprocessing/HousingData_clean.csv',
            '../HousingData_clean.csv',
            os.path.join(os.path.dirname(__file__), 'HousingData_clean.csv'),
        ]
        print("\n🔍 Searching in alternative locations:")
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✓ Found dataset at: {path}")
                csv_path = path
                break
        else:
            print("\n📁 Current directory contents:")
            for item in os.listdir('.'):
                item_type = "📁" if os.path.isdir(item) else "📄"
                print(f"  {item_type} {item}")
            raise FileNotFoundError(
                f"Dataset file not found. Please ensure 'HousingData_clean.csv' exists."
            )
    
    print(f"📂 Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"⚠️  Warning: Dataset contains {missing_count} missing values")
        raise ValueError("Dataset contains missing values! Use cleaned dataset.")
    else:
        print("✓ Dataset is clean - no missing values")
    
    if "MEDV" not in df.columns:
        raise ValueError(
            f"Target column 'MEDV' not found. Available: {', '.join(df.columns)}"
        )
    
    return df

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Melatih Linear Regression dengan scaling."""
    print("🔄 Training Linear Regression...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print("\n" + "="*50)
    print("📊 Linear Regression Results")
    print("="*50)
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print("="*50)
    
    return pipeline, rmse, r2, mae

def train_random_forest(X_train, X_test, y_train, y_test):
    """Melatih Random Forest."""
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
    
    print("\n" + "="*50)
    print("📊 Random Forest Results")
    print("="*50)
    print(f"RMSE : {rmse:.3f}")
    print(f"R²   : {r2:.3f}")
    print(f"MAE  : {mae:.3f}")
    print("="*50)
    
    return model, rmse, r2, mae

def main():
    print("\n" + "="*70)
    print("🏠 BOSTON HOUSING PRICE PREDICTION - ML PIPELINE")
    print("="*70)
    
    data_path = os.getenv('DATA_PATH', 'HousingData_clean.csv')
    print(f"📍 Dataset path: {data_path}")
    
    # Load data
    try:
        df = load_data(data_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
    
    # Separate features and target
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
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    # Setup MLflow
    print("\n" + "="*70)
    print("🔧 Setting up MLflow tracking...")
    print("="*70)
    
    try:
        # Initialize DagsHub
        dagshub.init(
            repo_owner='rezahmas',
            repo_name='boston-housing-mlflow',
            mlflow=True
        )
        print("✓ DagsHub initialized")
        
        # Set experiment
        experiment_name = os.getenv('MLFLOW_EXPERIMENT_NAME', 'CI_Docker_Build')
        mlflow.set_experiment(experiment_name)
        print(f"✓ Experiment set: {experiment_name}")
        
        # ✅ DISABLE autolog to avoid conflicts
        mlflow.sklearn.autolog(disable=True)
        print("✓ Autolog disabled (using manual logging)")
        
    except Exception as e:
        print(f"⚠️  Warning: MLflow setup issue: {e}")
        print("Continuing anyway...")
    
    # Store results
    results = []
    
    # Train Linear Regression
    print("\n" + "="*70)
    print("🚀 MODEL 1: LINEAR REGRESSION")
    print("="*70)
    
    with mlflow.start_run(run_name="Linear_Regression") as run:
        model_lr, rmse_lr, r2_lr, mae_lr = train_linear_regression(
            X_train, X_test, y_train, y_test
        )
        
        # Log metrics
        mlflow.log_metric("rmse", rmse_lr)
        mlflow.log_metric("r2_score", r2_lr)
        mlflow.log_metric("mae", mae_lr)
        
        # Log parameters
        mlflow.log_param("model_type", "Linear_Regression")
        mlflow.log_param("scaler", "StandardScaler")
        
        results.append({
            'name': 'Linear_Regression',
            'model': model_lr,
            'rmse': rmse_lr,
            'r2': r2_lr,
            'mae': mae_lr,
            'run_id': run.info.run_id
        })
        
        print("✓ Linear Regression completed")
    
    # Train Random Forest
    print("\n" + "="*70)
    print("🚀 MODEL 2: RANDOM FOREST REGRESSOR")
    print("="*70)
    
    with mlflow.start_run(run_name="Random_Forest") as run:
        model_rf, rmse_rf, r2_rf, mae_rf = train_random_forest(
            X_train, X_test, y_train, y_test
        )
        
        # Log metrics
        mlflow.log_metric("rmse", rmse_rf)
        mlflow.log_metric("r2_score", r2_rf)
        mlflow.log_metric("mae", mae_rf)
        
        # Log parameters
        mlflow.log_param("model_type", "Random_Forest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 10)
        
        results.append({
            'name': 'Random_Forest',
            'model': model_rf,
            'rmse': rmse_rf,
            'r2': r2_rf,
            'mae': mae_rf,
            'run_id': run.info.run_id
        })
        
        print("✓ Random Forest completed")
    
    # Determine best model
    best_result = min(results, key=lambda x: x['rmse'])
    
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)
    print(f"  {'Model':<25} {'RMSE':<12} {'R² Score':<12} {'MAE':<12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for res in results:
        marker = "🏆" if res['name'] == best_result['name'] else "  "
        print(f"{marker} {res['name']:<25} {res['rmse']:<12.3f} {res['r2']:<12.3f} {res['mae']:<12.3f}")
    
    # ✅ LOG BEST MODEL with explicit error handling
    print("\n" + "="*70)
    print(f"🏆 LOGGING BEST MODEL: {best_result['name']}")
    print("="*70)
    
    try:
        with mlflow.start_run(run_name=f"BEST_MODEL_{best_result['name']}") as run:
            # Log metrics
            mlflow.log_metric("rmse", best_result['rmse'])
            mlflow.log_metric("r2_score", best_result['r2'])
            mlflow.log_metric("mae", best_result['mae'])
            
            # Log parameters
            mlflow.log_param("model_type", best_result['name'])
            mlflow.log_param("selected_as_best", "True")
            mlflow.log_param("selection_metric", "rmse")
            
            print(f"✓ Metrics logged")
            print(f"  - RMSE: {best_result['rmse']:.3f}")
            print(f"  - R²: {best_result['r2']:.3f}")
            print(f"  - MAE: {best_result['mae']:.3f}")
            
            # ✅ CRITICAL: Log model with explicit artifact path
            print("\n📦 Logging model artifact...")
            
            # Get input example for model signature
            input_example = X_train.iloc[:5]
            
            mlflow.sklearn.log_model(
                sk_model=best_result['model'],
                artifact_path="model",
                registered_model_name=None,  # Don't register yet
                input_example=input_example
            )
            
            best_run_id = run.info.run_id
            
            print(f"✅ Model logged successfully!")
            print(f"   Run ID: {best_run_id}")
            print(f"   Run Name: BEST_MODEL_{best_result['name']}")
            print(f"   Artifact Path: model")
            
            # Verify artifact was logged
            print("\n🔍 Verifying artifact was logged...")
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Small delay to ensure write completes
            import time
            time.sleep(2)
            
            artifacts = client.list_artifacts(best_run_id)
            artifact_paths = [a.path for a in artifacts]
            
            if "model" in artifact_paths:
                print("✓ Artifact 'model' confirmed in run")
                # List model contents
                model_artifacts = client.list_artifacts(best_run_id, path="model")
                print("  Model contents:")
                for a in model_artifacts:
                    print(f"    - {a.path}")
            else:
                print("⚠️  WARNING: Artifact 'model' not found immediately after logging")
                print(f"   Available artifacts: {artifact_paths}")
                print("   This might be a sync delay - checking DagsHub...")
            
    except Exception as e:
        print(f"\n❌ ERROR logging best model: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️  Attempting fallback: logging without input_example...")
        
        try:
            with mlflow.start_run(run_name=f"BEST_MODEL_{best_result['name']}_fallback") as run:
                mlflow.log_metric("rmse", best_result['rmse'])
                mlflow.log_metric("r2_score", best_result['r2'])
                mlflow.log_metric("mae", best_result['mae'])
                mlflow.log_param("model_type", best_result['name'])
                mlflow.log_param("selected_as_best", "True")
                
                # Try without input_example
                mlflow.sklearn.log_model(
                    sk_model=best_result['model'],
                    artifact_path="model"
                )
                
                print("✓ Fallback logging succeeded")
                best_run_id = run.info.run_id
                print(f"   Run ID: {best_run_id}")
                
        except Exception as e2:
            print(f"\n❌ FATAL: Fallback also failed: {e2}")
            traceback.print_exc()
            sys.exit(1)
    
    print("\n" + "="*70)
    print("✅ TRAINING PIPELINE COMPLETED!")
    print("="*70)
    print(f"\n🔗 View results: https://dagshub.com/rezahmas/boston-housing-mlflow/experiments")
    print("="*70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)