import shutil
from pathlib import Path
from utils.preprocessing import load_and_preprocess_data
from models.training_models import ModelTrainer

def main():
    project_dir = Path(__file__).resolve().parent
    csv_path = project_dir / "dataset" / "cardio_data_processed.csv"
    scaler_path = project_dir / "scaler.pkl"
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data(
        csv_path, scaler_path=scaler_path
    )

    trainer = ModelTrainer(X_train, X_test, y_train, y_test)

    trainer.train_all_models()
    shutil.copy2(project_dir / "saved_models" / "xgboost.pkl", project_dir / "xgboost.pkl")

    print("All models trained and saved!")

if __name__ == "__main__":
    main()
