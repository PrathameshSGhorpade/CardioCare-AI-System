# CardioCare AI System

CardioCare AI is a Streamlit demonstration that classifies cardiovascular-disease labels from patient measurements and shows SHAP explanations with lifestyle and treatment guidance.

## Run the app

From the project folder, install the listed packages and start Streamlit:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501). The saved XGBoost model and scaler are included for inference.

## Train the models

The patient dataset is not included in this repository. The upstream Kaggle listing currently shows its license as unknown. If you have permission to use a copy, place the processed file at `dataset/cardio_data_processed.csv`, then run:

```powershell
python train.py
```

Training saves the fitted scaler and places the trained XGBoost model where the app loads it.

## Attribution and data

This repository is a clean-history copy/adaptation of the project by Ojobo Agbo at [ojobo-tech/Cardiovascular-Disease-Prediction-and-Treatment-Recommendation-System](https://github.com/ojobo-tech/Cardiovascular-Disease-Prediction-and-Treatment-Recommendation-System). See [ATTRIBUTION.md](ATTRIBUTION.md) for provenance and licensing notes.

## Important use note

The output estimates the likelihood of the dataset's CVD label. It is not a diagnosis or a substitute for care from a qualified healthcare professional.
