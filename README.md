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


## Important use note

The output estimates the likelihood of the dataset's CVD label. It is not a diagnosis or a substitute for care from a qualified healthcare professional.

## Run with Docker

Build the image from the project root:

```bash
docker build -t cvd-care-system .
```

Start the app and open [http://localhost:8501](http://localhost:8501):

```bash
docker run --rm -p 8501:8501 cvd-care-system
```

Or use Docker Compose:

```bash
docker compose up --build
```

Set `APP_PORT` to change the host port, for example `APP_PORT=8502 docker compose up --build`.

The container includes the Streamlit app, its Python modules, and the XGBoost model and scaler it uses. The training dataset and other model artifacts are excluded from the image.
