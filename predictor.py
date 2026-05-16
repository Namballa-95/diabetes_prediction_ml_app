import os
import logging
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from joblib import load

# load .env content to env vars
load_dotenv()

project_root_raw = os.getenv(r"C:\Users\bhara\OneDrive\Documents\SERVE_ML_MODEL_AS_API\.env")
if project_root_raw:
    project_root_raw = project_root_raw.strip().strip('"').strip("'")
    project_root_raw = project_root_raw.replace("\\", "/")
    PROJECT_ROOT = Path(project_root_raw)
else:
    PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / os.getenv("MODEL_DIR", "model_dir") / os.getenv("MODEL_NAME", "diabetes_pipeline_trained.joblib")
LOG_PATH = PROJECT_ROOT / os.getenv("LOG_DIR", "logs") / os.getenv("LOG_NAME", "app.log")

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
     handlers=[
          logging.FileHandler(LOG_PATH),
          logging.StreamHandler()
     ]
)

#loading the trained model
logging.info(f"Loading model from {MODEL_PATH}")
try:
    model = load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")

#function to make predictions using the loaded model
def prediction_function(input_data: dict):
    df = pd.DataFrame ([input_data])
    prediction = model.predict(df) [0]
    return prediction


# # Example usage
# if __name__ == "__main__":
#     sample_input = {
#         'Pregnancies': 2,
#         'Glucose': 120,
#         'BloodPressure': 70,
#         'SkinThickness': 20,
#         'Insulin': 85,
#         'BMI': 25.0,
#         'DiabetesPedigreeFunction': 0.5,
#         'Age': 30
#     }
#     result = prediction_function(sample_input)
#     logging.info(f"Prediction result: {result}")