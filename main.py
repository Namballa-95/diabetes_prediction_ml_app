from fastapi import FastAPI
from pydantic import BaseModel
from predictor import prediction_function

app = FastAPI(title="Diabetes Prediction App")

#define input schema
class PredictionInput (BaseModel):
     Pregnancies: int
     Glucose: float
     BloodPressure: float
     SkinThickness: float
     Insulin: float
     BMI: float
     DiabetesPedigreeFunction: float
     Age: int


#ml prediction endpoint
@app.post("/predict")
def predict_diabetes (input_data:PredictionInput):
     prediction = prediction_function(input_data.model_dump())
     return {"prediction": int(prediction)}