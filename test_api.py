import requests
#fastapi endpoint url
API_URL = "http://127.0.0.1:8000/predict"

payload = {
     "Pregnancies": 2,
     "Glucose": 140,
     "BloodPressure": 130, 
     "SkinThickness": 30,
     "Insulin": 80,
     "BMI": 35,
     "DiabetesPedigreeFunction": 0.355,
     "Age": 28
}

# send POST request
response = requests.post(API_URL, json=payload)
response_dictionary = response.json()
print("Response dictionary:", response_dictionary)

# print response
prediction = response_dictionary["prediction"]

print("Prediction:", prediction)
print("Status Code:", response.status_code)
print("Response JSON: \n ", response.json())
print("Prediction is : ",prediction)