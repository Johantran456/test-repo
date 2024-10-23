from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

# Load the pre-trained model
model = joblib.load('regression.joblib')

app = FastAPI()

class HouseFeatures(BaseModel):
    size: float
    nb_rooms: int
    garden: bool

@app.post("/predict")
def predict(features: HouseFeatures):
    try:
        # Convert the input data to the format expected by the model
        input_data = np.array([[features.size, features.nb_rooms, features.garden]])
        # Make a prediction
        prediction = model.predict(input_data)
        # Return the prediction as a JSON response
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Run the FastAPI app with the following command:
# uvicorn mini_project:app --reload
# The app will be available at http://