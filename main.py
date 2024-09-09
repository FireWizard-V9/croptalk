import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import io

app = FastAPI()

# Define the path to the model and class indices
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "model_name (1).h5")
class_indices_path = os.path.join(working_dir, "class_indices(1).json")

# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Load class indices
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# Function to preprocess the image
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype('float32') / 255.
    return image_array

# Endpoint for predicting crop disease
@app.post("/predict")
async def predict_crop_disease(file: UploadFile = File(...)):
    try:
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        preprocessed_img = preprocess_image(image)

        # Make prediction
        predictions = model.predict(preprocessed_img)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown class")
        confidence = np.max(predictions)

        return JSONResponse(content={
            "prediction": predicted_class_name,
            "confidence": float(confidence)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Root endpoint
@app.get("/")
def index():
    return {"message": "FastAPI server is up and running!"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
