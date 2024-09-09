from PIL import Image
import numpy as np
import requests
import io
import os
import json

def load_class_indices():
    # Load the class indices from the JSON file
    working_dir = os.path.dirname(os.path.abspath(__file__))
    return json.load(open(f"{working_dir}/class_indices(1).json"))

def load_and_preprocess_image(image, target_size=(224, 224)):
    # Ensure the input is a PIL Image object
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    
    # Resize the image to the target size
    img = image.resize(target_size)
    
    # Convert the image to a NumPy array and preprocess it for the model
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    
    return img_array

def predict_disease(image_file, fastapi_url):
    # Open the image file with PIL
    image = Image.open(image_file)
    
    # Convert the image to a byte stream
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')  # Save image as JPEG
    img_byte_arr.seek(0)  # Move to the beginning of the byte stream
    
    # Prepare the file payload
    files = {
        "file": ("image.jpg", img_byte_arr, "image/jpeg")
    }

    # Send the POST request to FastAPI
    try:
        response = requests.post(fastapi_url, files=files)
        if response.status_code == 200:
            prediction = response.json().get('prediction')
            return prediction
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    # FastAPI endpoint URL
    fastapi_url = "http://ec2-65-2-137-251.ap-south-1.compute.amazonaws.com:8000/predict"
    
    # Load class indices if needed for local use
    class_indices = load_class_indices()
    
    # Example image file
    image_file_path = "path/to/your/image.jpg"
    image_file = open(image_file_path, "rb")
    
    # Predict the disease
    result = predict_disease(image_file, fastapi_url)
    print(f"Prediction: {result}")
