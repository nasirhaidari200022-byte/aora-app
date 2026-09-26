import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
import tflite_runtime.interpreter as tflite
from PIL import Image
import io

app = FastAPI()

interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open("disease_data.json", "r", encoding="utf-8") as f:
    disease_info = json.load(f)

CLASS_NAMES = [
    "Apple___Apple_scab", 
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust", 
    "Apple___healthy"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_index = np.argmax(output_data[0])
    confidence = float(np.max(output_data[0])) * 100
    
    predicted_class = CLASS_NAMES[predicted_index]
    extra_details = disease_info.get(predicted_class, {"info": "اطلاعاتی در فایل JSON یافت نشد."})
    
    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "details": extra_details
    }

@app.get("/")
def read_root():
    return {"status": "Server is running successfully!"}
