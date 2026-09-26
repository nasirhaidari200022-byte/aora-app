import json
import numpy as np
from fastapi import FastAPI, UploadFile, File
import tflite_runtime.interpreter as tflite
from PIL import Image
import io

app = FastAPI()

# بارگذاری مدل لایت
interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# بارگذاری اطلاعات فارسی بیماری‌ها
with open("disease_data.json", "r", encoding="utf-8") as f:
    disease_info = json.load(f)

# ترتیب دقیق کلاس‌ها بر اساس خروجی کولب شما
CLASS_NAMES = [
    "Apple___Apple_scab", 
    "Apple___Black_rot", 
    "Apple___Cedar_apple_rust", 
    "Apple___healthy"
]

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # خواندن و باز کردن عکس ارسالی کاربر
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    
    # تبدیل به آرایه عددی (دقیقاً مثل کد تستی که جواب درست داد)
    img_array = np.array(image, dtype=np.float32)
    single_input = np.expand_dims(img_array, axis=0)
    
    # مدیریت ابعاد ورودی به صورت پویا (داینامیک) برای جلوگیری از کرش سرور
    try:
        interpreter.resize_tensor_input(input_details[0]['index'], single_input.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], single_input)
    except Exception:
        # اگر مدل اصرار بر بچ‌سایز ۸ داشت، تصویر را تکرار می‌کند
        batch_input = np.repeat(single_input, 8, axis=0)
        interpreter.resize_tensor_input(input_details[0]['index'], batch_input.shape)
        interpreter.allocate_tensors()
        interpreter.set_tensor(input_details[0]['index'], batch_input)
    
    # اجرای مدل لایت
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    # استخراج کلاس و درصد اطمینان برای نمونه اول
    predicted_index = np.argmax(output_data[0] if len(output_data.shape) > 1 else output_data)
    confidence = float(np.max(output_data[0] if len(output_data.shape) > 1 else output_data)) * 100
    
    predicted_class = CLASS_NAMES[predicted_index]
    extra_details = disease_info.get(predicted_class, {"info": "اطلاعاتی در فایل JSON یافت نشد."})
    
    return {
        "class": predicted_class,
        "confidence": round(confidence, 2),
        "details": extra_details
    }

@app.get("/")
def read_root():
    return {"status": "Server is running successfully with MobileNetV2!"}
