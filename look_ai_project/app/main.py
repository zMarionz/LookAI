# app/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import io
import os
from PIL import Image
import numpy as np
from openvino.runtime import Core

app = FastAPI(title="Look AI – Virtual Try-On")
app.mount("/web", StaticFiles(directory="web"), name="web")

# Load model
ie = Core()
model_ir_path = "model/cp_vton.xml"
model = ie.read_model(model=model_ir_path)
compiled_model = ie.compile_model(model, "CPU")

def preprocess_image(img_path):
    image = Image.open(img_path).convert("RGB")
    image = image.resize((192, 256))
    image_np = np.asarray(image).transpose(2, 0, 1) / 255.0
    return image_np.astype(np.float32).reshape(1, 3, 256, 192)

def run_inference(user_img, cloth_img):
    user_tensor = preprocess_image(user_img)
    cloth_tensor = preprocess_image(cloth_img)
    input_data = np.concatenate([user_tensor, cloth_tensor], axis=1)
    result = compiled_model([input_data])[compiled_model.output(0)]
    output_image = np.clip(result[0].transpose(1, 2, 0), 0, 1) * 255
    return Image.fromarray(output_image.astype(np.uint8))

@app.post("/tryon/")
async def tryon(user_image: UploadFile = File(...), cloth_image: UploadFile = File(...)):
    os.makedirs("temp", exist_ok=True)
    user_path = f"temp/user_{user_image.filename}"
    cloth_path = f"temp/cloth_{cloth_image.filename}"
    
    with open(user_path, "wb") as f:
        f.write(await user_image.read())
    with open(cloth_path, "wb") as f:
        f.write(await cloth_image.read())

    result_img = run_inference(user_path, cloth_path)
    buf = io.BytesIO()
    result_img.save(buf, format="PNG")
    buf.seek(0)

    os.remove(user_path)
    os.remove(cloth_path)

    return StreamingResponse(buf, media_type="image/png")