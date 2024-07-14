import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from utils import process_image
from prometheus_fastapi_instrumentator import Instrumentator


# Load the model and processor globally
model = VisionEncoderDecoderModel.from_pretrained("Edgar404/donut-shivi-cheques_KD_320")
processor = DonutProcessor.from_pretrained("Edgar404/donut-shivi-cheques_KD_320")
model.eval()

class InferenceResults(BaseModel):
    prediction: dict

# Building the server
app = FastAPI(title='Invoice parser')

# Instrument the FastAPI app
Instrumentator().instrument(app).expose(app)

@app.get('/')
def read_root():
    return {"hello": "world"}

@app.post('/predict')
async def get_prediction(file: UploadFile = File(...)):
    # 1. VALIDATE INPUT FILE
    filename = file.filename
    file_extension = filename.split(".")[-1].lower() in ("jpg", "jpeg", "png")
    if not file_extension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
        
    # 2. TRANSFORM RAW IMAGE INTO PIL image
    image = Image.open(file.file)
    image = image.convert('RGB')
    
    # Make the prediction
    prediction = process_image(image, model, processor, torch.float32)
    
    return InferenceResults(prediction=prediction)
    
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
