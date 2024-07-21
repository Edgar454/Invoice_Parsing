import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel
from utils import process_image
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Summary, Gauge, start_http_server
import time
import psutil

# Load the model and processor globally
model = VisionEncoderDecoderModel.from_pretrained("Edgar404/donut-shivi-cheques_KD_320")
processor = DonutProcessor.from_pretrained("Edgar404/donut-shivi-cheques_KD_320")
model.eval()

class InferenceResults(BaseModel):
    prediction: dict

# Define metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total number of requests')
SUCCESSFUL_REQUEST_COUNT = Counter('app_requests_successful_total', 'Total number of successful requests')
UNSUCCESSFUL_REQUEST_COUNT = Counter('app_requests_unsuccessful_total', 'Total number of unsuccessful requests')
INFERENCE_TIME = Summary('inference_processing_seconds', 'Time spent processing inference')
CPU_USAGE = Gauge('cpu_usage_percentage', 'Percentage of CPU used')

# Start Prometheus client server on port 8001
start_http_server(8001)

def get_cpu_usage():
    return psutil.cpu_percent(interval=None)

# Building the server
app = FastAPI(title='Invoice parser')

# Instrument the FastAPI app
Instrumentator().instrument(app).expose(app)

@app.middleware("http")
async def prometheus_middleware(request, call_next):
    REQUEST_COUNT.inc()
    start_time = time.time()
    try:
        response = await call_next(request)
        if response.status_code == 200:
            SUCCESSFUL_REQUEST_COUNT.inc()
        else:
            UNSUCCESSFUL_REQUEST_COUNT.inc()
        return response
    except Exception as e:
        UNSUCCESSFUL_REQUEST_COUNT.inc()
        raise e
    finally:
        elapsed_time = time.time() - start_time
        INFERENCE_TIME.observe(elapsed_time)
        CPU_USAGE.set(get_cpu_usage())

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
    start_time = time.time()
    try:
        prediction = process_image(image, model, processor, torch.float32)
        SUCCESSFUL_REQUEST_COUNT.inc()
    except Exception as e:
        UNSUCCESSFUL_REQUEST_COUNT.inc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        elapsed_time = time.time() - start_time
        INFERENCE_TIME.observe(elapsed_time)
        CPU_USAGE.set(get_cpu_usage())
    
    return InferenceResults(prediction=prediction)
    
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)
