import uvicorn
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import io
import torch
import numpy as np

from transformers import DonutProcessor, VisionEncoderDecoderModel
from utils import process_image

model = VisionEncoderDecoderModel.from_pretrained("Edgar404/donut-shivi-cheques_KD_320" ,torch_dtype = torch.float16 )
processor = DonutProcessor.from_pretrained("Edgar404/donut-shivi-cheques_KD_320")
model.eval()

class InferenceResults(BaseModel):
    prediction:dict

# building the server
app = FastAPI(title = 'Invoice parser')

@app.get('/')
def read_root():
    return {"hello":"world"}
    
@app.on_event('startup')
async def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("Edgar404/donut-shivi-cheques_KD_320" ,torch_dtype = torch.float16 )
    processor = DonutProcessor.from_pretrained("Edgar404/donut-shivi-cheques_KD_320")
    model.eval()
    

@app.post('/predict')
async def get_prediction(file: UploadFile = File(...)):
    
    # 1. VALIDATE INPUT FILE
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file provided.")
        
    # 2. TRANSFORM RAW IMAGE INTO PIL image
    
    # Read image as a stream of bytes
    image_stream = io.BytesIO(file.file.read())
    
    # Start the stream from the beginning (position zero)
    image_stream.seek(0)
    
    # Write the stream of bytes into a numpy array
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    
    #Decode the numpy array as a PIL  image
    image = Image.fromarray(file_bytes)
    image = image.convert('RGB')
    
    # make the prediction
    prediction = process_image(image ,model , processor , torch.float16)
    
    return InferenceResults(prediction = prediction)
    
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000)