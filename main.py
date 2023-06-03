from typing import Union
from fastapi import FastAPI, UploadFile
from model import (predict)
import torch

app = FastAPI()


@app.get('/')
def read_root():
    return {"msg":"Hello World!"}



@app.post('/predict')
async def predict_image(file: UploadFile):
    predicted = predict(file.file)
    return {"filename": file.filename, "object":predicted}
