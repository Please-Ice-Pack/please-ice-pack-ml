from typing import List
import json
import pandas as pd
import torch
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

app = FastAPI()
# pt_path = '/content/drive/MyDrive/마켓컬리_해커톤/weight/best.pt'
# model =torch.hub.load('./yolov5', 'custom', path=pt_path, source='local')
model =torch.hub.load('./', 'custom', path='./runs/train/pet/weights/best.pt', source='local',)
model.conf = 0.5

@app.post("/files/")
async def create_files(files: List[bytes] = File()):
    result = dict()
    im = Image.open(io.BytesIO(files[0]))
    results = model(im)
    detect_res = results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    for dic in detect_res:
        class_ = dic['class']
        if result.get(class_): result[class_] = result[class_] + 1
        else: result[class_] = 1
    return result


@app.get("/")
async def main():
    content = """
<body>
<form action="/files/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple>
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)
