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
model = torch.hub.load('./app', 'custom', path='./app/weight/best.pt', source='local',)
model.conf = 0.5

box_meta = pd.read_csv('./app/meta_data/box_meta.csv')
product_meta = pd.read_csv('./app/meta_data/product_meta.csv', encoding='cp949')

def box_select(product):

    for i in product:
        fro_size, ref_size = 0, 0
        result = []
        temp = product_meta[product_meta['code'] == int(i)]
        if int(temp['cold_type']) == 2:
            fro_size += temp['width'] * temp['height'] * temp['length'] * product[i]
        else:
            ref_size += temp['width'] * temp['height'] * temp['length'] * product[i]
    fro_size, ref_size = int(fro_size * 1.5), int(ref_size * 1.5)

    for total_size in [ref_size, fro_size]:
        if total_size == 0:
            result.append(None)
            continue
        for i in range(len(box_meta)):
            temp = box_meta.loc[i]
            temp_size = temp['width'] * temp['height'] * temp['length']
            if total_size < temp_size:
                print(total_size, temp_size)
                result.append(temp.to_dict())
                break

    return result

# def ref_select(product):
    

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
    ref_box, fro_box = box_select(result)


    return ref_box, fro_box, result


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
