from typing import List
import json
import pandas as pd
import torch
import boto3
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder # pPydantic 모델 수정
from PIL import Image
from pydantic import BaseModel

app = FastAPI()

# local test
model = torch.hub.load('./', 'custom', path='./weight/best.pt', source='local',)
box_meta = pd.read_csv('./meta_data/box_meta.csv')
ice_meta = pd.read_csv('./meta_data/ice_meta.csv')
product_meta = pd.read_csv('./meta_data/product_meta.csv', encoding='cp949')

# docker
# model = torch.hub.load('./app', 'custom', path='./app/weight/best.pt', source='local',)
# box_meta = pd.read_csv('./app/meta_data/box_meta.csv')
# ice_meta = pd.read_csv('./app/meta_data/ice_meta.csv')
# product_meta = pd.read_csv('./app/meta_data/product_meta.csv', encoding='cp949')

model.conf = 0.5

s3 = boto3.resource('s3', region_name='ap-northeast-2')
bucket = s3.Bucket('bucket-packing-pip-an2')

class Products(BaseModel):
    productId: int
    amount: int

class Item(BaseModel):
    orderId: int
    products: List[Products]
    imageUrl: str
        
def box_select(product):
    
    # 냉장, 냉동 제품 별 부피 계산
    for i in product:
        fro_size, ref_size = 0, 0
        box_result, ice_result = dict(), dict()
        temp = product_meta[product_meta['code'] == int(i)]
        if int(temp['cold_type']) == 2:
            fro_size += temp['width'] * temp['height'] * temp['length'] * product[i]
        else:
            ref_size += temp['width'] * temp['height'] * temp['length'] * product[i]
    fro_size, ref_size = int(fro_size * 2), int(ref_size * 2)
    
    # 박스 및 냉매제 선택 
    for total_size, name in zip([ref_size, fro_size],['refrigerated','frozen']):
        if total_size == 0:
            box_result[name] = None
            continue
        for i in range(len(box_meta)):
            temp = box_meta.loc[i]
            temp_size = temp['width'] * temp['height'] * temp['length']
            if total_size < temp_size:
                temp_dict = temp[['box_type','box_size']].to_dict()
                temp_dict['refrigerants_id'] = 'Ice_pack' if name == 'refrigerated' else 'Dry_ice'
                temp_dict['refrigerants_size'] = 2 if name == 'refrigerated' and temp_size > 6000 else 1
                temp_dict['refrigerants_amount'] = int(temp_size) // 3000
                box_result[name] = temp_dict
                break

    return box_result

def read_image_from_s3(filename):
    # bucket = s3.Bucket(bucket_name) # bucket_name 필요
    object = bucket.Object(filename)
    response = object.get()
    file_stream = response['Body']
    img = Image.open(file_stream)
    return img

@app.post("/files/")
async def create_files(item: Item):
    '''
    swagger test request body
{
  "orderId": 0,
  "products": [
    {
      "productId": 4,
      "amount": 3
    },
{
      "productId": 9,
      "amount": 1
    },
{
      "productId": 13,
      "amount": 2
    }
  ],
  "imageUrl": "https://bucket-packing-pip-an2.s3.ap-northeast-2.amazonaws.com/10_4.jpg"
}
    '''
    result = dict()
    data = jsonable_encoder(item)
    result['orderId'] = data['orderId']
    result['order_results'] = []
    result['detect_results'] = []
    order_list = data['products']
    detect_dict = dict()
    
#     im = Image.open(io.BytesIO(files[0]))

    # 이미지 로드 및 제품 인식 - s3 연동 후 진행 예정, 
    url = data['imageUrl'].split('/')[-1] # 이미지
    im = read_image_from_s3(url)
    # im = Image.open(r"./sample/9_16.jpg") 
    # im = Image.open(r"./app/sample/9_16.jpg") 
    
    det_raw_results = model(im)
    detect_res = det_raw_results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    
    # 인삭한 제품 형식 변경
    for dic in detect_res:
        class_ = dic['class']
        if detect_dict.get(class_): detect_dict[class_] = detect_dict[class_] + 1
        else: detect_dict[class_] = 1
    
    # 인식 제품과 주문 제품 매치 여부 확인
    for order_detail in order_list:
        productId = order_detail['productId']
        amount = order_detail['amount']
        cold_type = int(product_meta[product_meta['code'] == productId]['cold_type'])
        productName = str(product_meta[product_meta['code'] == productId]['name'].iloc[0])
        temp_order = {
                "productId": productId,
                "productName": productName,
                "amount": amount,
                'cold_type': cold_type,
                "isMatched": True
                }
        
        # detect 결과가 order에 있을 경우
        if detect_dict.get(productId):    
            temp_order['isMatched'] = False if detect_dict[productId] != amount else True
        # detect 결과와 다른 경우
        else: 
            temp_order['isMatched'] = False
            
        result['order_results'].append(temp_order)
    
    for detect_detail in detect_dict:
        temp_detect = {
                "productId": detect_detail,
                "productName": str(product_meta[product_meta['code'] == detect_detail]['name'].iloc[0]),
                "amount": detect_dict[detect_detail],
                'cold_type': int(product_meta[product_meta['code'] == detect_detail]['cold_type']),
        }
        result['detect_results'].append(temp_detect)
    
    # 제품 번호별 정렬
    result['order_results'] = sorted(result['order_results'], key = lambda x: (x['isMatched'],x['productId']))
    result['detect_results'] = sorted(result['detect_results'], key = lambda x: x['productId'])

    recommendedPackingOption = box_select(detect_dict)
    result['recommendedPackingOption'] = recommendedPackingOption

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
