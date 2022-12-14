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
# model = torch.hub.load('./', 'custom', path='./weight/best.pt', source='local',)
# box_meta = pd.read_csv('./meta_data/box_meta.csv')
# ice_meta = pd.read_csv('./meta_data/ice_meta.csv')
# product_meta = pd.read_csv('./meta_data/product_meta.csv', encoding='cp949')

# docker
model = torch.hub.load('./app', 'custom', path='./app/weight/best.pt', source='local',)
box_meta = pd.read_csv('./app/meta_data/box_meta.csv')
ice_meta = pd.read_csv('./app/meta_data/ice_meta.csv')
product_meta = pd.read_csv('./app/meta_data/product_meta.csv', encoding='cp949')

model.conf = 0.5

s3 = boto3.resource('s3', region_name='ap-northeast-2')
bucket = s3.Bucket('bucket-packing-pip-an2')

class Products(BaseModel):
    productId: int
    amount: int

class Item(BaseModel):
    orderId: int
    isPurpleBox: bool
    products: List[Products]
    imageUrl: str
        
def box_select(product, isPurpleBox):
    fro_size, ref_size = 0, 0
    # 냉장, 냉동 제품 별 부피 계산
    for i in product:
#         box_result, ice_result = dict(), list()
        box_result, ice_result = list(), list()
        temp = product_meta[product_meta['code'] == int(i)]
        temp_size = int(temp['width'] * temp['height'] * temp['length'] * product[i])
        if int(temp['cold_type']) == 2:
            fro_size += temp_size
        else:
            ref_size += temp_size
    fro_size = int(fro_size * 2)
    ref_size = int(ref_size * 2)
    
    print('냉동 사이즈:',fro_size, '냉장 사이즈:',ref_size)
    # 박스 및 냉매제 선택 
    for total_size, name in zip([ref_size, fro_size],['refrigerated','frozen']):
        if total_size == 0:
#             box_result[name] = None
            continue
        for i in range(len(box_meta)):
            temp = box_meta.loc[i]
            temp_size = temp['width'] * temp['height'] * temp['length']
            if total_size < temp_size:
                temp_dict = temp[['box_type','box_size']].to_dict()
                temp_dict['box_size'] = str(temp_dict['box_size'])
                if name == 'refrigerated':
                    temp_dict['box_flag'] = 'REF'
                else:
                    temp_dict['box_flag'] = 'FRE'
                temp_ice = dict()
                temp_ice['refrigerant_type'] = 'ICE_PACK' if name == 'refrigerated' else 'DRY_ICE'
                temp_ice['refrigerant_size'] = '2' if name == 'refrigerated' and temp_size > 6000 else '1'
                temp_ice['refrigerant_amount'] = int(temp_size) // 3000
#                 box_result[name] = temp_dict
                box_result.append(temp_dict)
                ice_result.append(temp_ice)
                break
    if isPurpleBox:
        box_result = [{'box_type':'PURPLE','box_size':'1','box_flag':'PUR'}]
#         box_meta[box_meta['box_type'] == 'PURPLE'][['box_type','box_size']].iloc[0].to_dict()
        
    return box_result, ice_result

def read_image_from_s3(filename):
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
  "isPurpleBox": true,
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
    result['orderMatched'] = True
    
    result['order_results'] = []
    result['detect_results'] = []
    order_list = data['products']
    isPurpleBox = data['isPurpleBox']
    detect_dict = dict()
    
    # 이미지 로드 및 제품 인식 
    url = data['imageUrl'].split('/')[-1] # 이미지
    im = read_image_from_s3(url)
    det_raw_results = model(im)
    detect_res = det_raw_results.pandas().xyxy[0].to_json(orient="records")
    detect_res = json.loads(detect_res)
    
    # 인삭한 제품 형식 변경
    for dic in detect_res:
        class_ = dic['class']
        if detect_dict.get(class_): detect_dict[class_] = detect_dict[class_] + 1
        else: detect_dict[class_] = 1
    
    # 인식 제품과 주문 제품 매치 여부 확인
    order_dict = dict()
    for order_detail in order_list:
        productId = order_detail['productId']
        amount = order_detail['amount']
        order_dict[productId] = amount
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
            if detect_dict[productId] != amount:
                temp_order['isMatched'] = False
                result['orderMatched'] = False
        # detect 결과와 다른 경우
        else: 
            temp_order['isMatched'] = False
            result['orderMatched'] = False
            
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
    
    print('주문내역:',order_dict)
    recommendedPackingOption, refrigerants = box_select(order_dict, isPurpleBox)
    result['recommendedPackingOption'] = recommendedPackingOption
    result['refrigerants'] = refrigerants

    return result
