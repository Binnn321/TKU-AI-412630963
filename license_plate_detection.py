import torch
from PIL import Image
import cv2
import pytesseract
import pandas as bd

# 載入預訓練的 YOLOv5 模型
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# 載入圖片
image_path = '/Users/liweibin/車牌辨識/11.jpg'
print('圖片路徑：', image_path) 
image = Image.open(image_path)

# 將圖片轉換為模型所需的格式
image = image.convert('RGB')

# 進行目標檢測
results = model(image)

# 取得檢測結果
detections = results.pandas().xyxy[0]

# 過濾出車牌的檢測結果
license_plate_detections = detections[detections['name'] == 'license_plate']

# 輸出車牌號碼
for _, detection in license_plate_detections.iterrows():
    license_plate_number = detection['name']
    confidense = detection['confidence']
    bbox = detection[['xmin', 'ymin', 'xmax', 'ymax']].tolist()
    print('車牌號碼：', license_plate_number)
    print('信心分數：', confidense)
    print('邊界匡座標：', bbox)


