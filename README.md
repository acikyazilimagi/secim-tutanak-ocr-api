# Seçim Tutanak OCR API
An endpoint was developed with Python FastAPI for scanning election minutes, detecting and reading QR, detecting table fields with vote numbers and reading values with OCR.

# Running

## To start API without DOCKER
* uvicorn secim_tutanak_ocr_api.main:app --reload --host 0.0.0.0 --port 8080

## To start API with Dockerfile 
  1. Build 
      *   ``` docker build -t secim-tutanak-ocr-api . ```
  2. RUN
      *   ``` docker run -p 8080:8080 -it --rm --gpus all -v secim-tutanak-ocr-vol:/code/ -t secim-tutanak-ocr-api  ```
    
## To start API with Docker-Compose in Production
  1. Build & Run
    * ```docker-compose -f docker-compose.dev.yml up --build ```

# Using
OCR Endpoint POST:
```
http://127.0.0.1:8080/api/ocr/predimg"
```

Usage Example Python:
```
file_path = '17013_bulk_999999999_1684082326979111104.jpeg'

#read image as byte
with open(file_path, 'rb') as f:
    img_file = f.read()


#send image as byte with file tag
files = {'file': ('custom_test_img.jpg',img_file,'multipart/form-data')}


# [POST] Request to api
response_predict = requests.post(url, files=files)
```


# Libraries Used
## Document Scanner (PreProcessing)
### Model files
- [Model - MobileNetV3-Large backend](https://www.dropbox.com/s/4znmfi5ew1u5z9y/model_mbv3_iou_mix_2C049.pth?dl=1)

Reference: https://github.com/spmallick/learnopencv/tree/master/Document-Scanner-Custom-Semantic-Segmentation-using-PyTorch-DeepLabV3


## QR Dedection and Decode

Reference: https://github.com/NaturalHistoryMuseum/pyzbar/


## Table Dedection

Reference: 
Demo: https://huggingface.co/TahaDouaji/detr-doc-table-detection

## OCR
Referance: https://github.com/PaddlePaddle/PaddleOCR
