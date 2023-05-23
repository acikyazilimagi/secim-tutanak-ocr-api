import requests

url = "http://127.0.0.1:8080/api/ocr/predimg"


file_path = '../_test_data/17013_bulk_999999999_1684082326979111104.jpeg'

    
#read image as byte
with open(file_path, 'rb') as f:
    img_file = f.read()

#send image as byte with file tag
files= {'file': ('custom_deneme.jpg',img_file,'multipart/form-data')}
        


# [POST] Request to api
response_predict = requests.post(url, files=files,timeout=(5,45))

if response_predict.ok:
    response_json = response_predict.json()
    print(response_json)