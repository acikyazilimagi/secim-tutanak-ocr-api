import requests,os,json
from urllib.parse import urlparse
from io import BytesIO
from pathlib import Path

from PIL import Image
import cv2
import numpy as np

from fastapi import APIRouter, Depends,status,HTTPException
from starlette.requests import Request

from secim_tutanak_ocr_api.core.config import UPLOAD_FOLDER_PATH,RESULT_FOLDER_PATH,SAVE_RESULTS
from secim_tutanak_ocr_api.models.ocr import ImageOCRRequest,ImageOCRResponse

from secim_tutanak_ocr_api.services.qr_detector import dedection_and_decode_qr_code


router = APIRouter()

@router.post("/predimg")
async def predimg(request: Request,request_data: ImageOCRRequest = Depends()) -> ImageOCRResponse : #TODO authentication ? 

    if request_data.file: # Capture Uploaded Image and save
        base_name = request_data.file.filename

        #Save the original image
        requested_img = request_data.file.file.read()
        image_path = f"{UPLOAD_FOLDER_PATH}/{base_name}"

        with open(image_path, "wb") as f:
            f.write(requested_img)

        nparr = np.fromstring(requested_img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    elif request_data.img_url: #  Second option as request - read image from url and save

        url = urlparse(request_data.img_url)
        base_name = os.path.basename(url.path) 
        image_path = f"{UPLOAD_FOLDER_PATH}/{base_name}"

        response = requests.get(request_data.img_url)
        pil_img = Image.open(BytesIO(response.content)).convert("RGB")
        pil_img.save(image_path)
        img = np.array(pil_img)

    else:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail='Acceptable types are Image or Image_URL ')

    
    # create a custom empty parent folder for the uploaded image
    if SAVE_RESULTS: 
        base_name = os.path.basename(image_path)
        custom_folder_path = os.path.join(RESULT_FOLDER_PATH,base_name)
        if not os.path.exists(custom_folder_path):
            os.makedirs(custom_folder_path)



    #Preprocessing Step - 1  (Document Alignment)
    scanned_document_img, saved_path_aligned_img = request.app.state.document_scanner_service.scan_and_align(base_name)
    

    # (QR Detection) from aligned document
    results_decoded = dedection_and_decode_qr_code(image_path=saved_path_aligned_img)
    if results_decoded:
        qr_codes_in_document = [result_decoded['text'] for result_decoded in results_decoded]
    else:
        qr_codes_in_document = ['This document does not contain any QR !']
        #raise HTTPException(status_code = 404, detail = 'This document does not contain any QR !')


    #Preprocessing Step - 2  (Detection)
    table_detector_results, base_name_table_detected_img = request.app.state.table_detector_service.detect(saved_path_aligned_img)


    #save all detected tables separately
    cropped_region_saved_paths = []
    for i,bbox in enumerate(table_detector_results["boxes"]):
        bbox = [round(i, 2) for i in bbox.tolist()]

       # Crop the detected regions
        cropped_img = scanned_document_img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        cropped_region_saved_path = Path(RESULT_FOLDER_PATH,base_name,f'cropped_region_{i}_{base_name}').as_posix()
        cropped_region_saved_paths.append(cropped_region_saved_path)

        cv2.imwrite(cropped_region_saved_path,cropped_img)


    # OCR  Step -3
    ocr_results = []

    save_path = Path(RESULT_FOLDER_PATH,base_name).as_posix()
    for i,cropped_img_path in enumerate(cropped_region_saved_paths):
        print(cropped_img_path)
        results = request.app.state.ocr_service.ocr(cropped_img_path)
        print('results ocr :',results)
        request.app.state.ocr_service.save_ocr(cropped_img_path,save_path,results[0])


        raw_texts = [f'{line[1][0]}\n' for line in results[0]]

        ocr_results.append({
            'table_id' : f'table-{i}',
            'raw_text' : raw_texts
        })


    result_data = {
        'file_name': base_name,
        'qr_codes' : qr_codes_in_document,
        'ocr_results' : ocr_results
    }

    return ImageOCRResponse(**result_data)