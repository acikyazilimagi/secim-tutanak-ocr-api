
import os
import cv2
import numpy as np
from pathlib import Path

from paddleocr import PaddleOCR, draw_ocr

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class PaddlePaddleOcr:
    def __init__(self) -> None:

        # default , Initializing OCR, OCR will automatically download PP-OCRv3 detector, recognizer and angle classifier. #TODO customization
        self.paddle_ocr = PaddleOCR(use_angle_cls=False,lang='tr',use_gpu=False) #TODO GPU,If it's from a request, it throws an error in recognition, but works fine in docker terminal with GPU

    def ocr(self,img_path):
        
        result = self.paddle_ocr.ocr(img_path)

        return result


    def save_ocr(self,img_path,save_path,result):
        font_path = Path('/code/secim_tutanak_ocr_api/utils/postprocess/simfang.ttf').as_posix()
        save_path = os.path.join(save_path,'ocr_result_'+ img_path.split('/')[-1].split('.')[0] + '.jpg')

        image = cv2.imread(img_path)

        # Extracting boxes, texts and its score from the output list.
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        # Plotting the outputs using PaddleOCR in-built function.
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        
        # Saving the output.
        cv2.imwrite(save_path, im_show)

        #img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)


 