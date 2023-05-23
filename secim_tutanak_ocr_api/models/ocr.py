from pydantic import BaseModel
from fastapi import UploadFile
from typing import Optional, List

class ImageOCRRequest(BaseModel):
    img_url: Optional[str]
    file: Optional[UploadFile]


class OcrResult(BaseModel):
    table_id : str
    raw_text : list
class ImageOCRResponse(BaseModel):
    file_name : str
    qr_codes : list
    ocr_results : List[OcrResult]