from pydantic import BaseModel
from fastapi import UploadFile
from typing import Optional, List

class ImageOCRRequest(BaseModel):
    img_url: Optional[str]
    file: Optional[UploadFile]


class ImageOCRResponse(BaseModel):
    pass