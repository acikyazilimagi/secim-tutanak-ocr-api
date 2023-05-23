from fastapi import APIRouter

from secim_tutanak_ocr_api.api.routes import ocr

api_router = APIRouter()
api_router.include_router(ocr.router, tags=["ocr"], prefix="/ocr")