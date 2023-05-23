from starlette.config import Config
from starlette.datastructures import Secret

APP_VERSION = "0.0.1"
APP_NAME = 'Secim Tutanak OCR API'
API_PREFIX = "/api"

config = Config(".env")

API_KEY: Secret = config("API_KEY", cast=Secret)
IS_DEBUG: bool = config("IS_DEBUG", cast=bool, default=False)


SAVE_RESULTS = True
UPLOAD_FOLDER_PATH = "secim_tutanak_ocr_api/_uploads"

RESULT_FOLDER_PATH = "secim_tutanak_ocr_api/_results"
OCR_TEST_DATA_FOLDER_PATH = "secim_tutanak_ocr_api/_data"

ROOT_STATIC_URL = "/api/static"