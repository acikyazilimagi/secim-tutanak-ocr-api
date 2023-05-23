from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from secim_tutanak_ocr_api.api.routes.router import api_router
from secim_tutanak_ocr_api.core.config import (API_PREFIX, APP_NAME, APP_VERSION, IS_DEBUG)
from secim_tutanak_ocr_api.core.event_handlers import (start_app_handler,stop_app_handler)
from secim_tutanak_ocr_api.core.middleware import SimpleASGIMiddleware


def get_app() -> FastAPI:
    fast_app = FastAPI(title=APP_NAME, version=APP_VERSION, debug=IS_DEBUG)
    fast_app.include_router(api_router, prefix=API_PREFIX)

    fast_app.mount(f"{API_PREFIX}/static", StaticFiles(directory="secim_tutanak_ocr_api/_results"), name="static")

    fast_app.add_event_handler("startup", start_app_handler(fast_app))
    fast_app.add_event_handler("shutdown", stop_app_handler(fast_app))

    fast_app.add_middleware(SimpleASGIMiddleware)

    return fast_app


app = get_app()