from typing import Callable

from fastapi import FastAPI
from loguru import logger



def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        
        #OCR Model load here #TODO
        #app.state.ocr_service = PaddlePaddleOcr()

    return startup

def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        #_shutdown_model(app)

    return shutdown