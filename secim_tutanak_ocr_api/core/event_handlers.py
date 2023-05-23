from typing import Callable

from fastapi import FastAPI
from loguru import logger

from secim_tutanak_ocr_api.services.table_detector import TableDetector
from secim_tutanak_ocr_api.services.paddlepaddle_ocr import PaddlePaddleOcr
from secim_tutanak_ocr_api.services.document_scanner import DocumentScanner



def start_app_handler(app: FastAPI) -> Callable:
    def startup() -> None:
        logger.info("Running app start handler.")
        
        # Load Models Only Once in Startup. #

        #Document Scanner
        checkpoint_path = "secim_tutanak_ocr_api/model_weights/document_scanner_weights/model_mbv3_iou_mix_2C049.pth"
        app.state.document_scanner_service = DocumentScanner(checkpoint_path=checkpoint_path,
                                                    num_classes=2,
                                                    model_name='mbv3',
                                                    device='cpu' #TODO: GPU-CUDA TEST Require 'cuda'
                                                )
        #Table Detector
        app.state.table_detector_service = TableDetector()

        #PaddplePaddleOcr
        app.state.ocr_service = PaddlePaddleOcr()

        logger.info("All models loaded in startup.")


    return startup

def stop_app_handler(app: FastAPI) -> Callable:
    def shutdown() -> None:
        logger.info("Running app shutdown handler.")
        #_shutdown_model(app)

    return shutdown