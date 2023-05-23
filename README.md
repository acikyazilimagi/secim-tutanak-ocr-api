# To start API without DOCKER
* uvicorn secim_tutanak_ocr_api.main:app --reload --host 0.0.0.0 --port 8080

# To start API with Dockerfile 
  1. Build 
      *   ``` docker build -t secim-tutanak-ocr-api . ```
  2. RUN
      *   ``` docker run -p 8080:8080 -it --rm --gpus all -v secim-tutanak-ocr-vol:/code/ -t secim-tutanak-ocr-api  ```
    
# To start API with Docker-Compose in Production
  1. Build & Run
    * ```docker-compose -f docker-compose.dev.yml up --build ```