import os
from pathlib import Path

from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw


from secim_tutanak_ocr_api.core.config import UPLOAD_FOLDER_PATH,RESULT_FOLDER_PATH

class TableDetector:
    def __init__(self) -> None:
        self.processor = DetrImageProcessor.from_pretrained("TahaDouaji/detr-doc-table-detection")
        self.model = DetrForObjectDetection.from_pretrained("TahaDouaji/detr-doc-table-detection")
                

    def detect(self,file_path):

        image = Image.open(file_path)

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])

        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0] # with score > 0.9

        # create rectangle on image
        img1 = ImageDraw.Draw(image)  

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                box = [round(i, 2) for i in box.tolist()]
                print(
                        f"Detected {self.model.config.id2label[label.item()]} with confidence "
                        f"{round(score.item(), 3)} at location {box}"
                )


                img1.rectangle(box, outline ="blue", width=5)


        base_name = os.path.basename(file_path).replace('prep_scan_','')
        base_name_table_detected_img = f'prep_table_detect_{base_name}'
        file_path_save =  Path(RESULT_FOLDER_PATH,base_name,base_name_table_detected_img).as_posix()
        image.save(file_path_save)

        return results, file_path_save

