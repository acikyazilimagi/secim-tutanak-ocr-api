import os
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as torchvision_T     
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

from secim_tutanak_ocr_api.core.config import UPLOAD_FOLDER_PATH,RESULT_FOLDER_PATH


class DocumentScanner:

    def __init__(self,checkpoint_path,num_classes=2,model_name="mbv3",device='cpu') -> None:
        self.trained_model = self._load_model(num_classes=num_classes, model_name=model_name, checkpoint_path=checkpoint_path, device=device)


    
    def _load_model(self,num_classes=1, model_name="mbv3", checkpoint_path=None, device=None):

        if model_name == "mbv3":
            model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)

        else:
            model = deeplabv3_resnet50(num_classes=num_classes)

        model.to(device)
        checkpoints = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoints, strict=False)
        model.eval()

        #_ = model(torch.randn((2, 3, 384, 384)))

        return model

    def order_points(self,pts):
        rect = np.zeros((4, 2), dtype="float32")
        pts = np.array(pts)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect.astype("int").tolist()


    def find_dest(self,pts):
        (tl, tr, br, bl) = pts
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
        return self.order_points(destination_corners)


    def image_preproces_transforms(self,mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109)):
        common_transforms = torchvision_T.Compose(
            [torchvision_T.ToTensor(), torchvision_T.Normalize(mean, std),]
        )

        return common_transforms



    def scan_and_align(self,image_filename): #TODO ? could be image:array instead of image:path 

        #https://github.com/spmallick/learnopencv/blob/master/Document-Scanner-Custom-Semantic-Segmentation-using-PyTorch-DeepLabV3/Document_extraction.ipynb


        preprocess_transforms = self.image_preproces_transforms()

        file_path = Path(UPLOAD_FOLDER_PATH,image_filename)
        print("file_path", file_path.as_posix())
        image = cv2.imread(file_path.as_posix(), cv2.IMREAD_COLOR)[:, :, ::-1]
        
        image_size=384
        BUFFER=10

        #document = extract(image_true=image, trained_model=trained_model)

        # EXTRACT DOCUMENT
        IMAGE_SIZE = image_size
        half = IMAGE_SIZE // 2

        imH, imW, C = image.shape

        image_model = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)

        scale_x = imW / IMAGE_SIZE
        scale_y = imH / IMAGE_SIZE

        image_model = preprocess_transforms(image_model)
        image_model = torch.unsqueeze(image_model, dim=0)

        with torch.no_grad():
            out = self.trained_model(image_model)["out"].cpu()

        del image_model
        gc.collect()

        out = torch.argmax(out, dim=1, keepdims=True).permute(0, 2, 3, 1)[0].numpy().squeeze().astype(np.int32)
        r_H, r_W = out.shape

        _out_extended = np.zeros((IMAGE_SIZE + r_H, IMAGE_SIZE + r_W), dtype=out.dtype)
        _out_extended[half : half + IMAGE_SIZE, half : half + IMAGE_SIZE] = out * 255
        out = _out_extended.copy()

        del _out_extended
        gc.collect()

        # Edge Detection.
        canny = cv2.Canny(out.astype(np.uint8), 225, 255)
        canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        contours, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        page = sorted(contours, key=cv2.contourArea, reverse=True)[0]

        # ==========================================
        epsilon = 0.02 * cv2.arcLength(page, True)
        corners = cv2.approxPolyDP(page, epsilon, True)

        corners = np.concatenate(corners).astype(np.float32)

        corners[:, 0] -= half
        corners[:, 1] -= half

        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y

        # check if corners are inside.
        # if not find smallest enclosing box, expand_image then extract document
        # else extract document

        if not (np.all(corners.min(axis=0) >= (0, 0)) and np.all(corners.max(axis=0) <= (imW, imH))):

            left_pad, top_pad, right_pad, bottom_pad = 0, 0, 0, 0

            rect = cv2.minAreaRect(corners.reshape((-1, 1, 2)))
            box = cv2.boxPoints(rect)
            box_corners = np.int32(box)
            #     box_corners = minimum_bounding_rectangle(corners)

            box_x_min = np.min(box_corners[:, 0])
            box_x_max = np.max(box_corners[:, 0])
            box_y_min = np.min(box_corners[:, 1])
            box_y_max = np.max(box_corners[:, 1])

            # Find corner point which doesn't satify the image constraint
            # and record the amount of shift required to make the box
            # corner satisfy the constraint
            if box_x_min <= 0:
                left_pad = abs(box_x_min) + BUFFER

            if box_x_max >= imW:
                right_pad = (box_x_max - imW) + BUFFER

            if box_y_min <= 0:
                top_pad = abs(box_y_min) + BUFFER

            if box_y_max >= imH:
                bottom_pad = (box_y_max - imH) + BUFFER

            # new image with additional zeros pixels
            image_extended = np.zeros((top_pad + bottom_pad + imH, left_pad + right_pad + imW, C), dtype=image.dtype)

            # adjust original image within the new 'image_extended'
            image_extended[top_pad : top_pad + imH, left_pad : left_pad + imW, :] = image
            image_extended = image_extended.astype(np.float32)

            # shifting 'box_corners' the required amount
            box_corners[:, 0] += left_pad
            box_corners[:, 1] += top_pad

            corners = box_corners
            image = image_extended

        corners = sorted(corners.tolist())
        corners = self.order_points(corners)
        destination_corners = self.find_dest(corners)
        M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))

        document = cv2.warpPerspective(image, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
        document = np.clip(document, a_min=0., a_max=255.)


        base_name_saved = f'prep_scan_{image_filename}'
        file_path_save = Path(RESULT_FOLDER_PATH,image_filename,base_name_saved).as_posix()
        print("file_path_save", file_path_save)
        cv2.imwrite(file_path_save, document)

        #plt.figure(figsize=(10, 5))
        #plt.subplot(1, 2, 1)
        #plt.imshow(image)
        #plt.subplot(1, 2, 2)
        #plt.imshow(document / 255.0)
        #plt.show()
        #plt.close()

        return document , file_path_save

  

