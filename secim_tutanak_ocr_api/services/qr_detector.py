import os
import cv2
import numpy as np
from pyzbar.pyzbar import decode

def dedection_and_decode_qr_code(image_path, is_save_results=True):
    results_decoded = []
    
    img = cv2.imread(image_path)
    
    for barcode in decode(img):

        #qr code text value
        barcode_text = barcode.data.decode("utf-8")

        #polygon
        barcode_polygon_points = barcode.polygon
        #barcode_polygon_points = np.array([barcode.polygon], np.int32)
        #barcode_polygon = barcode_polygon_points.reshape((-1, 1, 2))

        #bbox
        barcode_bounding_box_points = barcode.rect
        #barcode_bounding_box = (barcode_bounding_box_points[0], barcode_bounding_box_points[1])

        quality = barcode.quality
        orientation = barcode.orientation
	
        results_decoded.append({
            "text": barcode_text,
            "polygon_points": barcode_polygon_points,
            "bounding_box_points": barcode_bounding_box_points,
            "quality": quality,
            "orientation": orientation
        })


        if is_save_results:
            color = (0, 0, 255)

            pts = np.array([barcode.polygon], np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 5)
            pts2 = barcode.rect
            cv2.putText(img, barcode_text, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)


            file_path_save = image_path.replace('prep_scan_','qr_result_')
            print("file_path_save", file_path_save)
            cv2.imwrite(file_path_save, img)


    return results_decoded