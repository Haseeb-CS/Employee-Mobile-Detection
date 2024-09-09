import cv2
from ultralytics import YOLO
import numpy as np

class MobileDetection:
    def __init__(self):
        self.class_list_mobile = ['mobile']
        self.detection_colors_mobile = [(255, 0, 0)]
        self.model_mobile = YOLO("best.pt", "v8")
        self.mobile_detected = False

    def update_mobile_status(self, is_detected):
        self.mobile_detected = is_detected

    def detect_mobiles(self, frame):
        detect_params_mobile = self.model_mobile.predict(source=[frame], conf=0.60, save=False)
        DP_mobile = detect_params_mobile[0].numpy()

        is_mobile_detected = False  # Initialize to False

        if len(DP_mobile) != 0:
            is_mobile_detected = True  
        self.update_mobile_status(is_mobile_detected)
