import cv2
import numpy as np
import mediapipe as mp


class Drawer:
    def __init__(self):
        self.canvas = np.zeros([1000, 1000, 3], dtype=np.uint8)

    def draw(self, image, hands_detection, is_draw: bool = False):
        h, w, c = image.shape
        self.canvas = self.canvas[:h, :w, :]

        if is_draw:
            pos_x, pos_y = hands_detection[0][8]
            cv2.circle(self.canvas, (pos_x, pos_y), 3, (255, 255, 255), -1)

        return cv2.bitwise_or(image, self.canvas)