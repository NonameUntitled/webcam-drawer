import cv2
import numpy as np

from handDetector import HandDetector
from fingerCountDetector import FingerCountDetector
from drawerUtil import Drawer

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    detector = HandDetector(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    fingerCountDetector = FingerCountDetector()
    drawer = Drawer()

    while True:
        success, img = cap.read()

        img_flip = cv2.flip(img, 1)

        hands_result = detector.process(img_flip)
        fingers_result = fingerCountDetector.process(hands_result)

        isDraw = False
        if len(hands_result) > 0:
            isDraw = fingers_result[0][1] and not fingers_result[0][2] and not fingers_result[0][3] and not fingers_result[0][4]

        img_flip = drawer.draw(img_flip, hands_result, isDraw)

        # for hand_result in hands_result:
        #     for point in hand_result:
        #         cv2.circle(img_flip, point, 5, (0, 0, 0), -1)

        cv2.imshow("webcam", img_flip)

        cv2.waitKey(10)

    cap.release()