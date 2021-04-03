import cv2
import numpy as np
import mediapipe as mp


class HandDetector:
    def __init__(
            self,
            max_num_hands: int = 1,
            min_detection_confidence: float = 0.5,
            min_tracking_confidence: float = 0.5
    ):
        self.mp_hands_solution = mp.solutions.hands
        self.handDetector = self.mp_hands_solution.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        pass

    def process(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, c = image_rgb.shape

        detection_results = self.handDetector.process(image_rgb)

        results = []
        if detection_results.multi_hand_landmarks:
            for hand_detection_info in detection_results.multi_hand_landmarks:
                hand_result = []
                for id, lm in enumerate(hand_detection_info.landmark):
                    detected_x = lm.x
                    detected_y = lm.y
                    image_x, image_y = int(detected_x * w), int(detected_y * h)
                    hand_result.append((image_x, image_y))
                results.append(hand_result)

        return results


if __name__ == '__main__':
    pass