import cv2
import numpy as np
import mediapipe as mp


def dist(fst_point, snd_point):
    fst_x, fst_y = fst_point
    snd_x, snd_y = snd_point
    return (fst_x - snd_x) ** 2 + (fst_y - snd_y) ** 2


class FingerCountDetector:
    def __init__(self):
        self.thumb_finger_points_idxs = [4, 3, 0]
        self.index_finger_points_idxs = [8, 6, 0]
        self.middle_finger_points_idxs = [12, 10, 0]
        self.ring_finger_points_idxs = [16, 14, 0]
        self.pinky_finger_points_idxs = [20, 18, 0]

        self.fingers_list = [
            self.thumb_finger_points_idxs,
            self.index_finger_points_idxs,
            self.middle_finger_points_idxs,
            self.ring_finger_points_idxs,
            self.pinky_finger_points_idxs,
        ]

    def process(self, hands_detection):
        results = []

        for hand_info in hands_detection:
            hand_result = []
            for fst_point_idx, snd_point_idx, main_point_idx in self.fingers_list:
                fst_point = hand_info[fst_point_idx]
                snd_point = hand_info[snd_point_idx]
                main_point = hand_info[main_point_idx]

                if dist(fst_point, main_point) > dist(snd_point, main_point):
                    hand_result.append(1)
                else:
                    hand_result.append(0)

            results.append(hand_result)

        return results
