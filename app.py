#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import pyautogui
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# Landmark indices for fingertips
FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
NUM_FINGERS = len(FINGERTIP_INDICES)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)

    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument(
        "--min_detection_confidence",
        help="min_detection_confidence",
        type=float,
        default=0.7,
    )
    parser.add_argument(
        "--min_tracking_confidence",
        help="min_tracking_confidence",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = True

    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier() # Default invalid_value is 0

    # Read labels ###########################################################
    keypoint_classifier_labels = []
    keypoint_label_path = "model/keypoint_classifier/keypoint_classifier_label.csv"
    try:
        with open(keypoint_label_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in reader if row and row[0].strip()]
        if not keypoint_classifier_labels:
            print(f"Warning: {keypoint_label_path} is empty or contains no valid labels.")
    except FileNotFoundError:
        print(f"Error: {keypoint_label_path} not found. Hand sign names will not be displayed.")

    point_history_classifier_labels = []
    point_history_label_path = "model/point_history_classifier/point_history_classifier_label.csv"
    try:
        with open(point_history_label_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            point_history_classifier_labels = [row[0] for row in reader if row and row[0].strip()]
        if not point_history_classifier_labels:
            print(f"Warning: {point_history_label_path} is empty or contains no valid labels. Gesture names may not display.")
    except FileNotFoundError:
        print(f"Error: {point_history_label_path} not found. Gesture names will not be displayed.")

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)
    mode = 0
    previous_point_name = None

    while True:
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 27: break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret: break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)

                current_finger_coords = []
                if landmark_list: # Ensure landmark_list is not empty
                    for idx in FINGERTIP_INDICES:
                        if idx < len(landmark_list):
                             current_finger_coords.append(landmark_list[idx])
                        else: # Should not happen if mediapipe returns full landmarks
                            current_finger_coords.append([0,0]) 
                else: # No landmarks, append zeros
                    current_finger_coords = [[0, 0]] * NUM_FINGERS
                
                point_history.append(current_finger_coords)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                logging_csv(number, mode, pre_processed_landmark_list, pre_processed_point_history_list)

                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)

                finger_gesture_id = point_history_classifier.invalid_value # Default to invalid
                if len(pre_processed_point_history_list) == (history_length * NUM_FINGERS * 2):
                    finger_gesture_id = point_history_classifier(pre_processed_point_history_list)

                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_tuples = Counter(finger_gesture_history).most_common(1)

                current_gesture_name = "Unknown"
                final_gesture_id = point_history_classifier.invalid_value
                if most_common_fg_tuples:
                    final_gesture_id = most_common_fg_tuples[0][0]
                    if point_history_classifier_labels and \
                       0 <= final_gesture_id < len(point_history_classifier_labels):
                        current_gesture_name = point_history_classifier_labels[final_gesture_id]

                hand_sign_display_name = "Unknown"
                if keypoint_classifier_labels and \
                   0 <= hand_sign_id < len(keypoint_classifier_labels):
                    hand_sign_display_name = keypoint_classifier_labels[hand_sign_id]

                invalid_gesture_label_to_hide = None
                if point_history_classifier_labels and \
                   0 <= point_history_classifier.invalid_value < len(point_history_classifier_labels):
                    invalid_gesture_label_to_hide = point_history_classifier_labels[point_history_classifier.invalid_value]

                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    hand_sign_display_name,
                    current_gesture_name,
                    invalid_gesture_label_to_hide
                )

                target_hand_signs = ["Previous Slide", "Next Slide"]
                if hand_sign_display_name in target_hand_signs and hand_sign_display_name != previous_point_name:
                    if hand_sign_display_name == "Next Slide": pyautogui.press("right")
                    elif hand_sign_display_name == "Previous Slide": pyautogui.press("left")
                    print(f"Hand Sign Action: {hand_sign_display_name}")
                previous_point_name = hand_sign_display_name
        else:
            point_history.append([[0, 0]] * NUM_FINGERS)

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        cv.imshow("ME4: Gesture Control of Powerpoint | Christian Klein C. Ramos", debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0 # Normal mode
    if key == 107:  # k
        mode = 1 # Logging Key Point (static hand pose)
    if key == 104:  # h
        mode = 2 # Logging Point History (dynamic gesture)
    return number, mode


def calc_bounding_rect(image, landmarks):
    if not landmarks or not landmarks.landmark: return None
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    if landmark_array.shape[0] == 0: return None
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    if not landmarks or not landmarks.landmark: return []
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    if not landmark_list: return [] # Guard for empty list
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    if temp_landmark_list:
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1] # Wrist
        for index, landmark_point in enumerate(temp_landmark_list):
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    if not temp_landmark_list: return []

    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 0
    def normalize_(n):
        return n / max_value if max_value != 0 else 0
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image, point_history_deque_of_lists):
    image_width, image_height = image.shape[1], image.shape[0]
    if image_width == 0: image_width = 1
    if image_height == 0: image_height = 1
    processed_history_flat = []
    if not point_history_deque_of_lists: return []

    base_x_ref, base_y_ref = 0, 0
    found_ref = False
    # Try to find the oldest valid thumb tip as reference
    for t_step_points in point_history_deque_of_lists: # oldest to newest
        if t_step_points and len(t_step_points) > 0 and t_step_points[0] and \
           (t_step_points[0][0] != 0 or t_step_points[0][1] != 0):
            base_x_ref, base_y_ref = t_step_points[0][0], t_step_points[0][1]
            found_ref = True
            break
    
    if not found_ref and point_history_deque_of_lists[0] and \
       len(point_history_deque_of_lists[0]) > 0 and point_history_deque_of_lists[0][0]:
        # Fallback: use the oldest thumb tip even if it's (0,0)
        base_x_ref, base_y_ref = point_history_deque_of_lists[0][0][0], point_history_deque_of_lists[0][0][1]


    for points_at_t in point_history_deque_of_lists:
        for finger_idx in range(NUM_FINGERS):
            if finger_idx < len(points_at_t):
                finger_point = points_at_t[finger_idx]
                if finger_point[0] == 0 and finger_point[1] == 0:
                    norm_x, norm_y = 0.0, 0.0
                else:
                    norm_x = (finger_point[0] - base_x_ref) / image_width
                    norm_y = (finger_point[1] - base_y_ref) / image_height
                processed_history_flat.extend([norm_x, norm_y])
            else: # Should not happen if point_history is populated correctly
                processed_history_flat.extend([0.0, 0.0]) 

    return processed_history_flat


def logging_csv(number, mode, landmark_list, point_history_list_flat):
    if mode == 1 and (0 <= number <= 9):
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    if mode == 2 and (0 <= number <= 9):
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list_flat])
    return


def draw_landmarks(image, landmark_point):
    if not landmark_point or len(landmark_point) < 21: return image # Guard if not enough points

    # Lines (connections)
    # Thumb
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), (255, 255, 255), 2)
    # Index finger
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), (255, 255, 255), 2)
    # Middle finger
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), (255, 255, 255), 2)
    # Ring finger
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), (255, 255, 255), 2)
    # Little finger
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), (255, 255, 255), 2)
    # Palm
    cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), (255, 255, 255), 2)
    cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (0, 0, 0), 6); cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), (255, 255, 255), 2)

    # Key Points (Circles)
    fingertip_draw_color = (152, 251, 152)
    for index, landmark in enumerate(landmark_point):
        radius = 5
        color = (255, 255, 255)
        if index in FINGERTIP_INDICES:
            radius = 8
            color = fingertip_draw_color
        cv.circle(image, (landmark[0], landmark[1]), radius, color, -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect and brect and len(brect) == 4:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text, invalid_gesture_label_to_hide=None):
    if brect and len(brect) == 4:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
        info_text = ""
        if handedness and handedness.classification and handedness.classification[0]:
             info_text = handedness.classification[0].label[0:]

        if hand_sign_text != "" and hand_sign_text != "Unknown":
            info_text = info_text + ":" + hand_sign_text
        cv.putText(
            image,
            info_text,
            (brect[0] + 5, brect[1] - 4),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

    show_gesture = finger_gesture_text != "" and finger_gesture_text != "Unknown"
    if invalid_gesture_label_to_hide is not None:
        show_gesture = show_gesture and (finger_gesture_text != invalid_gesture_label_to_hide)

    if show_gesture:
        cv.putText(image, "Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    return image


def draw_point_history(image, point_history_deque_of_lists):
    trail_color = (152, 251, 152)
    for finger_trace_idx in range(NUM_FINGERS):
        for time_idx, points_at_t in enumerate(point_history_deque_of_lists):
            if finger_trace_idx < len(points_at_t):
                point = points_at_t[finger_trace_idx]
                if point[0] != 0 or point[1] != 0: # Only draw if not (0,0)
                    radius = 1 + int(time_idx / 3)
                    cv.circle(image, (point[0], point[1]), radius, trail_color, 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ["Normal", "Logging KeyPoint", "Logging PointHistory"]
    # Determine y-position for mode text based on whether gesture text might be displayed
    # This is a heuristic. A more robust way would be to track if gesture text was actually drawn.
    y_pos_mode_text = 90 if mode == 0 else 120 # Assume gesture text might be at 60 if mode is 0

    if 0 <= mode < len(mode_string):
        display_mode_text = "MODE: " + mode_string[mode]
        if mode in [1, 2] and (0 <= number <= 9):
            display_mode_text += " (Label: " + str(number) + ")"

        cv.putText(image, display_mode_text, (10, y_pos_mode_text),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


if __name__ == "__main__":
    main()