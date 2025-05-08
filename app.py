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
        type=float, # Changed from int to float to match mediapipe's typical range
        default=0.5,
    )

    args = parser.parse_args()

    return args

# stop = 0 # This global variable 'stop' was defined but not used. Removed.

def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open(
        "model/keypoint_classifier/keypoint_classifier_label.csv", encoding="utf-8-sig"
    ) as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in keypoint_classifier_labels]
    with open(
        "model/point_history_classifier/point_history_classifier_label.csv",
        encoding="utf-8-sig",
    ) as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    # point_history now stores a list of 5 fingertip coords [ [x,ythumb], [x,yindex], ..., [x,ypinky] ] for each time step
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0
    previous_point_name = None # For hand sign based actions
    # previous_gesture_name = None # For gesture based actions (if you re-enable gesture text)

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates for hand sign
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Store current 5 fingertip coordinates
                current_finger_coords = []
                for fingertip_idx in FINGERTIP_INDICES:
                    current_finger_coords.append(landmark_list[fingertip_idx])
                point_history.append(current_finger_coords)

                # Pre-process point history for gesture classification
                # This now processes the history of 5 fingers
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history 
                )
                
                # Write to the dataset file
                # logging_csv now handles the new multi-finger pre_processed_point_history_list
                logging_csv(
                    number,
                    mode,
                    pre_processed_landmark_list, # For keypoint.csv
                    pre_processed_point_history_list, # For point_history.csv (now 5 fingers)
                )

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                # Finger gesture classification (using 5 finger history)
                finger_gesture_id = 0 # Default: No gesture
                # Check if the pre_processed_point_history_list has the expected length for the model
                # Expected length: history_length * NUM_FINGERS * 2 (coords per finger)
                if len(pre_processed_point_history_list) == (history_length * NUM_FINGERS * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list
                    )

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(finger_gesture_history).most_common()
                
                current_gesture_name = "Unknown"
                if len(most_common_fg_id) > 0 and most_common_fg_id[0][0] < len(point_history_classifier_labels):
                     current_gesture_name = point_history_classifier_labels[most_common_fg_id[0][0]]


                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list) # Will draw green circles on fingertips
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    current_gesture_name, # Use current_gesture_name
                )
                
                # Actions based on hand sign (static pose)
                current_hand_sign_name = keypoint_classifier_labels[hand_sign_id]
                target_hand_signs = ["Previous Slide", "Next Slide"] # Static poses
                if current_hand_sign_name in target_hand_signs and current_hand_sign_name != previous_point_name:
                    if current_hand_sign_name == "Next Slide":
                        pyautogui.press("right")
                    elif current_hand_sign_name == "Previous Slide":
                        pyautogui.press("left")
                    print(f"Hand Sign Action: {current_hand_sign_name}")
                previous_point_name = current_hand_sign_name

                # TODO: Add actions based on dynamic finger_gesture_id (swipes) if needed
                # Example:
                # target_gestures = ["Swipe Left", "Swipe Right"] # Dynamic gestures from point_history_classifier
                # if current_gesture_name in target_gestures and current_gesture_name != previous_gesture_name:
                #    if current_gesture_name == "Swipe Right":
                #        pyautogui.press("right")
                #    elif current_gesture_name == "Swipe Left":
                #        pyautogui.press("left")
                #    print(f"Gesture Action: {current_gesture_name}")
                # previous_gesture_name = current_gesture_name

        else: # No hand landmarks detected
            # Append five [0,0] points to represent no detection for each finger
            no_detection_points = [[0, 0]] * NUM_FINGERS
            point_history.append(no_detection_points)

        # Draw history trails for all 5 fingers
        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number)
        
        # Screen reflection #############################################################
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
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]


def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])
    return landmark_point


def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0: # Wrist
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(list(map(abs, temp_landmark_list)))
    def normalize_(n):
        return n / max_value if max_value != 0 else 0 # Avoid division by zero
    temp_landmark_list = list(map(normalize_, temp_landmark_list))
    return temp_landmark_list


def pre_process_point_history(image, point_history_deque_of_lists):
    """
    Processes a deque where each element is a list of 5 finger [x,y] coordinates.
    Normalizes all coordinates relative to the thumb tip of the oldest entry in the deque.
    Returns a flat list of normalized coordinates for all fingers over time.
    Expected output length: history_length * NUM_FINGERS * 2
    """
    image_width, image_height = image.shape[1], image.shape[0]
    
    # Ensure image_width and image_height are not zero to prevent division by zero
    if image_width == 0: image_width = 1 
    if image_height == 0: image_height = 1

    processed_history_flat = []

    if not point_history_deque_of_lists:
        return []

    # Use the thumb tip (first finger in FINGERTIP_INDICES) of the oldest
    # time step in the deque as the reference for normalization.
    # point_history_deque_of_lists[0] is the oldest list of 5 finger coords.
    # point_history_deque_of_lists[0][0] is the thumb coords [x,y] from that oldest entry.
    base_x_ref, base_y_ref = point_history_deque_of_lists[0][0][0], point_history_deque_of_lists[0][0][1]

    for points_at_t in point_history_deque_of_lists:  # Iterate through time steps in the deque
        for finger_point in points_at_t:  # Iterate through the 5 finger points at this time step
            # finger_point is [x, y]
            if finger_point[0] == 0 and finger_point[1] == 0:
                # If the original point was (0,0) (e.g., no detection), keep normalized as (0,0)
                norm_x = 0.0
                norm_y = 0.0
            else:
                norm_x = (finger_point[0] - base_x_ref) / image_width
                norm_y = (finger_point[1] - base_y_ref) / image_height
            
            processed_history_flat.append(norm_x)
            processed_history_flat.append(norm_y)
            
    return processed_history_flat


def logging_csv(number, mode, landmark_list, point_history_list_flat):
    # Mode 0: Normal operation, no logging
    # Mode 1: Log keypoint data (static hand pose)
    if mode == 1 and (0 <= number <= 9): # Ensure number is valid for labels
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    # Mode 2: Log point history data (dynamic gesture, now 5 fingers)
    if mode == 2 and (0 <= number <= 9): # Ensure number is valid for labels
        # point_history_list_flat is already prepared by pre_process_point_history
        # It will contain history_length * NUM_FINGERS * 2 values
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list_flat])
    return


def draw_landmarks(image, landmark_point):
    # Lines (connections)
    if len(landmark_point) > 0:
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
    fingertip_draw_color = (152, 251, 152) # Light green, same as point_history trails
    # fingertip_draw_color = (0, 255, 0) # Bright Green
    
    for index, landmark in enumerate(landmark_point):
        radius = 5
        color = (255, 255, 255) # Default white for non-fingertips

        if index in FINGERTIP_INDICES:
            radius = 8
            color = fingertip_draw_color # Green for current fingertips
        
        # Draw filled circle
        cv.circle(image, (landmark[0], landmark[1]), radius, color, -1)
        # Draw border
        cv.circle(image, (landmark[0], landmark[1]), radius, (0, 0, 0), 1)
    
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    info_text = handedness.classification[0].label[0:]
    if hand_sign_text != "" and hand_sign_text != "Unknown": # Assuming "Unknown" is a possible label
        info_text = info_text + ":" + hand_sign_text
    cv.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    # Display finger gesture text if available (you can re-enable if needed)
    if finger_gesture_text != "" and finger_gesture_text != "Unknown" and finger_gesture_text != point_history_classifier_labels[0]: # Don't show if it's the "invalid" or default
        cv.putText(image, "Gesture: " + finger_gesture_text, (10, 60), # Adjusted y-position
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv.LINE_AA) # Increased size slightly
        cv.putText(image, "Gesture: " + finger_gesture_text, (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv.LINE_AA)
    return image


def draw_point_history(image, point_history_deque_of_lists):
    """Draws history trails for all 5 fingers."""
    trail_color = (152, 251, 152) # Light green
    
    # Iterate through each of the 5 fingers
    for finger_trace_idx in range(NUM_FINGERS):
        # Iterate through the time steps in the deque for this specific finger
        for time_idx, points_at_t in enumerate(point_history_deque_of_lists):
            # points_at_t is a list of 5 [x,y] coords for that time step
            # Get the specific finger's point for this time step
            point = points_at_t[finger_trace_idx] 
            
            if point[0] != 0 and point[1] != 0: # Check if it's a valid point
                # Radius increases slightly for older points in the trail
                radius = 1 + int(time_idx / 3) # Adjusted scaling for trail thickness
                cv.circle(image, (point[0], point[1]), radius, trail_color, 2)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv.LINE_AA)

    mode_string = ["Normal", "Logging KeyPoint", "Logging PointHistory"] # Adjusted mode names
    if 0 <= mode < len(mode_string): # Check mode is valid
        display_mode_text = "MODE: " + mode_string[mode]
        if mode == 1 or mode == 2: # If logging mode, show number
             if 0 <= number <= 9: # Check number is valid
                display_mode_text += " (Label: " + str(number) + ")"
        
        cv.putText(image, display_mode_text, (10, 90 if mode == 0 else 120), # Adjust y pos based on gesture text
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


if __name__ == "__main__":
    main()