#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
import pyautogui # For actions
import os
import time # For cooldown and display timing
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

ACTION_COOLDOWN_SECONDS = 2.0 # Cooldown period for performing actions
ACTION_GESTURE_DISPLAY_DURATION_SECONDS = 1.0 # How long to visually show an action gesture after it's performed

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help="cap width", type=int, default=960)
    parser.add_argument("--height", help="cap height", type=int, default=540)
    parser.add_argument("--use_static_image_mode", action="store_true")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7)
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5)
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
    
    try:
        point_history_classifier = PointHistoryClassifier()
        print("PointHistoryClassifier loaded successfully.")
    except Exception as e:
        print(f"Error loading PointHistoryClassifier: {e}")
        print("Dynamic gesture recognition will be disabled.")
        point_history_classifier = None

    keypoint_classifier_labels = []
    keypoint_label_path = "model/keypoint_classifier/keypoint_classifier_label.csv"
    try:
        with open(keypoint_label_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            keypoint_classifier_labels = [row[0] for row in reader if row and row[0].strip()]
    except FileNotFoundError:
        print(f"Warning: {keypoint_label_path} not found. (Static hand sign names are disabled from display)")

    point_history_classifier_labels = []
    point_history_label_path = "model/point_history_classifier/point_history_classifier_label.csv"
    try:
        with open(point_history_label_path, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            point_history_classifier_labels = [row[0] for row in reader if row and row[0].strip()]
        if not point_history_classifier_labels and point_history_classifier is not None:
            print(f"Warning: {point_history_label_path} is empty. Dynamic gesture names may not display.")
    except FileNotFoundError:
        if point_history_classifier is not None: 
            print(f"Error: {point_history_label_path} not found. Dynamic gesture names will not be displayed.")

    cvFpsCalc = CvFpsCalc(buffer_len=10)
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length) 
    
    mode = 0 
    previous_gesture_action_name = None 
    action_cooldown_until = 0.0 
    
    # For visual display of the last action
    last_action_display_name = None 
    action_display_until = 0.0

    do_nothing_gesture_name = "Do Nothing" 
    next_slide_gesture_name = "Next Slide"
    previous_slide_gesture_name = "Previous Slide"

    if point_history_classifier and point_history_classifier_labels:
        if 0 <= point_history_classifier.invalid_value < len(point_history_classifier_labels):
            do_nothing_gesture_name = point_history_classifier_labels[point_history_classifier.invalid_value]
        
        for i, label in enumerate(point_history_classifier_labels):
            if "next" in label.lower() and "slide" in label.lower():
                next_slide_gesture_name = label
                print(f"Found Next Slide gesture: '{label}' at index {i}")
            elif "previous" in label.lower() and "slide" in label.lower():
                previous_slide_gesture_name = label
                print(f"Found Previous Slide gesture: '{label}' at index {i}")
    
    previous_gesture_action_name = do_nothing_gesture_name


    print("App started. Point history (5-finger gestures) should be active.")
    print(f"Actions '{next_slide_gesture_name}' and '{previous_slide_gesture_name}' have a {ACTION_COOLDOWN_SECONDS}s cooldown.")
    print(f"Action gestures will be displayed for {ACTION_GESTURE_DISPLAY_DURATION_SECONDS}s after execution.")
    print(f"'{do_nothing_gesture_name}' is the neutral state.")
    print("Press 'k' for KeyPoint logging mode, 'h' for PointHistory logging mode, 'n' for Normal mode.")
    print("Press 'ESC' to quit.")

    while True:
        fps = cvFpsCalc.get()
        current_time = time.time()
        key = cv.waitKey(10)
        if key == 27: break
        
        number_from_key, selected_mode = select_mode(key, mode)
        mode = selected_mode 

        ret, image = cap.read()
        if not ret: break
        image = cv.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        current_dynamic_gesture_name = do_nothing_gesture_name 
        display_gesture_name = do_nothing_gesture_name 

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                current_finger_coords = []
                if landmark_list: 
                    for idx in FINGERTIP_INDICES:
                        if idx < len(landmark_list): current_finger_coords.append(landmark_list[idx])
                        else: current_finger_coords.append([0,0]) 
                else: 
                    current_finger_coords = [[0, 0]] * NUM_FINGERS
                
                point_history.append(current_finger_coords)
                pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                if mode == 1 and landmark_list: 
                   pre_proc_kp_list = pre_process_landmark(landmark_list)
                   logging_csv(number_from_key, mode, pre_proc_kp_list, None)
                elif mode == 2 and pre_processed_point_history_list: 
                   logging_csv(number_from_key, mode, None, pre_processed_point_history_list)

                dynamic_gesture_id = point_history_classifier.invalid_value if point_history_classifier else 0
                if point_history_classifier is not None:
                    if len(pre_processed_point_history_list) == (history_length * NUM_FINGERS * 2):
                        try:
                            dynamic_gesture_id = point_history_classifier(pre_processed_point_history_list)
                        except ValueError as e:
                            print(f"PointHistoryClassifier Error: {e}. Check model input requirements.")
                            dynamic_gesture_id = point_history_classifier.invalid_value 
                
                finger_gesture_history.append(dynamic_gesture_id)
                most_common_fg_tuples = Counter(finger_gesture_history).most_common(1)
                
                final_dynamic_gesture_id = point_history_classifier.invalid_value if point_history_classifier else 0
                if most_common_fg_tuples:
                    final_dynamic_gesture_id = most_common_fg_tuples[0][0]
                
                if point_history_classifier_labels and 0 <= final_dynamic_gesture_id < len(point_history_classifier_labels):
                    current_dynamic_gesture_name = point_history_classifier_labels[final_dynamic_gesture_id]
                else:
                     current_dynamic_gesture_name = f"GestureID:{final_dynamic_gesture_id}"

                # --- Determine Display Gesture Name ---
                is_action_cooldown_active = current_time < action_cooldown_until
                is_forced_display_active = current_time < action_display_until

                if is_forced_display_active and last_action_display_name:
                    display_gesture_name = last_action_display_name
                elif is_action_cooldown_active and \
                     (current_dynamic_gesture_name == next_slide_gesture_name or \
                      current_dynamic_gesture_name == previous_slide_gesture_name):
                    display_gesture_name = do_nothing_gesture_name
                else:
                    display_gesture_name = current_dynamic_gesture_name
                    # If forced display ends, clear last_action_display_name
                    if not is_forced_display_active:
                        last_action_display_name = None


                # --- ACTIONS BASED ON POINT HISTORY (using current_dynamic_gesture_name) ---
                action_taken_this_frame = False
                
                if current_dynamic_gesture_name == next_slide_gesture_name or \
                   current_dynamic_gesture_name == previous_slide_gesture_name:
                    if current_dynamic_gesture_name != previous_gesture_action_name: 
                        if not is_action_cooldown_active: # Check action cooldown, not display cooldown
                            if current_dynamic_gesture_name == next_slide_gesture_name:
                                pyautogui.press("right")
                                print(f"Action: Point History -> {next_slide_gesture_name}")
                                action_taken_this_frame = True
                            elif current_dynamic_gesture_name == previous_slide_gesture_name:
                                pyautogui.press("left")
                                print(f"Action: Point History -> {previous_slide_gesture_name}")
                                action_taken_this_frame = True
                            
                            if action_taken_this_frame:
                                previous_gesture_action_name = current_dynamic_gesture_name
                                action_cooldown_until = current_time + ACTION_COOLDOWN_SECONDS # For action logic
                                
                                # For visual display
                                last_action_display_name = current_dynamic_gesture_name
                                action_display_until = current_time + ACTION_GESTURE_DISPLAY_DURATION_SECONDS
                                display_gesture_name = last_action_display_name # Ensure it's displayed immediately

                                print(f"Cooldown started for {ACTION_COOLDOWN_SECONDS}s. Displaying '{last_action_display_name}' for {ACTION_GESTURE_DISPLAY_DURATION_SECONDS}s.")
                        # else: Cooldown active, action suppressed (message handled by display logic)
                
                elif current_dynamic_gesture_name == do_nothing_gesture_name:
                    if previous_gesture_action_name != do_nothing_gesture_name : 
                        previous_gesture_action_name = do_nothing_gesture_name 
                    # If we are in "Do Nothing" and not in forced display, ensure display is "Do Nothing"
                    if not is_forced_display_active:
                        display_gesture_name = do_nothing_gesture_name
                        last_action_display_name = None # Clear forced display if we are truly in "Do Nothing"

                debug_image = draw_landmarks(debug_image, landmark_list) 
                debug_image = draw_info_text(
                    debug_image,
                    display_gesture_name 
                )
        else: 
            point_history.append([[0, 0]] * NUM_FINGERS) 
            current_dynamic_gesture_name = do_nothing_gesture_name 
            
            is_forced_display_active = current_time < action_display_until
            if is_forced_display_active and last_action_display_name:
                 display_gesture_name = last_action_display_name
            else:
                 display_gesture_name = do_nothing_gesture_name
                 last_action_display_name = None # Clear if no hand and not in forced display

            if previous_gesture_action_name != do_nothing_gesture_name: 
                previous_gesture_action_name = do_nothing_gesture_name


        debug_image = draw_point_history(debug_image, point_history)
        
        number_for_draw_info = -1
        if mode == 1 or mode == 2: 
            number_for_draw_info = number_from_key
        
        is_gesture_text_active_for_draw_info = False
        if display_gesture_name not in ["", "N/A", "Unknown", f"GestureID:{point_history_classifier.invalid_value if point_history_classifier else 0}"]:
             is_gesture_text_active_for_draw_info = True
        
        debug_image = draw_info(debug_image, fps, mode, number_for_draw_info, 
                                is_gesture_text_active_for_draw_info, action_cooldown_until, current_time)
        cv.imshow("ME4: Gesture Control of Powerpoint | Christian Klein C. Ramos", debug_image)

    cap.release()
    cv.destroyAllWindows()


def select_mode(key, current_mode): 
    pressed_digit = -1 
    new_mode = current_mode
    if 48 <= key <= 57: pressed_digit = key - 48 
    if key == 110: new_mode = 0; print("Mode: Normal") 
    if key == 107: new_mode = 1; print("Mode: Log KeyPoint") 
    if key == 104: new_mode = 2; print("Mode: Log PointHistory") 
    return pressed_digit, new_mode


def logging_csv(number, mode, landmark_list, point_history_list_flat):
    if mode == 1 and (0 <= number <= 9) and landmark_list:
        csv_path = "model/keypoint_classifier/keypoint.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
        print(f"Logged KeyPoint for label {number}")
    if mode == 2 and (0 <= number <= 9) and point_history_list_flat:
        csv_path = "model/point_history_classifier/point_history.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *point_history_list_flat])
        print(f"Logged PointHistory for label {number}, features: {len(point_history_list_flat)}")
    return

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
    if not landmark_list: return [] 
    temp_landmark_list = copy.deepcopy(landmark_list)
    base_x, base_y = 0, 0
    if temp_landmark_list: 
        base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
        for index, landmark_point in enumerate(temp_landmark_list):
            temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
            temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
    
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    if not temp_landmark_list: return [] 

    max_value = max(list(map(abs, temp_landmark_list))) if temp_landmark_list else 0
    def normalize_(n): return n / max_value if max_value != 0 else 0
    return list(map(normalize_, temp_landmark_list))

def pre_process_point_history(image, point_history_deque_of_lists):
    image_width, image_height = image.shape[1], image.shape[0]
    if image_width == 0: image_width = 1 
    if image_height == 0: image_height = 1 
    
    processed_history_flat = []
    if not point_history_deque_of_lists: return []

    base_x_ref, base_y_ref = 0,0
    found_ref = False
    for t_step_points in point_history_deque_of_lists: 
        if t_step_points and len(t_step_points) > 0 and t_step_points[0] and \
           (t_step_points[0][0] != 0 or t_step_points[0][1] != 0): 
            base_x_ref, base_y_ref = t_step_points[0][0], t_step_points[0][1]
            found_ref = True; break
    
    if not found_ref and point_history_deque_of_lists and \
       point_history_deque_of_lists[0] and \
       len(point_history_deque_of_lists[0]) > 0 and \
       point_history_deque_of_lists[0][0] is not None: 
        base_x_ref = point_history_deque_of_lists[0][0][0]
        base_y_ref = point_history_deque_of_lists[0][0][1]

    for points_at_t in point_history_deque_of_lists:
        for finger_idx in range(NUM_FINGERS):
            if finger_idx < len(points_at_t) and points_at_t[finger_idx] is not None:
                finger_point = points_at_t[finger_idx]
                if finger_point[0] == 0 and finger_point[1] == 0:
                    norm_x, norm_y = 0.0, 0.0
                else:
                    norm_x = (finger_point[0] - base_x_ref) / image_width
                    norm_y = (finger_point[1] - base_y_ref) / image_height
                processed_history_flat.extend([norm_x, norm_y])
            else: 
                processed_history_flat.extend([0.0, 0.0]) 
    return processed_history_flat

def draw_landmarks(image, landmark_point):
    if not landmark_point or len(landmark_point) < 21: return image 
    lines = [ 
        (0,1), (1,2), (2,3), (3,4), 
        (0,5), (5,6), (6,7), (7,8), 
        (0,9), (9,10), (10,11), (11,12), 
        (0,13), (13,14), (14,15), (15,16), 
        (0,17), (17,18), (18,19), (19,20), 
        (5,9), (9,13), (13,17) 
    ]
    for p1_idx, p2_idx in lines:
        if p1_idx < len(landmark_point) and p2_idx < len(landmark_point):
            p1 = tuple(landmark_point[p1_idx])
            p2 = tuple(landmark_point[p2_idx])
            cv.line(image, p1, p2, (0,0,0),6)
            cv.line(image, p1, p2, (255,255,255),2)

    fingertip_draw_color = (152,251,152) 
    for i, lm_coord in enumerate(landmark_point):
        if lm_coord is None: continue 
        point = tuple(lm_coord)
        radius, color = (8, fingertip_draw_color) if i in FINGERTIP_INDICES else (5, (255,255,255))
        cv.circle(image, point, radius, color, -1)
        cv.circle(image, point, radius, (0,0,0), 1)
    return image

def draw_info_text(image, finger_gesture_text_to_display):
    if finger_gesture_text_to_display not in ["", "N/A", "Unknown"]: 
        cv.putText(image, "Gesture: " + finger_gesture_text_to_display, (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.8,(0,0,0),4,cv.LINE_AA)
        cv.putText(image, "Gesture: " + finger_gesture_text_to_display, (10,60), cv.FONT_HERSHEY_SIMPLEX, 0.8,(255,255,255),2,cv.LINE_AA)
    return image

def draw_point_history(image, point_history_deque_of_lists):
    trail_color = (152,251,152) 
    for finger_idx in range(NUM_FINGERS):
        for time_idx, points_at_t in enumerate(point_history_deque_of_lists):
            if finger_idx < len(points_at_t) and points_at_t[finger_idx] is not None:
                pt = points_at_t[finger_idx]
                if pt[0]!=0 or pt[1]!=0: 
                    cv.circle(image,(pt[0],pt[1]),1+int(time_idx/3),trail_color,2)
    return image

def draw_info(image, fps, mode, number, is_gesture_displayed, cooldown_until_time, current_frame_time):
    cv.putText(image, "FPS:"+str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX,1.0,(0,0,0),4,cv.LINE_AA)
    cv.putText(image, "FPS:"+str(fps), (10,30), cv.FONT_HERSHEY_SIMPLEX,1.0,(255,255,255),2,cv.LINE_AA)
    
    mode_map = {0: "Normal", 1: "Log KeyPoint", 2: "Log PointHistory"}
    mode_str = mode_map.get(mode, "Unknown Mode")
    if mode in [1,2] and number != -1: 
        mode_str += f" (Label: {number})"
    
    mode_text_y_pos = 60 
    if is_gesture_displayed: 
        mode_text_y_pos = 90 
                           
    cv.putText(image, "MODE:"+mode_str, (10, mode_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv.LINE_AA) 
    cv.putText(image, "MODE:"+mode_str, (10, mode_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv.LINE_AA)

    remaining_cooldown = cooldown_until_time - current_frame_time
    if remaining_cooldown > 0:
        cooldown_text_y_pos = mode_text_y_pos + 30 
        wait_text = f"Waiting: {int(round(remaining_cooldown))}s"
        cv.putText(image, wait_text, (10, cooldown_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv.LINE_AA)
        cv.putText(image, wait_text, (10, cooldown_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1, cv.LINE_AA) 

    return image


if __name__ == "__main__":
    main()
