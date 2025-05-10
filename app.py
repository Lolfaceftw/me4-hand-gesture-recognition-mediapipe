#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import warnings
import datetime
from datetime import timezone, timedelta

# Add before other imports
def suppress_absl_warnings():
    """Suppress ABSL warning messages programmatically"""
    # Try multiple methods to suppress ABSL warnings
    try:
        # Method 1: Using absl.logging directly
        try:
            from absl import logging
            logging.set_verbosity(logging.ERROR)
        except ImportError:
            pass
            
        # Method 2: Override the ABSL logger's stream
        import logging as py_logging
        loggers = [py_logging.getLogger(name) for name in py_logging.root.manager.loggerDict]
        for logger in loggers:
            if 'absl' in logger.name.lower():
                logger.setLevel(py_logging.ERROR)
                for handler in logger.handlers:
                    if hasattr(handler, 'setLevel'):
                        handler.setLevel(py_logging.ERROR)
    except Exception:
        # If anything goes wrong, just continue
        pass

# Special filter for TensorFlow warnings
class TFLiteWarningFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.blocked_phrases = [
            "Created TensorFlow Lite XNNPACK delegate",
            "All log messages before absl::InitializeLog",
            "Feedback manager requires a model with a single signature",
            "tensorflow:",
            "TensorFlow ",
            "W0000",  # This catches the W0000 timestamp prefix used by Mediapipe
            "inference_feedback_manager",
            "Disabling support for feedback tensors"
        ]
    
    def write(self, text):
        # Skip writing if text contains any of the blocked phrases
        if not any(phrase in text for phrase in self.blocked_phrases):
            self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()
    
    # Add these methods so it behaves like a file
    def fileno(self):
        return self.original_stderr.fileno()
    
    def isatty(self):
        return self.original_stderr.isatty()
    
    def close(self):
        pass  # Don't close the underlying stderr

# Parse only the debug flag first, before any imports that might produce warnings
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--debug", action="store_true", help="Show debug messages and warnings")
args, _ = parser.parse_known_args()

# Apply suppression if not in debug mode
if not args.debug:
    suppress_absl_warnings()

# Set up warning suppression
if not args.debug:
    # Basic environment variable suppression
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=debug, 1=info, 2=warning, 3=error
    os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = '0'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
    os.environ["GLOG_minloglevel"] = "2"
    os.environ["ABSL_LOGGING_LEVEL"] = "50"
    
    # Python warnings
    warnings.filterwarnings("ignore")
    
    # Stderr redirection
    sys.stderr = TFLiteWarningFilter(sys.stderr)

# Now import the rest of the libraries
import csv
import copy
import itertools
import pyautogui # For actions
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
    # Create a new parser with all options, including the debug flag we already parsed
    parser = argparse.ArgumentParser(
        description="Hand Gesture Recognition for Presentation Control",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--device", type=int, default=0,
                      help="Camera device number to use")
    parser.add_argument("--width", help="Camera capture width", type=int, default=960)
    parser.add_argument("--height", help="Camera capture height", type=int, default=540)
    parser.add_argument("--use_static_image_mode", action="store_true",
                      help="Enable static image mode for MediaPipe")
    parser.add_argument("--min_detection_confidence", type=float, default=0.7,
                      help="Minimum confidence value for hand detection")
    parser.add_argument("--min_tracking_confidence", type=float, default=0.5,
                      help="Minimum confidence value for hand tracking")
    parser.add_argument("--disable_webcam", action="store_true",
                      help="Run without webcam (for debugging or environments without camera)")
    parser.add_argument("--debug", action="store_true",
                      help="Show debug messages and warnings")
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    disable_webcam = args.disable_webcam
    debug_mode = args.debug
    
    # FPS display variables
    last_fps_update_time = 0
    display_fps = 0  # Initialize to 0
    fps_update_interval = 1.0  # Update FPS display once per second

    # Timezone adjustment for UTC+8 (Philippines)
    ph_timezone = timezone(timedelta(hours=8))
    
    # Print debug mode status
    if debug_mode:
        print("Debug mode: ON - Showing all warnings and debug messages")
    else:
        print("Debug mode: OFF - Suppressing warnings (use --debug to show them)")
        
        # Additional measures to prevent warnings from being shown
        try:
            # Suppress Python warnings
            import warnings
            warnings.filterwarnings('ignore')
            
            # Suppress TensorFlow warnings directly
            import tensorflow as tf
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            
            # Suppress all other warnings from Python libraries
            import logging
            logging.getLogger().setLevel(logging.ERROR)
            
            # Forcibly silence ABSL warning facility
            logging.getLogger('absl').disabled = True
        except Exception:
            # Continue even if these attempts fail
            pass

    # Initialize webcam or create a dummy video source
    if not disable_webcam:
        cap = cv.VideoCapture(cap_device)
        if not cap.isOpened():
            print(f"Error: Could not open camera device {cap_device}")
            print("Try running with --disable_webcam if no camera is available")
            return
        cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
        
        # Reset any camera buffer to ensure direct frame capture
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
    else:
        print("Running in no-webcam mode. Creating a blank image for demonstration.")
        # We'll create a dummy blank image for processing

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Initialize classifiers
    keypoint_classifier = KeyPointClassifier()

    # Redirect stdout/stderr during model loading
    if not debug_mode:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        null_out = open(os.devnull, 'w')
        sys.stdout = null_out
        sys.stderr = null_out

    try:
        point_history_classifier = PointHistoryClassifier()
        # Temporarily restore stdout to print success message
        if not debug_mode:
            sys.stdout = old_stdout
        print("PointHistoryClassifier loaded successfully.")
    except Exception as e:
        # Restore stdout for error messages
        if not debug_mode:
            sys.stdout = old_stdout
        print(f"Error loading PointHistoryClassifier: {e}")
        print("Dynamic gesture recognition will be disabled.")
        point_history_classifier = None
    finally:
        # Restore stdout/stderr
        if not debug_mode:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            null_out.close()

    # Load label files
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

    # Initialize variables
    cvFpsCalc = CvFpsCalc(buffer_len=10)  # Use smaller buffer for direct FPS readings
    history_length = 16
    point_history = deque(maxlen=history_length)
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0
    previous_gesture_action_name = None
    action_cooldown_until = 0.0

    # For visual display of the last action
    last_action_display_name = None
    action_display_until = 0.0

    # Define gesture names with defaults
    do_nothing_gesture_name = "Do Nothing"
    next_slide_gesture_name = "Next Slide"
    previous_slide_gesture_name = "Previous Slide"

    # Update gesture names if available from labels
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

    # UI visibility states and tracking
    ui_visible = True  # UI visibility state
    
    # Print instruction information
    print("\n===== 5-Pointer Hand Gesture Recognition =====")
    print("App started. Point history (5-finger gestures) should be active.")
    print(f"Actions '{next_slide_gesture_name}' and '{previous_slide_gesture_name}' have a {ACTION_COOLDOWN_SECONDS}s cooldown.")
    print(f"Action gestures will be displayed for {ACTION_GESTURE_DISPLAY_DURATION_SECONDS}s after execution.")
    print(f"'{do_nothing_gesture_name}' is the neutral state.")
    print("\n===== Controls =====")
    print("Press 'k' for KeyPoint logging mode, 'h' for PointHistory logging mode, 'n' for Normal mode.")
    print("Press 'v' to toggle visibility of landmarks and point history.")
    print("Press 'ESC' to quit.")
    print("====================================\n")

    # Create dummy image for no-webcam mode
    if disable_webcam:
        dummy_image = np.zeros((cap_height, cap_width, 3), dtype=np.uint8)
        # Add some text to the dummy image
        cv.putText(dummy_image, "No Webcam Mode", (int(cap_width/4), int(cap_height/2)), 
                  cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(dummy_image, "Press ESC to exit", (int(cap_width/4), int(cap_height/2) + 50), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 1, cv.LINE_AA)

    # Main loop
    while True:
        # Calculate FPS
        current_fps = cvFpsCalc.get()
        
        # Get current time
        current_time = time.time()
        
        # Update the display FPS once per second
        if current_time - last_fps_update_time >= fps_update_interval:
            display_fps = current_fps
            # Debug FPS calculation (only in debug mode)
            if debug_mode:
                fps_debug = cvFpsCalc.debug_info()
                print(f"FPS Info: {display_fps} (avg frame time: {fps_debug['avg_frame_time_ms']}ms)")
            last_fps_update_time = current_time
        
        # Use 1ms wait to ensure consistent frame processing without skipping
        key = cv.waitKey(1)
        if key == 27: break  # ESC to exit

        # Handle 'v' key for toggling UI visibility
        if key == ord('v'):
            ui_visible = not ui_visible
            print(f"UI visibility: {'ON' if ui_visible else 'OFF'}")

        number_from_key, selected_mode = select_mode(key, mode)
        mode = selected_mode
        
        # Get frame from camera or use dummy image
        if not disable_webcam:
            # Get a real frame from camera - no buffering, no skipping
            ret, frame = cap.read()
            if not ret: 
                print("Failed to get frame from camera. Exiting...")
                break
                
            image = cv.flip(frame, 1)  # Mirror display
        else:
            # Use dummy image in no-webcam mode
            image = dummy_image.copy()
            # Add current time to show it's updating
            time_text = time.strftime("%H:%M:%S")
            cv.putText(image, f"Time: {time_text}", (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 1, cv.LINE_AA)
            ret = True
            
        debug_image = image.copy()
        
        # Process image with MediaPipe
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        # Temporarily redirect stdout and stderr during MediaPipe processing
        if not debug_mode:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            null_out = open(os.devnull, 'w')
            sys.stdout = null_out
            sys.stderr = null_out
        
        try:
            results = hands.process(image)
        finally:
            # Restore stdout and stderr
            if not debug_mode:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                null_out.close()
        
        image.flags.writeable = True
        
        # UI visibility flag
        if ui_visible:
            # Set default gesture when no hand is detected
            current_gesture = do_nothing_gesture_name
            
            # Process hand landmarks if detected
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    # Calculate landmark list
                    landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                    # Extract fingertip coordinates
                    current_finger_coords = []
                    if landmark_list:
                        for idx in FINGERTIP_INDICES:
                            if idx < len(landmark_list): current_finger_coords.append(landmark_list[idx])
                            else: current_finger_coords.append([0,0])
                    else:
                        current_finger_coords = [[0, 0]] * NUM_FINGERS

                    # Add to history and preprocess
                    point_history.append(current_finger_coords)
                    pre_processed_point_history_list = pre_process_point_history(debug_image, point_history)

                    # Handle logging modes
                    if mode == 1 and landmark_list:
                       pre_proc_kp_list = pre_process_landmark(landmark_list)
                       logging_csv(number_from_key, mode, pre_proc_kp_list, None)
                    elif mode == 2 and pre_processed_point_history_list:
                       logging_csv(number_from_key, mode, None, pre_processed_point_history_list)

                    # Classify dynamic gesture using point history
                    dynamic_gesture_id = point_history_classifier.invalid_value if point_history_classifier else 0
                    if point_history_classifier is not None:
                        if len(pre_processed_point_history_list) == (history_length * NUM_FINGERS * 2):
                            try:
                                dynamic_gesture_id = point_history_classifier(pre_processed_point_history_list)
                            except ValueError as e:
                                print(f"PointHistoryClassifier Error: {e}. Check model input requirements.")
                                dynamic_gesture_id = point_history_classifier.invalid_value
                    
                    # Filter gestures with history
                    finger_gesture_history.append(dynamic_gesture_id)
                    most_common_fg_tuples = Counter(finger_gesture_history).most_common(1)

                    final_dynamic_gesture_id = point_history_classifier.invalid_value if point_history_classifier else 0
                    if most_common_fg_tuples:
                        final_dynamic_gesture_id = most_common_fg_tuples[0][0]

                    # Get gesture name
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
                        if not is_forced_display_active:
                            last_action_display_name = None

                    # Track current gesture
                    current_gesture = display_gesture_name
                    is_do_nothing = (current_gesture == do_nothing_gesture_name)
                    
                    # --- ACTIONS BASED ON POINT HISTORY (using current_dynamic_gesture_name) ---
                    action_taken_this_frame = False

                    if current_dynamic_gesture_name == next_slide_gesture_name or \
                       current_dynamic_gesture_name == previous_slide_gesture_name:
                        if current_dynamic_gesture_name != previous_gesture_action_name:
                            if not is_action_cooldown_active:
                                if current_dynamic_gesture_name == next_slide_gesture_name:
                                    if not disable_webcam:
                                        pyautogui.press("right")
                                    print(f"Action: Point History -> {next_slide_gesture_name}")
                                    action_taken_this_frame = True
                                elif current_dynamic_gesture_name == previous_slide_gesture_name:
                                    if not disable_webcam:
                                        pyautogui.press("left")
                                    print(f"Action: Point History -> {previous_slide_gesture_name}")
                                    action_taken_this_frame = True

                                if action_taken_this_frame:
                                    previous_gesture_action_name = current_dynamic_gesture_name
                                    action_cooldown_until = current_time + ACTION_COOLDOWN_SECONDS
                                    last_action_display_name = current_dynamic_gesture_name
                                    action_display_until = current_time + ACTION_GESTURE_DISPLAY_DURATION_SECONDS
                                    display_gesture_name = last_action_display_name
                                    print(f"Cooldown started for {ACTION_COOLDOWN_SECONDS}s. Displaying '{last_action_display_name}' for {ACTION_GESTURE_DISPLAY_DURATION_SECONDS}s.")

                    elif current_dynamic_gesture_name == do_nothing_gesture_name:
                        if previous_gesture_action_name != do_nothing_gesture_name:
                            previous_gesture_action_name = do_nothing_gesture_name
                        if not is_forced_display_active:
                            display_gesture_name = do_nothing_gesture_name
                            last_action_display_name = None

                    # Draw landmarks and connections
                    mp_drawing = mp.solutions.drawing_utils
                    mp_drawing_styles = mp.solutions.drawing_styles
                    
                    # Only draw hand landmarks if not in "Do Nothing" gesture
                    if current_gesture != do_nothing_gesture_name:
                        # Draw hand landmarks
                        if landmark_list:
                            # Draw connections between landmarks
                            for idx1, idx2 in [(0,1), (1,2), (2,3), (3,4), (0,5), (5,6), 
                                             (6,7), (7,8), (0,9), (9,10), (10,11), (11,12),
                                             (0,13), (13,14), (14,15), (15,16), (0,17), 
                                             (17,18), (18,19), (19,20), (5,9), (9,13), (13,17)]:
                                if idx1 < len(landmark_list) and idx2 < len(landmark_list):
                                    p1 = tuple(landmark_list[idx1])
                                    p2 = tuple(landmark_list[idx2])
                                    cv.line(debug_image, p1, p2, (0,0,0), 6)
                                    cv.line(debug_image, p1, p2, (255,255,255), 2)
                                    
                            # Draw landmark points
                            fingertip_draw_color = (152,251,152)  # Light green
                            for i, lm_coord in enumerate(landmark_list):
                                if lm_coord is None: continue
                                point = tuple(lm_coord)
                                radius, color = (8, fingertip_draw_color) if i in FINGERTIP_INDICES else (5, (255,255,255))
                                cv.circle(debug_image, point, radius, color, -1)
                                cv.circle(debug_image, point, radius, (0,0,0), 1)
                    
                    # Draw gesture name
                    cv.putText(debug_image, "Gesture: " + display_gesture_name, 
                              (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv.LINE_AA)
                    cv.putText(debug_image, "Gesture: " + display_gesture_name, 
                              (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv.LINE_AA)
                
                # Draw point history
                if point_history:
                    # Only draw point history if not in "Do Nothing" gesture
                    if current_gesture != do_nothing_gesture_name:
                        # Draw finger trails for 5 fingers
                        trail_color = (152,251,152)  # Light green
                        
                        for finger_idx in range(NUM_FINGERS):
                            prev_point = None
                            for time_idx, points_at_t in enumerate(point_history):
                                if finger_idx < len(points_at_t) and points_at_t[finger_idx] is not None:
                                    pt = points_at_t[finger_idx]
                                    if pt[0]!=0 or pt[1]!=0:
                                        # Draw with fading effect based on age
                                        radius = 1 + int(time_idx/2.5)
                                        
                                        # Draw circle
                                        cv.circle(debug_image, (pt[0], pt[1]), radius, 
                                                 trail_color, -1, cv.LINE_AA)
                                        
                                        # Draw connecting line
                                        if prev_point and (prev_point[0]!=0 or prev_point[1]!=0):
                                            cv.line(debug_image, prev_point, (pt[0], pt[1]), 
                                                   trail_color, 1, cv.LINE_AA)
                                            
                                        prev_point = (pt[0], pt[1])
            
            # Get current time in UTC+8 (Philippines)
            ph_time = datetime.datetime.now(timezone(timedelta(hours=8)))
            formatted_time = ph_time.strftime("%H:%M:%S")
            
            # Format FPS with consistent decimal places
            fps_text = f"{display_fps:.1f}" if display_fps else "0.0"
            
            # Draw FPS counter with timezone
            cv.putText(debug_image, f"FPS: {fps_text} | Time: {formatted_time}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 4, cv.LINE_AA)
            cv.putText(debug_image, f"FPS: {fps_text} | Time: {formatted_time}", 
                      (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv.LINE_AA)

            # Draw mode info
            mode_map = {0: "Normal", 1: "Log KeyPoint", 2: "Log PointHistory"}
            mode_str = mode_map.get(mode, "Unknown Mode")
            if mode in [1, 2] and number_from_key != -1:
                mode_str += f" (Label: {number_from_key})"

            mode_text_y_pos = 90
            cv.putText(debug_image, "MODE:" + mode_str, 
                      (10, mode_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
            cv.putText(debug_image, "MODE:" + mode_str, 
                      (10, mode_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv.LINE_AA)

            # Draw cooldown timer if active
            remaining_cooldown = action_cooldown_until - current_time
            if remaining_cooldown > 0:
                cooldown_text_y_pos = mode_text_y_pos + 30
                wait_text = f"Waiting: {int(round(remaining_cooldown))}s"
                cv.putText(debug_image, wait_text, 
                          (10, cooldown_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 4, cv.LINE_AA)
                cv.putText(debug_image, wait_text, 
                          (10, cooldown_text_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2, cv.LINE_AA)

            # Add help text at bottom of screen
            help_y_pos = debug_image.shape[0] - 20
            cv.putText(debug_image, "Controls: 'v':Toggle UI | 'k':KeyPoint Mode | 'h':History Mode | 'n':Normal Mode | ESC:Exit", 
                      (10, help_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 4, cv.LINE_AA)
            cv.putText(debug_image, "Controls: 'v':Toggle UI | 'k':KeyPoint Mode | 'h':History Mode | 'n':Normal Mode | ESC:Exit", 
                      (10, help_y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv.LINE_AA)
        
        # Show the image window
        window_title = "ME4: Gesture Control of Powerpoint | Christian Klein C. Ramos"
        if disable_webcam:
            window_title += " (No Webcam Mode)"
        if debug_mode:
            window_title += " [DEBUG]"
        cv.imshow(window_title, debug_image)

    # Clean up resources
    if not disable_webcam:
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

if __name__ == "__main__":
    main()