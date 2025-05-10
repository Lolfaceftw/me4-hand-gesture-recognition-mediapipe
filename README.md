# 5-Pointer Hand Gesture Recognition using MediaPipe
Estimate hand pose using MediaPipe (Python version) with enhanced 5 pointer tracking.<br> This is a modified 
fork that improves hand signs and finger gestures recognition by tracking all 5 fingers simultaneously.
<br> 
# Fork
<img src="sample.gif" width=300/>

# Original
![mqlrf-s6x16](https://user-images.githubusercontent.com/37477845/102222442-c452cd00-3f26-11eb-93ec-c387c98231be.gif)

This repository contains the following contents:
* Enhanced sample program with 5 pointer tracking
* Hand sign recognition model(TFLite)
* Finger gesture recognition model(TFLite)
* Learning data for hand sign recognition and notebook for learning
* Learning data for finger gesture recognition and notebook for learning

## Enhancement Features
This fork enhances the original project by:
* Enabling simultaneous tracking of all 5 finger pointers
* Improved gesture recognition accuracy
* More robust hand pose estimation
* **Focus on finger trajectory tracking only** (keypoint static pose classification is disabled)
* **Specialization for presentation control** with left and right swipe detection
* **Warning suppression by default** with optional debug mode
* **UI toggle with 'v' key** to hide all UI elements including landmarks and text

## Presentation Control Mode
This fork is specifically trained to control PowerPoint presentations using simple hand gestures:

1. **Left Swipe** → **Previous Slide**: Swipe your hand left to go back to the previous slide
2. **Right Swipe** → **Next Slide**: Swipe your hand right to advance to the next slide
3. **All Other Movements** → **No Action**: Any other hand position or movement does nothing

The system uses only three states and is optimized for presenting in a natural way without having to hold any device.

# Quick Start Guide for Beginners

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/5-pointer-hand-gesture-recognition.git
   cd 5-pointer-hand-gesture-recognition
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```
   If the requirements.txt file doesn't exist, install these packages:
   ```bash
   pip install mediapipe==0.8.1 opencv-python==4.5.1.48 tensorflow==2.4.0 pyautogui
   ```

## Running the Application

### Basic Usage
```bash
python app.py
```

### Options
* `--device 0` - Specify which camera to use (default: 0, which is usually your webcam)
* `--width 960` - Width of the camera capture (default: 960)
* `--height 540` - Height of the camera capture (default: 540)
* `--min_detection_confidence 0.7` - Set how confident the system must be to detect a hand (default: 0.7)
* `--min_tracking_confidence 0.5` - Set how confident the system must be to track a hand (default: 0.5)
* `--disable_webcam` - Run the app without webcam (for testing or environments without a camera)
* `--debug` - Show TensorFlow and Mediapipe warning messages (hidden by default)

### Example
```bash
# Run with a higher resolution
python app.py --width 1280 --height 720

# Run without webcam (for testing or environments without a camera)
python app.py --disable_webcam

# Run with debug messages enabled
python app.py --debug
```

## How to Use

1. **Start the application** using one of the commands above
2. **Position your hand** in front of the camera
3. **Make gestures**:
   - Swipe your hand to the right → Next slide 
   - Swipe your hand to the left → Previous slide
4. **Key controls**:
   - Press `v` to toggle visibility of all UI elements (hand landmarks, trails, gesture text, and controls)
   - Press `n` to enter Normal mode
   - Press `k` to enter KeyPoint collection mode (for training)
   - Press `h` to enter PointHistory collection mode (for training)
   - Press `ESC` to exit the application

## Troubleshooting

- **Camera not found**: Try specifying a different camera device number with `--device 1` or use `--disable_webcam`
- **Low performance**: Lower the resolution using `--width 640 --height 360`
- **Poor detection**: Make sure you're in a well-lit environment and your hand is clearly visible
- **Actions not registering**: Try moving your hand more distinctly when making swipe gestures
- **TensorFlow/Mediapipe warnings**: Use the `--debug` flag to see warning messages that might help diagnose issues

## Technical Implementation
This implementation uses MediaPipe's hand landmark detection but extends it to track all five fingertips simultaneously rather than just the index finger. Here's how it works:

1. **Fingertip Identification**: All five fingertips are identified using their landmark indices:
   ```python
   FINGERTIP_INDICES = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
   ```

2. **Multi-Point History Tracking**: Instead of tracking only the index finger's position history, the system tracks all five fingertips:
   ```python
   # Collect all five fingertip coordinates
   current_finger_coords = []
   for idx in FINGERTIP_INDICES:
       if idx < len(landmark_list): 
           current_finger_coords.append(landmark_list[idx])
       else: 
           current_finger_coords.append([0,0])
   
   # Add to point history
   point_history.append(current_finger_coords)
   ```

3. **Enhanced Preprocessing**: The preprocessing pipeline normalizes all five finger trajectories relative to the base of the hand:
   ```python
   # Process each finger's coordinates at each timestep
   for points_at_t in point_history_deque_of_lists:
       for finger_idx in range(NUM_FINGERS):
           if finger_idx < len(points_at_t) and points_at_t[finger_idx] is not None:
               # Normalize coordinates relative to base point
               norm_x = (finger_point[0] - base_x_ref) / image_width
               norm_y = (finger_point[1] - base_y_ref) / image_height
               processed_history_flat.extend([norm_x, norm_y])
   ```

4. **Visualization**: Each fingertip trajectory is visualized with fading trails to show movement patterns:
   ```python
   # Draw trails for each finger
   for finger_idx in range(NUM_FINGERS):
       for time_idx, points_at_t in enumerate(point_history_deque_of_lists):
           if finger_idx < len(points_at_t) and points_at_t[finger_idx] is not None:
               pt = points_at_t[finger_idx]
               # Create fading trail effect
               cv.circle(image, (pt[0],pt[1]), 1+int(time_idx/3), trail_color, 2)
   ```

5. **Visibility Toggle**: Press 'v' to toggle the visibility of hand landmarks, trails, and all UI elements:
   ```python
   # Toggle visibility of all UI elements
   if key == ord('v'):
       show_ui = not show_ui
   ```

6. **Gesture Action Cooldown**: A cooldown timer prevents accidental detection of gestures when transitioning between different hand positions:
   ```python
   # Define a cooldown period for actions to prevent rapid successive triggers
   ACTION_COOLDOWN_SECONDS = 2.0
   
   # In the main processing loop:
   is_action_cooldown_active = current_time < action_cooldown_until
   
   # When an action is executed:
   if current_dynamic_gesture_name == next_slide_gesture_name or \
      current_dynamic_gesture_name == previous_slide_gesture_name:
       if current_dynamic_gesture_name != previous_gesture_action_name:
           if not is_action_cooldown_active:
               # Execute the action (e.g., press right/left arrow)
               
               # Set cooldown timer
               action_cooldown_until = current_time + ACTION_COOLDOWN_SECONDS
               print(f"Cooldown started for {ACTION_COOLDOWN_SECONDS}s.")
   ```

   This mechanism is crucial for preventing false detections when:
   - Moving from "Next Slide" gesture to neutral position
   - Moving from "Previous Slide" gesture to neutral position
   - Transitioning between different gestures
   
   The cooldown is visually displayed on screen:
   ```python
   # Display the remaining cooldown time
   remaining_cooldown = cooldown_until_time - current_frame_time
   if remaining_cooldown > 0:
       wait_text = f"Waiting: {int(round(remaining_cooldown))}s"
       cv.putText(image, wait_text, (10, cooldown_text_y_pos), 
                 cv.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 1, cv.LINE_AA)
   ```

7. **No-Webcam Mode**: You can run the application without a webcam using the `--disable_webcam` flag:
   ```python
   # Create dummy image for no-webcam mode
   if disable_webcam:
       dummy_image = np.zeros((cap_height, cap_width, 3), dtype=np.uint8)
       # Add some text to the dummy image
       cv.putText(dummy_image, "No Webcam Mode", (int(cap_width/4), int(cap_height/2)), 
                 cv.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv.LINE_AA)
   ```

   This is useful for:
   - Testing the application on systems without a camera
   - Debugging in environments where camera access is restricted
   - Learning how the interface works without actually triggering actions

8. **Warning Suppression**: By default, the application suppresses TensorFlow and Mediapipe warnings to keep the console clean:
   ```python
   # Disable TensorFlow logging
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=debug, 1=info, 2=warning, 3=error
   
   # Disable Mediapipe logging
   os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
   
   # Disable Python warnings
   warnings.filterwarnings("ignore")
   ```

   For debugging purposes, you can enable these warnings with the `--debug` flag.

This approach significantly improves gesture recognition by providing a richer feature set (5x more spatial information) and enabling more complex gesture patterns that involve multiple fingers.

## Mathematical Model Enhancements

The enhancements to the point history classifier fundamentally transform the mathematical model used for gesture recognition. Here's a detailed explanation of the changes:

1. **Expanded Feature Dimensionality**:
   - Original model: Used only `TIME_STEPS × 1 × DIMENSION = 16 × 1 × 2 = 32` features (tracking only index finger)
   - Enhanced model: Uses `TIME_STEPS × NUM_FINGERS × DIMENSION = 16 × 5 × 2 = 160` features (all five fingers)
   
   This 5× increase in feature dimensionality provides much richer motion information for the gesture classifier.

2. **Modified Neural Network Architecture**:
   ```python
   # Define key dimensions
   TIME_STEPS = 16
   DIMENSION = 2
   NUM_FINGERS = 5  # Was implicitly 1 in original
   TOTAL_FEATURES = TIME_STEPS * NUM_FINGERS * DIMENSION  # Now 160 features instead of 32
   ```

3. **LSTM Model Adaptation**:
   The LSTM sequence model was reconfigured to handle the multi-finger input:
   ```python
   model = tf.keras.models.Sequential([
       tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * NUM_FINGERS * DIMENSION, )),
       # Reshape from flat (160,) to sequence (16, 10) where 10 = 5 fingers × 2 coordinates
       tf.keras.layers.Reshape((TIME_STEPS, NUM_FINGERS * DIMENSION)),
       tf.keras.layers.Dropout(0.2),
       tf.keras.layers.LSTM(16),
       tf.keras.layers.Dropout(0.5),
       tf.keras.layers.Dense(10, activation='relu'),
       tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
   ])
   ```

4. **Mathematical Formulation**:
   - Let $P_{t,f} = (x_{t,f}, y_{t,f})$ represent the position of finger $f$ at time step $t$.
   - For each gesture, we now model the temporal sequence $\{P_{1,1}, P_{1,2}, ..., P_{1,5}, P_{2,1}, ..., P_{16,5}\}$
   - This sequence is normalized relative to a base point (hand landmark 0) to ensure scale and translation invariance:
     $P_{t,f}' = \frac{P_{t,f} - P_{base}}{S}$
     where $S$ is the normalization factor (typically image width/height)

5. **Information Gain**:
   - With 5 fingers tracked instead of 1, the system captures complex inter-finger relationships
   - The classifier can now recognize gestures that involve multiple fingers moving in coordinated patterns
   - The feature space encodes relative finger positions, enabling recognition of static poses and dynamic movements

This mathematical enhancement significantly increases the discriminative power of the model, allowing it to recognize a wider range of natural hand gestures with greater accuracy.

# Advanced Topics

## Requirements
* mediapipe 0.8.1
* OpenCV 3.4.2 or Later
* Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model)
* scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix) 
* matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)
* pyautogui (For presentation control with keyboard commands)

## Command Line Options
Here's a complete list of the command line options available:
```bash
python app.py --help
```

Output:
```
usage: app.py [-h] [--device DEVICE] [--width WIDTH] [--height HEIGHT]
              [--use_static_image_mode] [--min_detection_confidence MIN_DETECTION_CONFIDENCE]
              [--min_tracking_confidence MIN_TRACKING_CONFIDENCE] [--disable_webcam]
              [--debug]

Hand Gesture Recognition for Presentation Control

optional arguments:
  -h, --help            show this help message and exit
  --device DEVICE       Camera device number to use (default: 0)
  --width WIDTH         Camera capture width (default: 960)
  --height HEIGHT       Camera capture height (default: 540)
  --use_static_image_mode
                        Enable static image mode for MediaPipe (default: False)
  --min_detection_confidence MIN_DETECTION_CONFIDENCE
                        Minimum confidence value for hand detection (default: 0.7)
  --min_tracking_confidence MIN_TRACKING_CONFIDENCE
                        Minimum confidence value for hand tracking (default: 0.5)
  --disable_webcam      Run without webcam (for debugging or environments without camera) (default: False)
  --debug               Show debug messages and warnings (default: False)
```

## Directory Structure
<pre>
│  app.py                       # Main application file
│  keypoint_classification.ipynb  # Training notebook for hand signs
│  point_history_classification.ipynb  # Training notebook for gestures
│  
├─model                         # Model directory
│  ├─keypoint_classifier        # Hand sign classifier
│  │  │  keypoint.csv           # Training data
│  │  │  keypoint_classifier.hdf5  # Keras model
│  │  │  keypoint_classifier.py   # Inference module
│  │  │  keypoint_classifier.tflite  # TFLite model
│  │  └─ keypoint_classifier_label.csv  # Class labels
│  │          
│  └─point_history_classifier   # Gesture classifier
│      │  point_history.csv     # Training data
│      │  point_history_classifier.hdf5  # Keras model
│      │  point_history_classifier.py  # Inference module
│      │  point_history_classifier.tflite  # TFLite model
│      └─ point_history_classifier_label.csv  # Class labels
│          
└─utils                         # Utility functions
    └─cvfpscalc.py              # FPS calculation
</pre>

## File Descriptions

### app.py
This is the enhanced sample program for inference with 5 pointer tracking.<br>
In addition, learning data (key points) for hand sign recognition,<br>
You can also collect training data (all five finger coordinate history) for finger gesture recognition.

### keypoint_classification.ipynb
This is a model training script for hand sign recognition. Note: In this fork, keypoint classification is disabled in favor of using only the point history classifier.

### point_history_classification.ipynb
This is a model training script for finger gesture recognition. This fork specifically trains a model to recognize three states: "Do Nothing", "Next Slide", and "Previous Slide" using all five finger trajectories.

### model/keypoint_classifier
This directory stores files related to hand sign recognition.<br>
The following files are stored.
* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### model/point_history_classifier
This directory stores files related to finger gesture recognition.<br>
The following files are stored.
* Training data(point_history.csv)
* Trained model(point_history_classifier.tflite)
* Label data(point_history_classifier_label.csv) - Contains three labels: "Do Nothing", "Next Slide", "Previous Slide"
* Inference module(point_history_classifier.py)

### utils/cvfpscalc.py
This is a module for FPS measurement.

# Training (Advanced Users)
Hand sign recognition and finger gesture recognition can add and change training data and retrain the model.

### Hand sign recognition training
#### 1.Learning data collection
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）<br>
<img src="https://user-images.githubusercontent.com/37477845/102235423-aa6cb680-3f35-11eb-8ebd-5d823e211447.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates<br>
<img src="https://user-images.githubusercontent.com/37477845/102345725-28d26280-3fe1-11eb-9eeb-8c938e3f625b.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102242918-ed328c80-3f3d-11eb-907c-61ba05678d54.png" width="80%">
<img src="https://user-images.githubusercontent.com/37477845/102244114-418a3c00-3f3f-11eb-8eef-f658e5aa2d0d.png" width="80%"><br><br>
In the initial state, three types of learning data are included: open hand (class ID: 0), close hand (class ID: 1), and pointing (class ID: 2).<br>
If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.<br>
<img src="https://user-images.githubusercontent.com/37477845/102348846-d0519400-3fe5-11eb-8789-2e7daec65751.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348855-d2b3ee00-3fe5-11eb-9c6d-b8924092a6d8.jpg" width="25%">　<img src="https://user-images.githubusercontent.com/37477845/102348861-d3e51b00-3fe5-11eb-8b07-adc08a48a760.jpg" width="25%">

#### 2.Model training
Open "[keypoint_classification.ipynb](keypoint_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
To change the number of training data classes, change the value of "NUM_CLASSES = 3" <br>and modify the label of "model/keypoint_classifier/keypoint_classifier_label.csv" as appropriate.<br><br>

#### X.Model structure
The image of the model prepared in "[keypoint_classification.ipynb](keypoint_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246723-69c76a00-3f42-11eb-8a4b-7c6b032b7e71.png" width="50%"><br><br>

### Finger gesture recognition training
#### 1.Learning data collection
Press "h" to enter the mode to save the history of fingertip coordinates (displayed as "MODE:Logging Point History").<br>
<img src="https://user-images.githubusercontent.com/37477845/102249074-4d78fc80-3f45-11eb-9c1b-3eb975798871.jpg" width="60%"><br><br>
If you press "0" to "9", the key points will be added to "model/point_history_classifier/point_history.csv" as shown below.<br>
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Coordinate history<br>
<img src="https://user-images.githubusercontent.com/37477845/102345850-54ede380-3fe1-11eb-8d04-88e351445898.png" width="80%"><br><br>
The key point coordinates are the ones that have undergone the following preprocessing up to ④.<br>
<img src="https://user-images.githubusercontent.com/37477845/102244148-49e27700-3f3f-11eb-82e2-fc7de42b30fc.png" width="80%"><br><br>
In this fork, 3 types of gesture data are included: Do Nothing (class ID: 0), Next Slide (class ID: 1), and Previous Slide (class ID: 2).<br>
If necessary, you can add more gesture types by adding training data and modifying the classifier.<br>
<img src="https://user-images.githubusercontent.com/37477845/102350939-02b0c080-3fe9-11eb-94d8-54a3decdeebc.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350945-05131a80-3fe9-11eb-904c-a1ec573a5c7d.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350951-06444780-3fe9-11eb-98cc-91e352edc23c.jpg" width="20%">　<img src="https://user-images.githubusercontent.com/37477845/102350942-047a8400-3fe9-11eb-9103-dbf383e67bf5.jpg" width="20%">

#### 2.Model training
Open "[point_history_classification.ipynb](point_history_classification.ipynb)" in Jupyter Notebook and execute from top to bottom.<br>
This notebook has been modified to use all five finger coordinates in the model training, significantly increasing the feature dimensions from 32 to 160.<br><br>

#### X.Model structure
The image of the model prepared in "[point_history_classification.ipynb](point_history_classification.ipynb)" is as follows.
<img src="https://user-images.githubusercontent.com/37477845/102246771-7481ff00-3f42-11eb-8ddf-9e3cc30c5816.png" width="50%"><br>
The model using "LSTM" is as follows. <br>Please change "use_lstm = False" to "True" when using (tf-nightly required (as of 2020/12/16))<br>
<img src="https://user-images.githubusercontent.com/37477845/102246817-8368b180-3f42-11eb-9851-23a7b12467aa.png" width="60%">

# Reference
* [MediaPipe](https://mediapipe.dev/)
* [Original Repository by Kazuhito Takahashi](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
* [English Translated Fork by Nikita Kiselov](https://github.com/kinivi/hand-gesture-recognition-mediapipe)

# Author
Forked and enhanced by Christian Klein C. Ramos
 
# License 
hand-gesture-recognition-using-mediapipe is under [Apache v2 license](LICENSE).
