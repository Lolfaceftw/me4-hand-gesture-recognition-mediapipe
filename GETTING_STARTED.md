# Getting Started with 5-Pointer Hand Gesture Recognition

This guide will help you get up and running with the 5-Pointer Hand Gesture Recognition system for controlling presentations using hand gestures.

## What You Need

- A computer with a webcam (or you can run in no-webcam mode)
- Python 3.6 or newer
- Basic understanding of command line operations

## Step 1: Install Python (if needed)

If you don't have Python installed, download and install it from [python.org](https://www.python.org/downloads/). Make sure to check the "Add Python to PATH" option during installation.

## Step 2: Download the Code

### Option A: Using Git
```bash
git clone https://github.com/your-username/5-pointer-hand-gesture-recognition.git
cd 5-pointer-hand-gesture-recognition
```

### Option B: Download ZIP
1. Go to the GitHub repository
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a location on your computer
5. Open a command prompt/terminal and navigate to the extracted folder

## Step 3: Set Up the Environment

### Windows:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Step 4: Run the Application

```bash
python app.py
```

That's it! You should now see a window showing your webcam feed with hand tracking enabled.

## Basic Controls

- **Move your hand to the right**: Go to next slide
- **Move your hand to the left**: Go to previous slide
- **Press v**: Toggle visibility of hand landmarks and trails
- **Press ESC**: Exit the application

## No Webcam?

You can still run the application without a webcam:

```bash
python app.py --disable_webcam
```

This will show a demonstration mode where you can see the interface without camera input.

## Troubleshooting

### "Module not found" errors
Make sure you've installed all requirements:
```bash
pip install -r requirements.txt
```

### Camera not detected
Try specifying a different camera:
```bash
python app.py --device 1
```

### Poor performance
Lower the resolution:
```bash
python app.py --width 640 --height 360
```

## Next Steps

Once you're comfortable with the basic operation, check out the full README.md for more advanced options and features.

Happy presenting! 