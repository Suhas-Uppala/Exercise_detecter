# AI Exercise Form Trainer

This project uses computer vision (OpenCV and MediaPipe) to provide real-time feedback on exercise form. It monitors arm exercises through your webcam and alerts you when your form needs correction.

## Features
- Real-time pose detection and tracking
- Split-screen display with pose overlay and stick figure visualization
- Instant audio and visual feedback on form
- Precise angle measurements for proper form validation
- Easy-to-use keyboard controls for exercise selection

## Supported Exercises

### 1. Hand Raise
- Goal: Raise arm vertically overhead
- Form Check:
  - Arm should be within 30° of vertical
  - Wrist position above nose level
  - Maintain straight arm alignment

### 2. Bicep Curl
- Goal: Controlled elbow flexion
- Form Check:
  - Peak position: elbow angle < 50°
  - Stable upper arm position
  - Smooth, controlled movement

### 3. Lateral Raise
- Goal: Raise arm horizontally to side
- Form Check:
  - Arm parallel to ground (±30°)
  - Minimal elbow bend
  - Shoulder-level position

### 4. Front Raise
- Goal: Raise arm forward
- Form Check:
  - Minimal side-to-side deviation
  - Raise to shoulder height or above
  - Maintain relatively straight arm

### 5. Arm Extension
- Goal: Full arm straightening
- Form Check:
  - Elbow angle > 160°
  - Controlled extension
  - Avoid hyperextension

## Controls
- Press '1': Hand Raise
- Press '2': Bicep Curl
- Press '3': Lateral Raise
- Press '4': Front Raise
- Press '5': Arm Extension
- Press 'q': Exit program

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- playsound

## Installation
```bash
pip install opencv-python mediapipe playsound
```

## Usage
1. Ensure your webcam is connected
2. Run the program:
```bash
python posture_detection1.py
```
3. Select an exercise using number keys
4. Position yourself 2-3 meters from camera
5. Perform the exercise and watch for feedback

## Technical Details
The application uses:
- MediaPipe's pose detection for skeletal tracking
- Real-time angle calculations between body segments
- Computer vision for form analysis
- Audio-visual feedback system

## Best Practices
- Ensure good lighting conditions
- Wear fitted clothing for better tracking
- Keep your full upper body in frame
- Face the camera directly
- Allow 2-3 meters distance from camera

## Feedback System
- Visual indicators show current form status
- Audio alerts trigger when form needs correction
- Real-time angle measurements displayed
- Stick figure visualization for movement reference

## Note
This version focuses on upper body exercises, specifically arm movements. The system provides immediate feedback to help maintain proper form throughout each exercise.
