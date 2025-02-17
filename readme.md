# Body Posture Detection

This project uses OpenCV and MediaPipe to detect and monitor body posture during exercises. It supports up to 10 different exercise positions and provides real-time feedback on form correctness.

## Supported Exercises
1. Hand Raise
2. Hand Curl
3. Shoulder Press
4. Squat
5. Lunge
6. Plank
7. Push Up
8. Sit Up
9. Leg Raise
10. Side Plank

## Controls
- Press '1' for Hand Raise exercise
- Press '2' for Hand Curl exercise
- Press '3' for Shoulder Press exercise
- Press '4' for Squat exercise
- Press '5' for Lunge exercise
- Press '6' for Plank exercise
- Press '7' for Push Up exercise
- Press '8' for Sit Up exercise
- Press '9' for Leg Raise exercise
- Press '0' for Side Plank exercise
- Press 'ESC' to exit

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
Run the `posture_detection1.py` script to start the exercise monitor.

```bash
python posture_detection1.py
```

## Description
The script captures video from the webcam and uses MediaPipe to detect body landmarks. It calculates angles between key points to determine if the exercise form is correct. If the form is incorrect for a certain number of frames, an alarm sound is played to alert the user.
