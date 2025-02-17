from flask import Flask, jsonify, request
import cv2
import mediapipe as mp
import math
import threading
from playsound import playsound

app = Flask(__name__)

# Constants for exercise angles
HAND_RAISE_MIN_ANGLE = 150
HAND_CURL_MAX_ANGLE = 120
SHOULDER_PRESS_MIN_ANGLE = 160

wrong_posture_detected = False
cap = None
running = False
exercise_type = "hand_raise"

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    """Calculates the angle at point b"""
    ba = (a[0] - b[0], a[1] - b[1])
    bc = (c[0] - b[0], c[1] - b[1])
    
    dot_product = ba[0] * bc[0] + ba[1] * bc[1]
    magnitude_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    if magnitude_ba * magnitude_bc == 0:
        return 0

    angle_rad = math.acos(dot_product / (magnitude_ba * magnitude_bc))
    return math.degrees(angle_rad)

def is_exercise_incorrect(shoulder_angle, elbow_angle, exercise):
    """Checks if posture is incorrect"""
    if exercise == "hand_raise":
        return shoulder_angle < HAND_RAISE_MIN_ANGLE
    elif exercise == "hand_curl":
        return elbow_angle > HAND_CURL_MAX_ANGLE
    elif exercise == "shoulder_press":
        return shoulder_angle < SHOULDER_PRESS_MIN_ANGLE
    return False

def play_alarm_sound():
    """Plays an alarm sound when incorrect posture is detected"""
    global wrong_posture_detected
    if wrong_posture_detected:
        playsound('alarm.wav')

def get_landmark_coords(landmarks, landmark_point, w, h):
    """Extracts landmark coordinates"""
    return (
        int(landmarks[landmark_point].x * w),
        int(landmarks[landmark_point].y * h)
    )

def detect_posture():
    """Runs OpenCV posture analysis"""
    global wrong_posture_detected, cap, running, exercise_type
    cap = cv2.VideoCapture(0)
    running = True
    wrong_form_counter = 0
    threshold_wrong_frames = 30

    while cap.isOpened() and running:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            h, w, _ = image.shape
            landmarks = results.pose_landmarks.landmark

            left_wrist = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_WRIST.value, w, h)
            left_elbow = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_ELBOW.value, w, h)
            left_shoulder = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_SHOULDER.value, w, h)
            left_hip = get_landmark_coords(landmarks, mp_pose.PoseLandmark.LEFT_HIP.value, w, h)

            shoulder_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
            elbow_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)

            if is_exercise_incorrect(shoulder_angle, elbow_angle, exercise_type):
                wrong_form_counter += 1
                if wrong_form_counter >= threshold_wrong_frames:
                    wrong_posture_detected = True
                    threading.Thread(target=play_alarm_sound, daemon=True).start()
                    wrong_form_counter = 0
            else:
                wrong_form_counter = 0
                wrong_posture_detected = False

        cv2.imshow("Posture Detection", image)
        if cv2.waitKey(1) & 0xFF == 27:
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/start', methods=['GET'])
def start_posture_detection():
    """Starts the posture detection"""
    global running
    if not running:
        threading.Thread(target=detect_posture, daemon=True).start()
    return jsonify({"status": "Posture detection started"})

@app.route('/status', methods=['GET'])
def get_posture_status():
    """Gets the current posture status"""
    return jsonify({"wrong_posture": wrong_posture_detected})

@app.route('/stop', methods=['GET'])
def stop_posture_detection():
    """Stops the posture detection"""
    global running
    running = False
    return jsonify({"status": "Posture detection stopped"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
