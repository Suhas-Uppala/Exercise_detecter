import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading

# Function to play the alarm sound in a separate thread
def play_alarm():
    playsound("alarm.wav")  # Ensure "alarm.wav" exists in your working directory

# Helper function to decide if a finger is extended.
# We compare the distance from the wrist to the tip versus wrist to the MCP joint.
def is_extended(landmarks, wrist_idx, mcp_idx, tip_idx):
    wrist = np.array(landmarks[wrist_idx])
    mcp = np.array(landmarks[mcp_idx])
    tip = np.array(landmarks[tip_idx])
    return np.linalg.norm(tip - wrist) > np.linalg.norm(mcp - wrist) * 1.2

# Given 21 hand landmarks as a list of (x,y), determine extension status for each finger.
def get_finger_status(hand_landmarks):
    status = {}
    # Thumb: use wrist (0), MCP (2) and tip (4)
    status['thumb'] = is_extended(hand_landmarks, 0, 2, 4)
    # Index: MCP (5), tip (8)
    status['index'] = is_extended(hand_landmarks, 0, 5, 8)
    # Middle: MCP (9), tip (12)
    status['middle'] = is_extended(hand_landmarks, 0, 9, 12)
    # Ring: MCP (13), tip (16)
    status['ring'] = is_extended(hand_landmarks, 0, 13, 16)
    # Pinky: MCP (17), tip (20)
    status['pinky'] = is_extended(hand_landmarks, 0, 17, 20)
    return status

# Detect the hand exercise (gesture) from the finger status.
def detect_hand_exercise(finger_status):
    if not any(finger_status.values()):
        return "Fist Clench"
    elif all(finger_status.values()):
        return "Open Hand"
    elif finger_status['thumb'] and not finger_status['index'] and not finger_status['middle'] and not finger_status['ring'] and not finger_status['pinky']:
        return "Thumbs Up"
    elif not finger_status['thumb'] and finger_status['index'] and not finger_status['middle'] and not finger_status['ring'] and not finger_status['pinky']:
        return "Pointing"
    elif not finger_status['thumb'] and finger_status['index'] and finger_status['middle'] and not finger_status['ring'] and not finger_status['pinky']:
        return "Peace Sign"
    else:
        return "Unknown"

# Map keys to exercises (keys 1-5)
exercise_map = {
    ord('1'): "Fist Clench",
    ord('2'): "Open Hand",
    ord('3'): "Thumbs Up",
    ord('4'): "Pointing",
    ord('5'): "Peace Sign"
}
selected_exercise = "Fist Clench"  # default exercise

# Initialize Mediapipe modules: Hands for detecting the hand gesture and Pose for drawing the stick man.
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start capturing video
cap = cv2.VideoCapture(0)
alarm_on = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for a mirror view and convert to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Process hand and pose detections
    hand_results = hands.process(frame_rgb)
    pose_results = pose.process(frame_rgb)

    # Create a separate black image for the stick man (skeleton)
    stick_man = np.zeros_like(frame)
    if pose_results.pose_landmarks:
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = pose_results.pose_landmarks.landmark[start_idx]
            end = pose_results.pose_landmarks.landmark[end_idx]
            start_point = (int(start.x * image_width), int(start.y * image_height))
            end_point = (int(end.x * image_width), int(end.y * image_height))
            cv2.line(stick_man, start_point, end_point, (255, 255, 255), 2)
        # Draw landmark points
        for lm in pose_results.pose_landmarks.landmark:
            cx = int(lm.x * image_width)
            cy = int(lm.y * image_height)
            cv2.circle(stick_man, (cx, cy), 4, (0, 255, 0), -1)

    # Default detected exercise
    detected_exercise = "Unknown"
    if hand_results.multi_hand_landmarks:
        # Use the first detected hand
        hand_landmark_obj = hand_results.multi_hand_landmarks[0]
        # Convert normalized landmarks to a list of (x,y) tuples
        hand_landmarks = [(lm.x, lm.y) for lm in hand_landmark_obj.landmark]
        finger_status = get_finger_status(hand_landmarks)
        detected_exercise = detect_hand_exercise(finger_status)
        # Optionally, draw the hand landmarks on the original frame
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmark_obj, mp_hands.HAND_CONNECTIONS)

    # Compare detected exercise with the selected one and set status accordingly.
    if detected_exercise == selected_exercise:
        status_text = "Good Posture"
        status_color = (0, 255, 0)
    else:
        status_text = "Wrong Exercise"
        status_color = (0, 0, 255)
        if not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm, daemon=True).start()

    # Overlay the selected exercise, detected gesture, and status on the main video frame.
    cv2.putText(frame, f"Selected: {selected_exercise}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Detected: {detected_exercise}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, status_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    # Display the two windows: one for the live hand exercise feed and one for the stick man.
    cv2.imshow("Hand Exercise", frame)
    cv2.imshow("Stick Man", stick_man)

    # Check for key presses:
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key in exercise_map:
        selected_exercise = exercise_map[key]
        alarm_on = False  # reset alarm flag when changing exercises

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()
