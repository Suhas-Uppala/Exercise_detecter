import cv2
import mediapipe as mp
import numpy as np
import math
from playsound import playsound
import threading

# Function to play the alarm sound (non-blocking)
def play_alarm():
    playsound("alarm.wav")  # Make sure "alarm.wav" is in your working directory

# Calculate the angle between three points (with b as the vertex)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# Calculate the angle between two vectors (in degrees)
def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0
    cos_theta = dot / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

# Map keys 1-5 to exercise names
exercise_map = {
    ord('1'): "Squat",
    ord('2'): "Push-Up",
    ord('3'): "Lunge",
    ord('4'): "Plank",
    ord('5'): "Shoulder Press"
}
selected_exercise = "Squat"  # Default exercise

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
alarm_on = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for a mirror view and convert BGR to RGB
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Process the image for pose landmarks
    results = pose.process(frame_rgb)

    # Create a black image for the stick man (skeleton)
    stick_man = np.zeros_like(frame)

    detected_exercise = "Unknown"

    # Initialize variables for joint angles and conditions
    left_elbow_angle = None
    right_elbow_angle = None
    left_knee_angle = None
    right_knee_angle = None
    angle_with_vertical = None
    horizontal = False

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key landmarks (normalized coordinates)
        nose = (landmarks[0].x, landmarks[0].y)
        left_shoulder = (landmarks[11].x, landmarks[11].y)
        right_shoulder = (landmarks[12].x, landmarks[12].y)
        left_elbow = (landmarks[13].x, landmarks[13].y)
        right_elbow = (landmarks[14].x, landmarks[14].y)
        left_wrist = (landmarks[15].x, landmarks[15].y)
        right_wrist = (landmarks[16].x, landmarks[16].y)
        left_hip = (landmarks[23].x, landmarks[23].y)
        right_hip = (landmarks[24].x, landmarks[24].y)
        left_knee = (landmarks[25].x, landmarks[25].y)
        right_knee = (landmarks[26].x, landmarks[26].y)
        left_ankle = (landmarks[27].x, landmarks[27].y)
        right_ankle = (landmarks[28].x, landmarks[28].y)

        # Compute joint angles for arms and legs
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)

        # Check if body is nearly horizontal (for Push-Up and Plank)
        relevant_y = [
            landmarks[11].y, landmarks[12].y,  # shoulders
            landmarks[23].y, landmarks[24].y,  # hips
            landmarks[27].y, landmarks[28].y   # ankles
        ]
        if max(relevant_y) - min(relevant_y) < 0.1:
            horizontal = True

        # For Shoulder Press: compute angle between right arm and upward vertical vector.
        arm_vector = (right_wrist[0] - right_shoulder[0], right_wrist[1] - right_shoulder[1])
        vertical_vector = (0, -1)
        angle_with_vertical = angle_between(arm_vector, vertical_vector)

        # --- Determine the exercise using computed angles ---
        if horizontal:
            # Horizontal body posture
            if (left_elbow_angle is not None and left_elbow_angle < 90) or \
               (right_elbow_angle is not None and right_elbow_angle < 90):
                detected_exercise = "Push-Up"
            elif (left_elbow_angle is not None and left_elbow_angle > 160) and \
                 (right_elbow_angle is not None and right_elbow_angle > 160):
                detected_exercise = "Plank"
        else:
            # For vertical postures
            # Shoulder Press: right wrist above nose, arm nearly vertical, and right elbow nearly straight.
            if (right_wrist[1] < nose[1]) and (angle_with_vertical < 30) and (right_elbow_angle > 160):
                detected_exercise = "Shoulder Press"
            # Squat: both knees bent sharply (knee angles below about 90°)
            elif left_knee_angle < 90 and right_knee_angle < 90:
                detected_exercise = "Squat"
            # Lunge: one knee is bent (angle < 90°) and the other is nearly straight (angle > 160°)
            elif (left_knee_angle < 90 and right_knee_angle > 160) or \
                 (right_knee_angle < 90 and left_knee_angle > 160):
                detected_exercise = "Lunge"
            else:
                detected_exercise = "Unknown"

        # Draw pose landmarks on the main video frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        # Draw stick man skeleton on the separate image
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            start_point = (int(start.x * image_width), int(start.y * image_height))
            end_point = (int(end.x * image_width), int(end.y * image_height))
            cv2.line(stick_man, start_point, end_point, (255, 255, 255), 2)
        for lm in landmarks:
            cx = int(lm.x * image_width)
            cy = int(lm.y * image_height)
            cv2.circle(stick_man, (cx, cy), 4, (0, 255, 0), -1)

    # Compare the detected exercise with the selected exercise.
    if detected_exercise == selected_exercise:
        status_text = "Good Posture"
        status_color = (0, 255, 0)
    else:
        status_text = "Wrong Exercise"
        status_color = (0, 0, 255)
        if not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm, daemon=True).start()

    # Overlay text information on the main frame.
    cv2.putText(frame, f"Selected: {selected_exercise}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Detected: {detected_exercise}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, status_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    if left_elbow_angle is not None:
        cv2.putText(frame, f"Left Elbow Angle: {int(left_elbow_angle)}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if right_elbow_angle is not None:
        cv2.putText(frame, f"Right Elbow Angle: {int(right_elbow_angle)}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if left_knee_angle is not None:
        cv2.putText(frame, f"Left Knee Angle: {int(left_knee_angle)}", (10, 210),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if right_knee_angle is not None:
        cv2.putText(frame, f"Right Knee Angle: {int(right_knee_angle)}", (10, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    if angle_with_vertical is not None:
        cv2.putText(frame, f"Arm-Vertical Angle: {int(angle_with_vertical)}", (10, 270),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Show both windows
    cv2.imshow("Exercise Detection", frame)
    cv2.imshow("Stick Man", stick_man)

    # Check for key presses: 'q' to quit, or keys 1-5 to select an exercise.
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key in exercise_map:
        selected_exercise = exercise_map[key]
        alarm_on = False  # Reset the alarm flag when the exercise selection changes

cap.release()
cv2.destroyAllWindows()
pose.close()
