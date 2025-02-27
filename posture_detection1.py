import cv2
import mediapipe as mp
import numpy as np
import math
from playsound import playsound
import threading

# Play the alarm sound in a separate thread.
def play_alarm():
    playsound("alarm.wav")  # Ensure "alarm.wav" exists in your working directory.

# Calculate the angle between three points (in degrees).
def calculate_angle(a, b, c):
    # a, b, c are (x,y) tuples; b is the vertex.
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians * 180.0 / math.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

# Compute the angle between two vectors (in degrees).
def angle_between(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    cos_theta = dot / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_theta))
    return angle

# Map keys 1-5 to exercise names.
exercise_map = {
    ord('1'): "Hand Raise",
    ord('2'): "Bicep Curl",
    ord('3'): "Lateral Raise",
    ord('4'): "Front Raise",
    ord('5'): "Arm Extension"
}
selected_exercise = "Hand Raise"  # Default exercise.

# Initialize Mediapipe Pose.
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)
alarm_on = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirror view and convert to RGB.
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_height, image_width, _ = frame.shape

    # Process the image to detect pose landmarks.
    results = pose.process(frame_rgb)

    # Create a black image for drawing the stick man.
    stick_man = np.zeros_like(frame)

    detected_exercise = "Unknown"
    elbow_angle = None
    angle_with_vertical = None
    angle_with_horizontal = None

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Extract key landmarks (normalized coordinates):
        # Nose: landmark 0
        nose = (landmarks[0].x, landmarks[0].y)
        # Right Shoulder: landmark 12
        right_shoulder = (landmarks[12].x, landmarks[12].y)
        # Right Elbow: landmark 14
        right_elbow = (landmarks[14].x, landmarks[14].y)
        # Right Wrist: landmark 16
        right_wrist = (landmarks[16].x, landmarks[16].y)

        # Compute the elbow angle (for Bicep Curl and Arm Extension).
        elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        # Compute arm vector from shoulder to wrist.
        arm_vector = (right_wrist[0] - right_shoulder[0], right_wrist[1] - right_shoulder[1])
        
        # For Hand Raise: compute angle between arm vector and upward vertical vector.
        vertical_vector = (0, -1)
        angle_with_vertical = angle_between(arm_vector, vertical_vector)
        
        # For Lateral Raise: compute angle between arm vector and horizontal vector.
        horizontal_vector = (1, 0)
        angle_with_horizontal = angle_between(arm_vector, horizontal_vector)

        # --- Define exercise conditions using computed angles ---
        # 1. Hand Raise: arm nearly vertical (angle with vertical < 30°) and wrist above nose.
        if angle_with_vertical < 30 and right_wrist[1] < nose[1]:
            detected_exercise = "Hand Raise"
        # 2. Bicep Curl: elbow is strongly flexed (elbow angle < 50°).
        elif elbow_angle < 50:
            detected_exercise = "Bicep Curl"
        # 3. Lateral Raise: arm is nearly horizontal (angle with horizontal < 30° or >150°).
        elif angle_with_horizontal < 30 or angle_with_horizontal > 150:
            detected_exercise = "Lateral Raise"
        # 4. Front Raise: wrist is raised above the shoulder with minimal horizontal displacement.
        elif abs(right_wrist[0] - right_shoulder[0]) < 0.08 and right_wrist[1] < right_shoulder[1]:
            detected_exercise = "Front Raise"
        # 5. Arm Extension: arm is nearly straight (elbow angle > 160°).
        elif elbow_angle > 160:
            detected_exercise = "Arm Extension"
        else:
            detected_exercise = "Unknown"

        # Draw pose landmarks on the main frame.
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Draw the stick man on the separate image.
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

    # Compare detected exercise with the selected one.
    if detected_exercise == selected_exercise:
        status_text = "Good Posture"
        status_color = (0, 255, 0)
    else:
        status_text = "Wrong Exercise"
        status_color = (0, 0, 255)
        if not alarm_on:
            alarm_on = True
            threading.Thread(target=play_alarm, daemon=True).start()

    # Overlay selected exercise, detected exercise, and status on the main frame.
    cv2.putText(frame, f"Selected: {selected_exercise}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, f"Detected: {detected_exercise}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(frame, status_text, (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    if elbow_angle is not None:
        cv2.putText(frame, f"Elbow Angle: {int(elbow_angle)}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    if angle_with_vertical is not None:
        cv2.putText(frame, f"Vertical Diff: {int(angle_with_vertical)}", (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    if angle_with_horizontal is not None:
        cv2.putText(frame, f"Horizontal Diff: {int(angle_with_horizontal)}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)

    # Show the two windows.
    cv2.imshow("Exercise Detection", frame)
    cv2.imshow("Stick Man", stick_man)

    # Check for key presses: 'q' to quit or keys 1-5 to select an exercise.
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key in exercise_map:
        selected_exercise = exercise_map[key]
        alarm_on = False  # Reset alarm flag when switching exercises.

cap.release()
cv2.destroyAllWindows()
pose.close()
