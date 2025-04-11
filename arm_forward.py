import cv2
import mediapipe as mp
import numpy as np
import csv

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c) in degrees."""
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (vertex)
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    # angle = np.abs(radians * 180.0 / np.pi)
    angle = -(radians * 180.0 / np.pi)

    return angle

cap = cv2.VideoCapture("arm_forward.mp4")
output_vid = "arm_forward_mediapipe.mp4"

# Get frame width, height, and FPS
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid, fourcc, fps, (frame_width, frame_height))

angles = ['Shoulder Angles']
frame_count = 0
second_count = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert color format
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            # Get key points
            shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

            # Calculate shoulder angle (front movement)
            shoulder_forward_angle = calculate_angle(hip, shoulder, wrist)  # Hip → Shoulder → Wrist

            if frame_count >= fps:
                angles.append(shoulder_forward_angle)
                second_count += 1
                frame_count = 0

            # Display angles on screen
            cv2.putText(image, f"Shoulder Forward: {int(shoulder_forward_angle)}°",
                        tuple(np.multiply(shoulder, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            print(f"Shoulder Forward Angle: {shoulder_forward_angle}°")

        except:
            pass

        frame_count += 1




        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        out.write(image)  # Save frame to output video
        cv2.imshow('Mediapipe Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
with open('angles.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(angles)
cv2.destroyAllWindows()
