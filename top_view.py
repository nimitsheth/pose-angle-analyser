# import cv2
# import mediapipe as mp
# import numpy as np
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# def calculate_angle(a, b, c):
#     """Calculate angle between three points (a-b-c) in degrees."""
#     a = np.array(a)  # First point
#     b = np.array(b)  # Middle point (vertex)
#     c = np.array(c)  # Last point
#
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
#
#     if angle > 180.0:
#         angle = 360 - angle
#
#     return angle
#
# cap = cv2.VideoCapture("top_view.mp4")
#
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Convert color format
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#         results = pose.process(image)
#
#         # Convert back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         try:
#             landmarks = results.pose_landmarks.landmark
#
#             # Get key points for estimation
#             left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
#                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
#                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
#
#             # Midpoint of shoulders (this acts as our vertex)
#             shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2,
#                             (left_shoulder[1] + right_shoulder[1]) / 2]
#
#             # Estimate head position by extending upwards from shoulder_mid
#             head_top = [shoulder_mid[0], shoulder_mid[1] - 0.1]  # Adjusting upward by an estimated value
#
#             # Calculate head tilt angle
#             head_angle = calculate_angle(left_shoulder, shoulder_mid, head_top)
#
#             # Display angle on screen
#             cv2.putText(image, f"Head Rotation: {int(head_angle)}°",
#                         tuple(np.multiply(shoulder_mid, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
#
#             print(f"Head Rotation Angle: {head_angle}°")
#
#         except Exception as e:
#             print(f"Error: {e}")
#
#         # Draw skeleton
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
#
#         cv2.imshow('Mediapipe Pose Estimation - Top View', image)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
# cap.release()
# cv2.destroyAllWindows()


import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate angle between three points (a-b-c) in degrees."""
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point (vertex)
    c = np.array(c)  # Last point

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

input_vid = "top_view.mp4"
output_vid = "top_view_mediapipe.mp4"

cap = cv2.VideoCapture(input_vid)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid, fourcc, frame_fps, (frame_width, frame_height))

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

            # Get key points for estimation
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

            # Midpoint of shoulders (this acts as our vertex)
            shoulder_mid = [(left_shoulder[0] + right_shoulder[0]) / 2,
                            (left_shoulder[1] + right_shoulder[1]) / 2]

            # Estimate head position by extending upwards from shoulder_mid
            head_top = [shoulder_mid[0], shoulder_mid[1] - 0.1]  # Adjusting upward by an estimated value

            # Calculate head tilt angle
            head_angle = calculate_angle(left_shoulder, shoulder_mid, head_top)

            # Display angle on screen
            cv2.putText(image, f"Head Rotation: {int(head_angle)}°",
                        tuple(np.multiply(shoulder_mid, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            print(f"Head Rotation Angle: {head_angle}°")

        except Exception as e:
            print(f"Error: {e}")

        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Write frame to video
        out.write(image)

        cv2.imshow('Mediapipe Pose Estimation - Top View', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
