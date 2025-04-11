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
# cap = cv2.VideoCapture("wrist.mp4")
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
#             # Get key points for wrist angle
#             elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
#                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
#             wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
#                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
#             index_finger = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
#                             landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]
#
#             # Calculate wrist angle (elbow as vertex)
#             wrist_angle = calculate_angle(elbow, wrist, index_finger)
#
#             # Display angles on screen
#             cv2.putText(image, f"Wrist Angle: {int(wrist_angle)}°",
#                         tuple(np.multiply(wrist, [640, 480]).astype(int)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
#
#             print(f"Wrist Angle: {wrist_angle}°")
#
#         except:
#             pass
#
#         # Draw skeleton
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
#
#         cv2.imshow('Mediapipe Pose Estimation', image)
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

input_vid = "wrist.mp4"
output_vid = "wrist_mediapipe.mp4"

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

            # Get key points for wrist angle
            elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            index_finger = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y]

            # Calculate wrist angle (elbow as vertex)
            wrist_angle = calculate_angle(elbow, wrist, index_finger)

            # Display angles on screen
            cv2.putText(image, f"Wrist Angle: {int(wrist_angle)}°",
                        tuple(np.multiply(wrist, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            print(f"Wrist Angle: {wrist_angle}°")

        except Exception as e:
            print(f"Error: {e}")

        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Write frame to video
        out.write(image)

        cv2.imshow('Mediapipe Pose Estimation - Wrist Angle', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
