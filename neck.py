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
    angle = np.abs(radians * 180.0 / np.pi)

    # if angle > 180.0:
    #     angle = 360 - angle

    return angle

input_vid = "neck.mp4"
output_vid = "neck_mediapipe.mp4"

cap = cv2.VideoCapture(input_vid)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_vid, fourcc, frame_fps, (frame_width, frame_height))

angles = ['Neck Angles Mediapipe']
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

            # Get key points for neck angle (side view)
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            ear = [landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # Calculate neck angle (shoulder as vertex)
            neck_angle = calculate_angle(shoulder, ear, nose)

            frame_count += 1
            if frame_count >= frame_fps:
                angles.append(neck_angle)
                second_count += 1
                frame_count = 0

            # Display angles on screen
            cv2.putText(image, f"Neck Angle: {int(neck_angle)}",
                        tuple(np.multiply(ear, [frame_width, frame_height]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)

            print(f"Neck Angle: {neck_angle}°")

        except:
            pass

        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Write frame to video
        out.write(image)

        cv2.imshow('Mediapipe Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
out.release()
with open('angles.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(angles)
cv2.destroyAllWindows()


# import cv2
# import mediapipe as mp
# import numpy as np
# import csv
#
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
#                                    min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# def calculate_angle(a, b, c):
#     """
#     Calculate the angle between three points a, b, and c (with b as the vertex).
#     """
#     a = np.array(a)
#     b = np.array(b)
#     c = np.array(c)
#
#     ba = a - b
#     bc = c - b
#
#     cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
#     angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # Clip for numerical stability
#     return np.degrees(angle)
#
# # Setup video
# input_vid = "neck_slowed.mp4"
# output_vid = "neck_facemesh.mp4"
# cap = cv2.VideoCapture(input_vid)
#
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter(output_vid, fourcc, fps, (frame_width, frame_height))
#
# # Tracking
# angles = ['Neck Angle (FaceMesh)']
# frame_count = 0
#
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     # Convert to RGB
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb_frame)
#
#     if results.multi_face_landmarks:
#         landmarks = results.multi_face_landmarks[0].landmark
#
#         # Get 2D coordinates of key landmarks
#         chin = landmarks[152]
#         nose = landmarks[1]
#         jaw = landmarks[234]  # Side jaw point near ear
#
#         chin_pt = [chin.x * frame_width, chin.y * frame_height]
#         nose_pt = [nose.x * frame_width, nose.y * frame_height]
#         jaw_pt = [jaw.x * frame_width, jaw.y * frame_height]
#
#         # Calculate neck angle
#         angle = calculate_angle(jaw_pt, chin_pt, nose_pt)
#
#         # Increment counter and store angle every second
#         frame_count += 1
#         if frame_count >= fps:
#             angles.append(int(angle))
#             frame_count = 0
#
#         # Display on frame
#         cv2.putText(frame, f"Neck Angle: {int(angle)}°", (50, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#
#         # Draw key points
#         for pt in [chin_pt, nose_pt, jaw_pt]:
#             cv2.circle(frame, tuple(np.int32(pt)), 3, (0, 0, 255), -1)
#
#     out.write(frame)
#     cv2.imshow("FaceMesh Neck Angle", frame)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         break
#
# # Cleanup
# cap.release()
# out.release()
# cv2.destroyAllWindows()
#
# # Save to CSV
# with open("neck_angles_facemesh.csv", "a", newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(angles)
#
# print("CSV saved.")
