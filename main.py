# import cv2
# import mediapipe as mp
# import numpy as np
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
# def calculate_angle(a, b, c):
# 	a = np.array(a)  # First
# 	b = np.array(b)  # Mid
# 	c = np.array(c)  # End
#
# 	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
# 	angle = np.abs(radians * 180.0 / np.pi)
#
# 	if angle > 180.0:
# 		angle = 360 - angle
#
# 	return angle
#
# cap = cv2.VideoCapture(0)
# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# 	while cap.isOpened():
# 		ret, frame = cap.read()
#
# 		# Recolor image to RGB
# 		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		image.flags.writeable = False
#
# 		# Make detection
# 		results = pose.process(image)
#
# 		# Recolor back to BGR
# 		image.flags.writeable = True
# 		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
# 		# Extract landmarks
# 		try:
# 			landmarks = results.pose_landmarks.landmark
#
# 			# Get coordinates
# 			shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
# 						landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# 			elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
# 					 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# 			wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
# 					 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#
# 			# Calculate angle
# 			angle = calculate_angle(shoulder, elbow, wrist)
#
# 			# Visualize angle
# 			cv2.putText(image, str(angle),
# 						tuple(np.multiply(elbow, [640, 480]).astype(int)),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
# 						)
#
# 		except:
# 			pass
#
# 		# Render detections
# 		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
# 								  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
# 								  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
# 								  )
#
# 		cv2.imshow('Mediapipe Feed', image)
#
# 		shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
# 					landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# 		elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# 		wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
#
# 		print(calculate_angle(shoulder, elbow, wrist))
#
# 		if cv2.waitKey(10) & 0xFF == ord('q'):
# 			break
#
# 	cap.release()
# 	cv2.destroyAllWindows()
#
#
#
# shoulder, elbow, wrist
#
# print(calculate_angle(shoulder, elbow, wrist))
#
# tuple(np.multiply(elbow, [640, 480]).astype(int))
#

#
# import cv2
# import mediapipe as mp
# import numpy as np
#
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose
#
#
# def calculate_angle(a, b, c):
# 	"""Calculate angle between three points (a-b-c) in degrees."""
# 	a = np.array(a)  # First point
# 	b = np.array(b)  # Middle point (vertex)
# 	c = np.array(c)  # Last point
#
# 	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
# 	angle = np.abs(radians * 180.0 / np.pi)
#
# 	if angle > 180.0:
# 		angle = 360 - angle
#
# 	return angle
#
#
# cap = cv2.VideoCapture(0)
#
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
# 	while cap.isOpened():
# 		ret, frame = cap.read()
#
# 		# Convert color format
# 		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# 		image.flags.writeable = False
# 		results = pose.process(image)
#
# 		# Convert back to BGR
# 		image.flags.writeable = True
# 		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
# 		try:
# 			landmarks = results.pose_landmarks.landmark
#
# 			# Get key points
# 			shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
# 						landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
# 			elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
# 					 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
# 			wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
# 					 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
# 			hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
# 				   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
#
# 			# Calculate elbow angle (existing logic)
# 			elbow_angle = calculate_angle(shoulder, elbow, wrist)
#
# 			# Calculate shoulder angle with respect to body
# 			shoulder_angle = calculate_angle(hip, shoulder, elbow)  # Hip → Shoulder → Elbow
#
# 			# Display angles on screen
# 			cv2.putText(image, f"Elbow: {int(elbow_angle)}°",
# 						tuple(np.multiply(elbow, [640, 480]).astype(int)),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
#
# 			cv2.putText(image, f"Shoulder: {int(shoulder_angle)}°",
# 						tuple(np.multiply(shoulder, [640, 480]).astype(int)),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
#
# 			print(f"Elbow Angle: {elbow_angle}° | Shoulder Angle: {shoulder_angle}°")
#
# 		except:
# 			pass
#
# 		# Draw skeleton
# 		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
# 								  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
# 								  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
#
# 		cv2.imshow('Mediapipe Pose Estimation', image)
#
# 		if cv2.waitKey(10) & 0xFF == ord('q'):
# 			break
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

cap = cv2.VideoCapture("neck.mp4")

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

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
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                   landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

            # Calculate shoulder angle (front movement)
            shoulder_forward_angle = calculate_angle(hip, shoulder, wrist)  # Hip → Shoulder → Wrist

            # Display angles on screen
            cv2.putText(image, f"Shoulder Forward: {int(shoulder_forward_angle)}°",
                        tuple(np.multiply(shoulder, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

            print(f"Shoulder Forward Angle: {shoulder_forward_angle}°")

        except:
            pass

        # Draw skeleton
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Pose Estimation', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
