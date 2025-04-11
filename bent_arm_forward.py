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
# cap = cv2.VideoCapture("bent_arm_forward.mp4")
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
# 			elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
# 						landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
# 			wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
# 					 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
# 			hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
# 				   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
#
# 			# Calculate shoulder angle (front movement)
# 			shoulder_forward_angle = calculate_angle(hip, elbow, wrist)  # Hip → Shoulder → Wrist
#
# 			# Display angles on screen
# 			cv2.putText(image, f"Shoulder Forward: {int(shoulder_forward_angle)}°",
# 						tuple(np.multiply(elbow, [640, 480]).astype(int)),
# 						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
#
# 			print(f"Shoulder Forward Angle: {shoulder_forward_angle}°")
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
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)

	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
	angle = np.abs(radians * 180.0 / np.pi)

	if angle > 180.0:
		angle = 360 - angle

	return angle


cap = cv2.VideoCapture("bent_arm_forward.mp4")

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define output video file name
output_vid = 'bent_arm_forward_mediapipe.mp4'

# Define the codec and create VideoWriter object
out = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, (frame_width, frame_height))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		image.flags.writeable = False
		results = pose.process(image)

		image.flags.writeable = True
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		try:
			landmarks = results.pose_landmarks.landmark
			elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
					 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
			wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
					 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
			hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
				   landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]

			shoulder_forward_angle = calculate_angle(hip, elbow, wrist)

			cv2.putText(image, f"Shoulder Forward: {int(shoulder_forward_angle)}°",
						tuple(np.multiply(elbow, [frame_width, frame_height]).astype(int)),
						cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

			print(f"Shoulder Forward Angle: {shoulder_forward_angle}°")
		except:
			pass

		mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
								  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
								  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

		out.write(image)
		cv2.imshow('Mediapipe Pose Estimation', image)

		if cv2.waitKey(10) & 0xFF == ord('q'):
			break

cap.release()
out.release()
cv2.destroyAllWindows()
