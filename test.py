import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


def calculate_angle(a, b, c):
	"""
	Calculate the angle (in degrees) between three points (a, b, c) with 'b' as the vertex.
	"""
	a = np.array(a)
	b = np.array(b)
	c = np.array(c)
	radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
	angle = np.abs(radians * 180.0 / np.pi)
	if angle > 180.0:
		angle = 360 - angle
	return angle


# Configure Detectron2 for keypoint detection
cfg = get_cfg()
# Load the keypoint R-CNN configuration from the model zoo (COCO dataset)
cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set detection threshold
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")

# Create the predictor
predictor = DefaultPredictor(cfg)

# Open video capture (webcam)
cap = cv2.VideoCapture(0)

while cap.isOpened():
	ret, frame = cap.read()
	if not ret:
		break

	# Run inference
	outputs = predictor(frame)
	instances = outputs["instances"]

	if instances.has("pred_keypoints"):
		# Process the first detected person (if multiple persons, you can loop over all)
		keypoints = instances.pred_keypoints[0].cpu().numpy()  # shape: (17, 3)
		# Keypoints order for COCO:
		# 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear,
		# 5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow,
		# 9: left wrist, 10: right wrist, 11: left hip, 12: right hip, etc.

		# Extract keypoints for left shoulder, left wrist, and left hip
		left_shoulder = keypoints[5][:2]
		left_wrist = keypoints[9][:2]
		left_hip = keypoints[11][:2]

		# Calculate the shoulder forward angle: hip -> shoulder -> wrist
		angle = calculate_angle(left_hip, left_shoulder, left_wrist)

		# Annotate the image with the computed angle
		cv2.putText(frame, f"Shoulder Forward: {int(angle)}°",
					(int(left_shoulder[0]), int(left_shoulder[1])),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
		print("Shoulder Forward Angle:", angle)

	# Display the frame with annotations
	cv2.imshow("Detectron2 Pose Estimation", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
