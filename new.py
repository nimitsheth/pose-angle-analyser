import cv2
import numpy as np
import torch
from posenet import posenet

def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) between three points (a, b, c) with b as the vertex.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# Choose your model; 101 or 75 (101 gives higher accuracy but is slower)
model_id = 101

# Load the model (this will download weights if not already cached)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = posenet.load_model(model_id, device=device)
model = model.eval()

# Output stride and input scale are model-dependent
output_stride = posenet.get_output_stride(model)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # PoseNet expects a BGR image in numpy format
    input_image, display_image, output_scale = posenet.process_input(frame, scale_factor=1.0, output_stride=output_stride)
    input_image = torch.Tensor(input_image).to(device)

    with torch.no_grad():
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

    # Decode the poses (this example decodes the single highest-scoring pose)
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
        heatmaps_result.squeeze(0),
        offsets_result.squeeze(0),
        displacement_fwd_result.squeeze(0),
        displacement_bwd_result.squeeze(0),
        output_stride=output_stride,
        max_pose_detections=1,
        min_pose_score=0.2
    )

    # Scale keypoint coordinates to the original image size
    keypoint_coords *= output_scale

    if keypoint_coords.shape[0] > 0:
        keypoints = keypoint_coords[0]  # First (and highest scoring) pose
        # PoseNet keypoint order (for COCO):
        # 0: nose, 1: left eye, 2: right eye, 3: left ear, 4: right ear,
        # 5: left shoulder, 6: right shoulder, 7: left elbow, 8: right elbow,
        # 9: left wrist, 10: right wrist, 11: left hip, 12: right hip,
        # 13: left knee, 14: right knee, 15: left ankle, 16: right ankle
        left_shoulder = keypoints[5]
        left_wrist = keypoints[9]
        left_hip = keypoints[11]

        angle = calculate_angle(left_hip, left_shoulder, left_wrist)
        cv2.putText(display_image, f"Shoulder Forward: {int(angle)}°",
                    (int(left_shoulder[0]), int(left_shoulder[1])), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        print("Shoulder Forward Angle:", angle)

    cv2.imshow("PoseNet", display_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
