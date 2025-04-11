import cv2
import torch
import posenet
import tensorflow as tf
import matplotlib.pyplot as plt

import torch
from posenet.constants import *
from posenet.decode_multi import decode_multiple_poses
from posenet.models.model_factory import load_model
from posenet.utils import *

net = load_model(101)
net = net.cuda()
output_stride = net.output_stride
scale_factor = 1.0


def posenet_model(file):
	input_image, draw_image, output_scale = posenet.read_imgfile(file, scale_factor=scale_factor,
																 output_stride=output_stride)
	with torch.no_grad():
		input_image = torch.Tensor(input_image).cuda()

		heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = net(input_image)

		pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
			heatmaps_result.squeeze(0),
			offsets_result.squeeze(0),
			displacement_fwd_result.squeeze(0),
			displacement_bwd_result.squeeze(0),
			output_stride=output_stride,
			max_pose_detections=10,
			min_pose_score=0.25)

		# Find keypoints on the image
		image = plt.imread(file)
		poses = []

		for pi in range(len(pose_scores)):
			if pose_scores[pi] != 0.:
				print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
				keypoints = keypoint_coords.astype(np.int32)
				print(keypoints[pi])
				poses.append(keypoints[pi])

		# Show keypoints on the image
		img = plt.imread(file)
		i = 0
		pose = poses[0]
		plt.imshow(img)
		for y, x in pose:
			plt.plot(x, y, 'w.')
			plt.text(x, y, str(i), color='r', fontsize=10)
			i += 1
		plt.show()

file = 'test.png'

import matplotlib.pyplot as plt
img = plt.imread(file)
plt.imshow(img)
plt.show()