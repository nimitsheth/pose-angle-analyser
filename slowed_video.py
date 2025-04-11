import cv2

input_path = "neck.mp4"
output_path = "neck_slowed.mp4"
slow_factor = 2  # 2x slower

cap = cv2.VideoCapture(input_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# New FPS is half, same number of frames, just slower playback
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps / slow_factor, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Write each frame multiple times (for slower playback)
    for _ in range(slow_factor):
        out.write(frame)

cap.release()
out.release()
print("✅ Slowed-down video saved!")
