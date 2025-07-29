import cv2
import os

input_path = "/sample-videos/qsr-usecase.mp4"
output_dir = "/sample-videos/increased-video"
output_path = os.path.join(output_dir, "qsr-usecase.mp4")

# Skip processing if file already exists
if os.path.exists(output_path):
    print(f"Compressed 5-minute video already exists: {output_path}")
else:
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to 1280x720
        resized = cv2.resize(frame, (1280, 720))
        frames.append(resized)

    cap.release()

    frame_count = len(frames)
    duration = frame_count / fps
    loop_count = int(300 // duration)  # 300 seconds = 5 minutes

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(loop_count):
        for frame in frames:
            out.write(frame)

    out.release()
    print(f"Raw 5-minute video saved to: {output_path}")
