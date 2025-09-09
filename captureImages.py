import pyrealsense2 as rs
import cv2, os
import numpy as np
#-------Capture 15-20 different images of checkerboard at different angles,distance,position-------------------#
pipeline = rs.pipeline()
pipeline.start()

output_folder = "calib_images"
os.makedirs(output_folder, exist_ok=True)

print("Press 's' to save a frame. Press ESC to exit.")
frame_id = 0

while True:
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    color_image = np.asanyarray(color_frame.get_data())
    cv2.imshow("Calibration View", color_image)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == ord('s'):
        filename = os.path.join(output_folder, f"calib_{frame_id}.jpg")
        cv2.imwrite(filename, color_image)
        print(f"Saved {filename}")
        frame_id += 1

pipeline.stop()
cv2.destroyAllWindows()
