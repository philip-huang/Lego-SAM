#!/usr/bin/env python

import os
import cv2
import rosbag
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from collections import deque

BAG_PATH = "/home/mfi/repos/ros1_ws/src/philip/data/lego_R_2025-05-25-19-41-14.bag"  # path to your bag file
TOPIC1 = "/cam_destroyer/color/image_raw/compressed"
TOPIC2 = "/cam_architect/color/image_raw/compressed"
OUTPUT_DIR1 = "outputs/cam1/R"
OUTPUT_DIR2 = "outputs/cam2/R"
OUTPUT_FREQ = 0.5  # Hz (i.e. one image every 2 seconds)
SYNC_TOLERANCE = rospy.Duration(0.05)  # seconds

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def msg_to_cv2(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def main():
    rospy.init_node("image_extractor", anonymous=True)
    bridge = CvBridge()

    ensure_dir(OUTPUT_DIR1)
    ensure_dir(OUTPUT_DIR2)

    bag = rosbag.Bag(BAG_PATH, "r")

    buffer1 = deque()
    buffer2 = deque()
    last_saved_time = None
    img_count = 0

    print("Reading bag...")

    for topic, msg, t in bag.read_messages(topics=[TOPIC1, TOPIC2]):
        if topic == TOPIC1:
            buffer1.append((t, msg))
        elif topic == TOPIC2:
            buffer2.append((t, msg))

        # Try to synchronize messages
        while buffer1 and buffer2:
            t1, m1 = buffer1[0]
            t2, m2 = buffer2[0]
            dt = abs((t1 - t2).to_sec())

            if dt < SYNC_TOLERANCE.to_sec():
                if last_saved_time is None or (t1 - last_saved_time).to_sec() >= 1.0 / OUTPUT_FREQ:
                    # Synchronization OK
                    img1 = msg_to_cv2(m1)
                    img2 = msg_to_cv2(m2)

                    filename = f"{img_count:06d}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_DIR1, filename), img1)
                    cv2.imwrite(os.path.join(OUTPUT_DIR2, filename), img2)
                    print(f"[{img_count}] Saved synchronized pair at {t1.to_sec():.2f}s")

                    img_count += 1
                    last_saved_time = t1

                buffer1.popleft()
                buffer2.popleft()

            elif t1 < t2:
                buffer1.popleft()
            else:
                buffer2.popleft()

    bag.close()
    print("Done.")

if __name__ == "__main__":
    main()
