#!/usr/bin/env python

import os
import cv2
import rosbag
import rospy
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from collections import deque


# "/mnt/hdd2/yizhouhu/bags/lego_fish_high_2025-06-12-16-17-24.bag"

BAG_PATH = "/mnt/hdd2/yizhouhu/bags/lego_cliff_2025-06-16-17-13-46.bag"  # path to your bag file
TASK = "cliff"
depth = True

TOPIC1 = "/cam_destroyer/color/image_raw/compressed"
TOPIC2 = "/cam_architect/color/image_raw/compressed"
TOPIC1_DEPTH = "/cam_destroyer/depth/image_raw/compressedDepth"
TOPIC2_DEPTH = "/cam_architect/depth/image_raw/compressedDepth"
TOPIC_STEP = "/lego/assembling_step"
OUTPUT_DIR1 = f"outputs/cam1/{TASK}" # output directory for cam1
OUTPUT_DIR2 = f"outputs/cam2/{TASK}" 
OUTPUT_DIR_STEPS = f"outputs/assembling_step" 
OUTPUT_FREQ = 1  # Hz (i.e. one image every 1 seconds)
SYNC_TOLERANCE = rospy.Duration(0.05)  # seconds

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def msg_to_cv2(msg):
    np_arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def depth_to_cv2(msg, bridge):
    np_arr = bridge.imgmsg_to_cv2(msg, "32FC1")
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def compressed_depth_to_cv2(msg):
    # 'msg' as type CompressedImage
    depth_fmt, compr_type = msg.format.split(';')
    
    # remove white space
    depth_fmt = depth_fmt.strip()
    compr_type = compr_type.strip()
    if 'compressedDepth' not in compr_type:
        raise Exception("Compression type is not 'compressedDepth'."
                        "You probably subscribed to the wrong topic.")

    # remove header from raw data
    depth_header_size = 12
    raw_data = msg.data[depth_header_size:]

    depth_img_raw = cv2.imdecode(np.fromstring(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
    if depth_img_raw is None:
        raise Exception("Could not decode compressed depth image."
                        "You may need to change 'depth_header_size'!")

    if depth_fmt == "16UC1":
       return depth_img_raw
    else:
        raise Exception("Decoding of '" + depth_fmt + "' is not implemented!")

def main():
    rospy.init_node("image_extractor", anonymous=True)
    bridge = CvBridge()

    ensure_dir(OUTPUT_DIR1)
    ensure_dir(OUTPUT_DIR2)
    ensure_dir(OUTPUT_DIR_STEPS)

    bag = rosbag.Bag(BAG_PATH, "r")

    buffer1 = deque()
    buffer2 = deque()
    depth_buffer1 = deque()
    depth_buffer2 = deque()
    step_buffer = deque()
    buffers = [buffer1, buffer2, depth_buffer1, depth_buffer2, step_buffer]

    last_saved_time = None
    img_count = 0

    print("Reading bag...")

    if depth:
        topics = [TOPIC1, TOPIC2, TOPIC1_DEPTH, TOPIC2_DEPTH, TOPIC_STEP]
    else:
        topics = [TOPIC1, TOPIC2]

    for topic, msg, t in bag.read_messages(topics=topics):
        if topic == TOPIC1:
            buffer1.append((t, msg))
        elif topic == TOPIC2:
            buffer2.append((t, msg))
        elif topic == TOPIC1_DEPTH:
            depth_buffer1.append((t, msg))
        elif topic == TOPIC2_DEPTH:
            depth_buffer2.append((t, msg))
        elif topic == TOPIC_STEP:
            step_buffer.append((t, msg))
            
        # Try to synchronize messages
        while buffer1 and buffer2 and (not depth or (depth_buffer1 and depth_buffer2)) and step_buffer:
            t1, m1 = buffer1[0]
            t2, m2 = buffer2[0]
            d_t1, d_m1 = depth_buffer1[0]
            d_t2, d_m2 = depth_buffer2[0]
            s_t, s_m = step_buffer[0]

            times = [t1, t2, d_t1, d_t2, s_t]
            dt = max([abs((ta - tb).to_sec()) for ta in times for tb in times])

            if dt < SYNC_TOLERANCE.to_sec():
                if last_saved_time is None or (t1 - last_saved_time).to_sec() >= 1.0 / OUTPUT_FREQ:
                    # Synchronization OK
                    img1 = msg_to_cv2(m1)
                    img2 = msg_to_cv2(m2)
                    depth_img1 = compressed_depth_to_cv2(d_m1)
                    depth_img2 = compressed_depth_to_cv2(d_m2)

                    filename = f"{img_count:06d}.jpg"
                    cv2.imwrite(os.path.join(OUTPUT_DIR1, filename), img1)
                    cv2.imwrite(os.path.join(OUTPUT_DIR2, filename), img2)
                    
                    depth_filename = f"{img_count:06d}_depth"
                    cv_depth_cam1 = np.float16(np.nan_to_num(depth_img1))
                    cv_depth_cam2 = np.float16(np.nan_to_num(depth_img2))
                    np.savez_compressed(os.path.join(OUTPUT_DIR1, depth_filename), cv_depth_cam1)
                    np.savez_compressed(os.path.join(OUTPUT_DIR2, depth_filename), cv_depth_cam2)

                    with open(OUTPUT_DIR_STEPS + f'/{TASK}.txt', 'a') as f:
                        f.write(str(s_m.data) + '\n')

                    print(f"[{img_count}] Saved synchronized pair at {t1.to_sec():.2f}s")

                    img_count += 1
                    last_saved_time = t1

                buffer1.popleft()
                buffer2.popleft()
                depth_buffer1.popleft()
                depth_buffer2.popleft()
                step_buffer.popleft()

            else:
                min_index = min(range(len(times)), key=lambda i: times[i])
                earliest_buffer = buffers[min_index]
                earliest_buffer.popleft()

    bag.close()
    print("Done.")

if __name__ == "__main__":
    main()
