#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np

import argparse
import os
from datetime import datetime

import lego_online_infer
from lego_visualize import visualize

import shutil
import os
import json

class SideviewCalibrateNode:
    def __init__(self, task):
        self.task = task
        self.bridge = CvBridge()

        self.save_dir = '/home/mfi/repos/ros1_ws/src/philip/Lego-SAM/crop_calibration'
        
        self.cam1_color = None
        self.cam2_color = None
        self.cam1_depth = None
        self.cam2_depth = None

        self.image_dir = '/home/mfi/repos/ros1_ws/src/philip/Lego-SAM/crop_calibration/temp/'
        if os.path.exists(self.image_dir):
            shutil.rmtree(self.image_dir)
        os.makedirs(self.image_dir, exist_ok=True)
        date = datetime.now().strftime("%m_%d_%Y")
        self.json_dir = f'/home/mfi/repos/ros1_ws/src/philip/Lego-SAM/crop_calibration'
        self.json_path = self.json_dir + f'/crop_calibration_{date}.json' # 08_21_2025_crop_calibration.json

        self.counter = 0

        # Initialize inferer
        SIM_DATA_ROOT = "outputs" # Root dir where sim_cam1/, sim_cam2/ exist
        SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt" # Path to your SAM2 model
        SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"    # Path to your SAM2 config

        self.inferer = lego_online_infer.OnlineLegoInferer(
            sim_data_root_dir=SIM_DATA_ROOT,
            sam2_checkpoint_path=SAM2_CHECKPOINT,
            sam2_model_config_path=SAM2_CONFIG,
            crop_calibration_json_path=self.json_dir,
            device="cuda" # Use "cuda" if available and desired
        )

        rospy.Subscriber(f'/cam_destroyer/color/image_raw/compressed', CompressedImage, self.cam1_color_callback)
        rospy.Subscriber(f'/cam_architect/color/image_raw/compressed', CompressedImage, self.cam2_color_callback)
        rospy.Subscriber(f'/cam_destroyer/depth/image_raw/compressedDepth', CompressedImage, self.cam1_depth_callback)
        rospy.Subscriber(f'/cam_architect/depth/image_raw/compressedDepth', CompressedImage, self.cam2_depth_callback)
        rospy.loginfo("SideviewCalibrateNode initialized. Waiting for images...")
        self.run()

    def cam1_color_callback(self, msg):
        try:
            self.cam1_color = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
    def cam2_color_callback(self, msg):
        try:
            self.cam2_color = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
    def cam1_depth_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            self.cam1_depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")
    def cam2_depth_callback(self, msg):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            self.cam2_depth = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        except Exception as e:
            rospy.logerr(f"CV Bridge error: {e}")

    def calibrate(self, compute_new_T=True, reject_bad_T=False, depth=False):
        results = self.inferer.infer_dual_camera(
            self.cam1_color,
            self.cam2_color,
            self.task,
            live_depth_cam1_np=self.cam1_depth,
            live_depth_cam2_np=self.cam2_depth,
            compute_new_T=compute_new_T,
            save=False,
            display=False,
        )

        for sim_id, data in results.details.items():
            if sim_id != results.best_sim_id: continue #
            print(f"  Sim ID {sim_id}: Cam1 IoU={data['cam1_iou']:.4f}, Cam2 IoU={data['cam2_iou']:.4f}, Combined IoU: {data['combined_iou']:.4f}")
    
        visualize(results, cur_assembling_step=-1, save_path=f'{self.image_dir}vis_{self.counter:02}.png')
        
        if results.T1[0] is None or results.T2[0] is None:
            print("Warning: color transform was not found")
            return
        
        if depth and (results.T1[1] is None or results.T2[1] is None):
            print("Warning: depth transform was not found")
            return

        # print('====================')

        # print("self.calibrated_T = {'sim_cam1': (np.array(" + str(results.T1[0].tolist()) + "), " 
        #                                         + str(results.T1[1]) + "),"
        #                                         + "\n\t\t'sim_cam2': (np.array(" + str(results.T2[0].tolist()) + "), " 
        #                                         + str(results.T2[1]) + ")}")

        json_data = {
            "sim_cam1": {
                "matrix": results.T1[0].tolist(),
                "params": list(results.T1[1]) if depth else 'None'
            },
            "sim_cam2": {
                "matrix": results.T2[0].tolist(),
                "params": list(results.T2[1]) if depth else 'None'
            }
        }

        # Pretty print as JSON
        print(json.dumps(json_data, indent=2))
        with open(self.json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved calibration to {self.json_path}")
        self.counter += 1

    def run(self):
        while not rospy.is_shutdown():
            s = input("Press enter to calibrate (-1 to exit): ")
            if s == '-1': exit(0)
            
            if self.cam1_color is not None and self.cam2_color is not None \
                    and self.cam1_depth is not None and self.cam2_depth is not None:
                self.calibrate(depth=True)
            elif self.cam1_color is not None and self.cam2_color is not None:
                print('Warning: no depth')
                self.calibrate(depth=False)
            else:
                print(self.cam1_color is not None, self.cam2_color is not None, \
                        self.cam1_depth is not None, self.cam2_depth is not None)
                rospy.loginfo(f"Waiting for images.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task')
    args = parser.parse_args()
    print(args.task)

    rospy.init_node(f'sideview_calibration_node', anonymous=False)
    node = SideviewCalibrateNode(args.task)
    rospy.spin()
