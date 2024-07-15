#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os

class ImageStitchingNode(Node):
    def __init__(self):
        super().__init__('image_stitching_node')
        
        self.bridge = CvBridge()
        self.images = {}
        self.subscribers = []
        
        camera_topics = [
            "/overhead_camera/overhead_camera1/image_raw",
            "/overhead_camera/overhead_camera2/image_raw",
            "/overhead_camera/overhead_camera3/image_raw",
            "/overhead_camera/overhead_camera4/image_raw"
        ]
        
        for i, topic in enumerate(camera_topics):
            self.subscribers.append(self.create_subscription(
                Image,
                topic,
                self.make_image_callback(i),
                10))
        
        self.stitched_image_pub = self.create_publisher(Image, '/stitched_image', 10)
        
    def make_image_callback(self, camera_index):
        def image_callback(msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.images[camera_index] = self.preprocess_image(cv_image)
                
                self.get_logger().info(f'Received image from camera {camera_index}: {cv_image.shape}')
                
                if len(self.images) == 4:
                    self.stitch_images()
            except Exception as e:
                self.get_logger().error(f"Error in image_callback: {e}")
        return image_callback
    
    def preprocess_image(self, image):
        # Save the received images to disk for inspection
        os.makedirs("received_images", exist_ok=True)
        image_path = os.path.join("received_images", f"camera_{len(self.images)}.png")
        cv2.imwrite(image_path, image)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        # Convert back to BGR
        processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
        return processed

    def stitch_images(self, retry_count=3):
        images = [self.images[i] for i in sorted(self.images.keys())]

        # ORB Feature Detection
        orb = cv2.ORB_create()
        keypoints = []
        descriptors = []
        for img in images:
            kp, des = orb.detectAndCompute(img, None)
            keypoints.append(kp)
            descriptors.append(des)
            self.get_logger().info(f"Detected {len(kp)} keypoints in image")

        # Match features using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = []
        for i in range(len(images) - 1):
            matches.append(bf.match(descriptors[i], descriptors[i+1]))
            self.get_logger().info(f"Found {len(matches[-1])} matches between image {i} and {i+1}")

        # Stitch images using custom homography calculation
        try:
            stitched_image = self.custom_stitch(images, keypoints, matches)
            stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, "bgr8")
            self.stitched_image_pub.publish(stitched_msg)

            # Save the stitched image to disk for inspection
            os.makedirs("stitched_images", exist_ok=True)
            cv2.imwrite("stitched_images/stitched_image.png", stitched_image)

        except Exception as e:
            self.get_logger().error(f"Image stitching failed: {e}")
            if retry_count > 0:
                self.get_logger().info(f"Retrying image stitching ({retry_count} retries left)")
                self.stitch_images(retry_count - 1)
    
    def custom_stitch(self, images, keypoints, matches):
        # This method should implement custom homography calculation and stitching logic
        # For simplicity, let's assume a basic pairwise stitching approach
        stitched_image = images[0]
        for i in range(len(images) - 1):
            kp1, kp2 = keypoints[i], keypoints[i+1]
            match = matches[i]

            src_pts = np.float32([kp1[m.queryIdx].pt for m in match]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in match]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w, _ = images[i+1].shape
            stitched_image = cv2.warpPerspective(stitched_image, M, (w, h))

            # Blending the stitched image and the next image
            stitched_image[0:h, 0:w] = images[i+1]

        return stitched_image

def main(args=None):
    rclpy.init(args=args)
    node = ImageStitchingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()