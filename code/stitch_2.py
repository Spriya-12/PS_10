import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImageStitchingNode(Node):
    def __init__(self):
        super().__init__('image_stitching_node')

        self.bridge = CvBridge()
        self.images = {}
        self.min_matches_threshold = 10  # Minimum acceptable matches between images
        self.camera_topics = self.declare_parameter('camera_topics', [
            '/overhead_camera/overhead_camera1/image_raw',
            '/overhead_camera/overhead_camera2/image_raw',
            '/overhead_camera/overhead_camera3/image_raw',
            '/overhead_camera/overhead_camera4/image_raw'
        ]).get_parameter_value().string_array_value

        # Create subscriptions for each camera topic
        for i in range(len(self.camera_topics)):
            self.create_subscription(Image, self.camera_topics[i], self.make_image_callback(i), 10)

        self.stitched_image_pub = self.create_publisher(Image, '/stitched_image', 10)

    def make_image_callback(self, camera_index):
        def image_callback(msg):
            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                self.images[camera_index] = self.preprocess_image(cv_image)

                self.get_logger().info(f'Received image from camera {camera_index + 1}: {cv_image.shape}')

                if len(self.images) == 4:
                    self.get_logger().info("All four images received. Proceeding to stitch.")
                    success, stitched_image = self.stitch_images()
                    if success:
                        self.publish_stitched_image(stitched_image)
                    else:
                        self.get_logger().error("Stitching failed. Not enough matches found.")

            except Exception as e:
                self.get_logger().error(f"Error in image_callback: {e}")
        return image_callback

    def preprocess_image(self, image):
        # Implement your desired preprocessing here (e.g., grayscale conversion)
        return image

    def stitch_images(self):
        images = [self.images[i] for i in sorted(self.images.keys())]

        detector = cv2.SIFT_create()
        keypoints, descriptors = [], []
        for img in images:
            kp, des = self.detect_and_compute_keypoints(img, detector)
            keypoints.append(kp)
            descriptors.append(des)

        matches = []
        for i in range(len(images) - 1):
            match = self.find_matches(images[i], images[i+1], keypoints[i], descriptors[i], keypoints[i+1], descriptors[i+1])
            if len(match) < self.min_matches_threshold:
                self.get_logger().warn(f"Found only {len(match)} matches between image {i} and {i+1}. Stitching might be unreliable.")
            matches.append(match)

        try:
            stitched_image = self.custom_stitch(images, keypoints, matches)
            return True, stitched_image
        except Exception as e:
            self.get_logger().error(f"Image stitching failed: {e}")
            return False, None

    def publish_stitched_image(self, stitched_image):
        stitched_msg = self.bridge.cv2_to_imgmsg(stitched_image, "bgr8")
        self.stitched_image_pub.publish(stitched_msg)

        # Display stitched image
        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(1)  # Wait for a key press to close the window

    def detect_and_compute_keypoints(self, image, detector):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def find_matches(self, img1, img2, kp1, des1, kp2, des2):
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        return matches[:50]  # Consider top 50 matches

    def custom_stitch(self, images, keypoints, matches):
        # Implement your custom stitching algorithm here.
        # For now, let's just return the first image as a placeholder.
        return images[0]

def main(args=None):
    rclpy.init(args=args)
    node = ImageStitchingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
