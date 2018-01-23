import cv2
import numpy as np


class CallbackWrapper(object):
    def __init__(self, points):
        self.points = points

    def __call__(self, event, x, y, flags, param, *args, **kwargs):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if self.points:
                # Look for closest point and change it
                closest_point_index, dist = min(enumerate(self.points),
                                                key=lambda p: np.sqrt((p[1][0]-x)**2+(p[1][1]-y)**2))
                self.points[closest_point_index] = (x, y)

            print(x, y)

if __name__ == '__main__':
    # Connect with webcam
    webcam = cv2.VideoCapture()
    webcam.open(1)

    # Attach callback
    points = []  # As lists are mutable this is ok because we can modify it later
    click_callback = CallbackWrapper(points)
    cv2.namedWindow('Calibration')
    cv2.setMouseCallback('Calibration', click_callback)

    # Get first image from webcam (needed to get image dimensions)
    _, new_image = webcam.read()

    # Initialization of calibration points
    p1 = (new_image.shape[1] / 3, new_image.shape[0] / 3)  # Green upper left
    p2 = (2 * new_image.shape[1] / 3, new_image.shape[0] / 3)  # Blue upper right
    p3 = (new_image.shape[1] / 3, 2 * new_image.shape[0] / 3)  # Red lower left
    p4 = (2 * new_image.shape[1] / 3, 2 * new_image.shape[0] / 3)  # Black lower right
    points.append(p1)
    points.append(p2)
    points.append(p3)
    points.append(p4)

    # Define other useful stuff
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 0, 0)]

    # Do things here
    while cv2.waitKey(10) == -1:
        _, new_image = webcam.read()

        # Find homography given the known dimensions of the air hockey
        ah_markers_h = 50  # Distance between marker centers in the direction parallel to the goal line
        ah_markers_w = 200  # Distance between marker centers in the direction perpendicular to the goal line
        ah_markers_points = [(0, 0), (ah_markers_w, 0), (0, ah_markers_h), (ah_markers_w, ah_markers_h)]

        if len(points) == len(ah_markers_points):
            h, status = cv2.findHomography(np.array(points, dtype=np.float), np.array(ah_markers_points, dtype=np.float))
            print(h, status)

            # Get corrected image
            image_corrected = cv2.warpPerspective(new_image, h, (ah_markers_w, ah_markers_h))
            cv2.imshow("correction", image_corrected)
        else:
            print(len(points), len(ah_markers_points))

        for point, color in zip(points, colors):
            cv2.circle(new_image, point, 3, color, 3)

        cv2.imshow("Calibration", new_image)
       
    cv2.destroyAllWindows()
