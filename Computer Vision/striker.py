import cv2
import numpy as np

from skin import Skin as GripPipeline


def detect_closed_hand(src):
    closed_hand = None
    try:
        # Get contours of hand mask
        contours, _ = cv2.findContours(src.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        skin_contour = max(contours, key=cv2.contourArea)
        # I used to do a DP approximation here with epsilon = 1
    except ValueError:
        return None, None

    # Find maximum inscribed circumference (MIC)
    skin_rect = cv2.boundingRect(skin_contour)  # Returns (x, y, w, h)
    test_points = [(x, y) for x, y in zip(range(skin_rect[0], skin_rect[0]+skin_rect[2]),
                                          range(skin_rect[1], skin_rect[1]+skin_rect[3]))]
    mic_center = max(test_points, key=lambda x: cv2.pointPolygonTest(skin_contour, x, measureDist=True))
    mic_radius = cv2.pointPolygonTest(skin_contour, mic_center, measureDist=True)
    print("Min. Inscribed Circunf. -> c:{} r:{}".format(mic_center, mic_radius))

    # Find minimum enclosing circumference
    # Note: it is assumed that hand is usually within 3.5 times the radius of the MIC
    # Note2: boundaries are checked to avoid using a ROI outside the image
    hand_region = (max(mic_center[0]-3.5*mic_radius, 0),
                   max(mic_center[1]-3.5*mic_radius, 0),
                   2*3.5*mic_radius if mic_center[0]+2*3.5*mic_radius < src.shape[1] else src.shape[1]-1,
                   2*3.5*mic_radius if mic_center[1]+2*3.5*mic_radius < src.shape[0] else src.shape[0]-1)
    hand_contour = np.array(filter(lambda p: hand_region[0] <= p[0][0] <= hand_region[0]+hand_region[2] and hand_region[1] <= p[0][1] <= hand_region[1]+hand_region[3],
                          skin_contour))
    try:
        mec_center, mec_radius = cv2.minEnclosingCircle(hand_contour)
    except:
        return None, None

    # Compare min and mec



    closed_hand =    tuple([int(i) for i in mec_center]), mec_radius

    return closed_hand

if __name__ == '__main__':

    grip = GripPipeline()

    # Connect with webcam
    webcam = cv2.VideoCapture()
    webcam.open(0)

    # Do things here
    while True:
        _, new_image = webcam.read()
        grip.process(new_image)
        center, radius = detect_closed_hand(grip.output)
        try:
            cv2.circle(new_image, center, int(radius), (255, 0, 0))
        except TypeError:
            print("No detection")
        cv2.imshow("Image", new_image)
        cv2.imshow("Mask", grip.output)
        if cv2.waitKey(30)!=-1:
            break

    cv2.destroyAllWindows()
