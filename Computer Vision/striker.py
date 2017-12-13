import cv2

from grip import GripPipeline as GripPipeline


def detect_closed_hand(src):
    closed_hand = None
    # Get contours of hand mask
    contours, _ = cv2.findContours(src.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_contour = max(contours, key=cv2.contourArea)

    # Find bounding box
    hand_rect = cv2.boundingRect(hand_contour)  # Returns (x, y, w, h)
    test_points = [(x, y) for x, y in zip(range(hand_rect[0], hand_rect[0]+hand_rect[2]),
                                          range(hand_rect[1], hand_rect[1]+hand_rect[3]))]
    center = max(test_points, key=lambda x: cv2.pointPolygonTest(hand_contour, x, measureDist=True))
    dist = cv2.pointPolygonTest(hand_contour, center, measureDist=True)
    print(center, dist)
    closed_hand = center, dist

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
        center, radius = detect_closed_hand(grip.cv_medianblur_output)
        cv2.circle(new_image, center, int(radius), (255, 0, 0))
        cv2.imshow("Image", new_image)
        cv2.imshow("Mask", grip.cv_medianblur_output)
        if cv2.waitKey(30)!=-1:
            break

    cv2.destroyAllWindows()
