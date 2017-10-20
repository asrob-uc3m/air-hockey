import cv2
from grip import GripPipeline as GripPipeline

if __name__ == '__main__':

    grip = GripPipeline()

    # Connect with webcam
    webcam = cv2.VideoCapture()
    webcam.open(1)

    # Do things here
    while True:
        _, new_image = webcam.read()
        grip.process(new_image)
        cv2.imshow("Image", grip.cv_medianblur_output)
        cv2.waitKey(30)
       
    cv2.destroyAllWindows()
