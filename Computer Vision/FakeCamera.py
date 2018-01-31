import cv2


class FakeCamera(object):
    """
    Fake camera for debug purposes. Read a still frame from a file and return it whenever the camera is read
    """
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)

    def open(self, *args, **kwargs):
        # Don't need to do anything for fake camera
        pass

    def read(self):
        return True, self.img.copy()

