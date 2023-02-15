import cv2


# Load MoveNet Thunder model
import utils
from data import BodyPart
from ml import Movenet

movenet = Movenet('movenet_thunder')

def test():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        #img = tf.cast(frame, dtype=tf.float32)
        img = frame
        inference_count = 3
        for _ in range(inference_count - 1):
            person = movenet.detect(img, reset_crop_region=False)
        image_np = utils.visualize(img, [person])
        cv2.imshow("frame", image_np)
        cv2.waitKey(1)

test()