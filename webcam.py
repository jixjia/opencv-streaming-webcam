import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera', default=0, type=int,
                    help='Select camera device id, in most cases 0 is Front and 1 for Back camera')
args = parser.parse_args()

cap = cv2.VideoCapture(args.camera)

while True:

    _, frame = cap.read()

    cv2.imshow('Output', frame)

    # Press ESC to exit
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
