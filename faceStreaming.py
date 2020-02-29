import cv2
import argparse
from sys import argv

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
parser = argparse.ArgumentParser(description='Specify protobuf model and a matching label text')
parser.add_argument('-c', '--camera', default=0, type=int, help='Camera (0)=Front (1)=Back')
args = parser.parse_args()

# Initiate Video Cam input (live feed)
cap = cv2.VideoCapture(args.camera)
while (True):
    ret,img = cap.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(grayImg, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            
    cv2.imshow('Face Streaming',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

