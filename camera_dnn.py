#!/usr/bin/env python
import cv2
import imutils
import config
import time
import numpy as np
from imutils.video import VideoStream

# Initialise runtime variables
# These variables are used to measure FPS rate
start_time = time.time()
frame_number = 0
fps = 0


class Camera(object):

    def __init__(self):
        # Initialise and load our pre-built DNN model
        self.net = cv2.dnn.readNetFromCaffe(config.prototxt, config.model)
        self.vs = VideoStream(src=0).start()

        # Warm-up model
        time.sleep(1)

    def __del__(self):
        self.vs.stop()

    def release(self):
        self.vs.stop()

    def input(self):

        # Access our global variables to track and calculate FPS
        global start_time
        global frame_number
        global fps

        # Initialize camera feed
        frame = self.vs.read()

        # Downsize frame to reduce processing complexity
        (_, originalWidth) = frame.shape[:2]
        frameResized = imutils.resize(
            frame, width=int(originalWidth * config.scale_factor))
        (height, width) = frameResized.shape[:2]

        # Pass frame into our model for prediction
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frameResized, (200, 200)),
            1.0,
            (200, 200),
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        predictions = self.net.forward()

        # Measure the FPS rate every 30 frames
        frame_number += 1
        if frame_number == 30:
            current_time = time.time()
            fps = round(30/(current_time-start_time), 1)
            frame_number = 0
            start_time = time.time()

        # Draw marker around detected faces
        for i in range(0, predictions.shape[2]):

            confidence = predictions[0, 0, i, 2]

            # Filter out weak predictions by ensuring the `confidence level` is
            # no less than the minimum threshold set in config.py
            if confidence < config.confience_threshold:
                continue

            # Get the bounding box coordinates of the faces
            box = predictions[0, 0, i, 3:7] * \
                np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw bounding box and put confidence level on screen
            cv2.rectangle(frameResized, (startX, startY),
                          (endX, endY), (255, 255, 0), 2)

            text = "{:.1f}%".format(confidence * 100)
            textY = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(frameResized, text, (startX, textY),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Display FPS rate when available
        cv2.putText(frameResized, "FPS: "+str(fps), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

        _, mjpeg = cv2.imencode('.jpg', frameResized)
        return mjpeg.tobytes()
