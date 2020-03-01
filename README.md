## Apply Deep Learning Model on Live Video and Display it in Browser

This sample program shows how to run your own deep leraning model on a webcam (or network IP camera) live feed, process them and stream results to a web client browser in real-time.

## Demo
![DEMO](https://jixjiastorage.blob.core.windows.net/blog-resources/opencv-stream-webcam/demo-static.png)

## Architecture
![DEMO](https://jixjiastorage.blob.core.windows.net/blog-resources/opencv-stream-webcam/architecture-static.png)

1. Use OpenCV to read and process live camera feed
1. Initialise and load our DNN model to perform scoring (in this demo I will use a pre-trained facial model to detect human faces)
1. Stream outputs back to client browser in real-time using Flask

## How-To
1. Clone my repo git clone https://github.com/jixjia/opencv-streaming-webcam.git 
1. Create a virtual environment conda create -n myenv python==3.6.8
1. Activate it by activate myenv
1. Install dependencies pip install -r requirements.txt
1. Run python main.py
1. Open up your browser and navigate to localhost:5000

## Further Details
More details about the solution is available on my [blog](https://jixjia.com/2020/03/01/dnn-live-feed-streaming/)
