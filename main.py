from flask import Flask, render_template, url_for, Response
from model import Face

app = Flask(__name__, static_folder='static')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')


def livestream(source):
    while True:
        frame = source.input()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(livestream(Face()), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="localhost", debug=True)
