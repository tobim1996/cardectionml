from flask import Flask, render_template, Response
import cv2
import pafy
import datetime
import numpy as np
import imutils
import pandas as pd
from object_detection import ObjectDetection
import time

app = Flask(__name__)

url = "https://www.youtube.com/watch?v=YByJ2h0T5JY"
od = ObjectDetection()
camera_source = pafy.new(url).getbest()
camera = cv2.VideoCapture(camera_source.url)

#DEFINE AREA FOR DETECTION COORDINATES
area_detection = [(327,188),(549,171),(969,303),(563,413)]
area_speed = [(1213,369),(779, 561)]

#VARIABles FOR CAR COUNT & CAR SPEED
carcount = 0
last_carspeed = 0

def gen_frames():  # generate frame by frame from LIVE STREAM
    while True:
        success, frame = camera.read()  # read the camera frame
        frame = imutils.resize(frame, width=1280,height=720 )

        # BLEND TIME IN FRAME WITH COLOUR RIGHT HIGH CORNER INKLUSIVE FONT DEFINITION
        font_date = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        font_carcount = cv2.FONT_ITALIC
        dt = str(datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
        carcountstr = "Vehicle Detected: "
        lastspeedstr = "Last Speed: "

        frame = cv2.putText(frame, dt, (0, 25), font_date, 1, (0, 255, 255), 4, cv2.LINE_8)
        frame = cv2.putText(frame, carcountstr + str(carcount), (0, 700), font_carcount, 1, (0, 0, 255),4, cv2.LINE_8)
        frame = cv2.putText(frame, lastspeedstr + str(last_carspeed) + " Km/h", (940, 25),font_carcount, 1, (0, 0, 128), 4, cv2.LINE_8)

        #DRAW AREA PERMANANTLY ON FRAME FOR ROI AND SPEED DETECTION WITH area_detection and area_speed VARIABLES
        cv2.polylines(frame, [np.array(area_detection, np.int32)], True, (15, 220, 10), 8)
        cv2.polylines(frame, [np.array(area_speed, np.int32)], True, (212, 255, 127), 8)

        #Detect Objects on Frame
        (class_ids, scores,boxes) = od.detect(frame)
        for box in boxes:
            #print(box)
            (x,y,w,h) = box
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/video_feed')
def video_feed():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

#RUN WEB-SERVER to see PANDAS DATAFRAME in BROWSER OVER FLASK WEB SERVER WITHOUT CSS
@app.route('/dataframe')
def hello():
    x = pd.read_csv('carexample.csv',index_col=0)
    return x.style.to_html()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

app.run(host='0.0.0.0', port=81)