from ultralytics import YOLO
import cv2
import numpy as np
from flask import Flask,jsonify,request

model = YOLO("yolov8n.pt")
app = Flask(__name__)

@app.route("/detect",methods = ["POST"])
def detection():
    file = request.files["image"]
    try:
        bytes = file.read()
        np_arr  = np.frombuffer(bytes,np.uint8)
        img = cv2.imdecode(np_arr,cv2.IMREAD_COLOR)

        reuslts = model(img)
        detection_Objetcs = []


        for r in reuslts:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())

                class_names = model.names[int(box.cls[0])]
                conf  = float(box.conf[0])

                detection_Objetcs.append({
                    "class":class_names,
                    "conf":conf,
                    "x1":x1,
                    "x2":x2,
                    "y1":y1,
                    "y2":y2,
                })
        return jsonify(detection_Objetcs)

    except Exception as e:
        return jsonify(f"{e}")
    
if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=False)