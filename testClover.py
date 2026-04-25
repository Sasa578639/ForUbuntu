import rospy
import cv2
import requests
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

rospy.init_node("hs2")
bridge = CvBridge()

def detect():
    msg = rospy.wait_for_message("main_camera/image_raw",Image,timeout=1)
    img_msg = bridge.imgmsg_to_cv2(msg,"bgr8")
    _,imencode=  cv2.imencode(".jpg",img_msg)

    results = requests.post("http://192.168.31.197:5000/detect",files={"image":imencode.tobytes()},timeout=3)
    data = results.json()
    class_names = data[0]["class"]
    x1 = data[0]["x1"]
    x2 = data[0]["x2"]
    y1 = data[0]["y1"]
    y2 = data[0]["y2"]
    conf = data[0]["conf"]
    print(conf,class_names,x1,x2,y1,y2)

detect()