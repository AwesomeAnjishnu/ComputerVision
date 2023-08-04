import cv2
import numpy as np
from random import randint
#OpenCV DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320),scale=1/255)

#Load Class Lists
classes = []
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

#Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    #Get frames
    ret, frame = cap.read()

    #Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    colors = {'person': [5, 58, 90], 'bicycle': [106, 198, 182], 'car': [217, 218, 10], 'motorbike': [73, 212, 246], 'aeroplane': [47, 204, 139], 'bus': [73, 255, 1], 'train': [15, 78, 206], 'truck': [132, 155, 126], 'boat': [12, 169, 74], 'traffic light': [18, 52, 107], 'fire hydrant': [119, 195, 251], 'stop sign': [20, 154, 176], 'parking meter': [98, 74, 126], 'bench': [184, 139, 232], 'bird': [85, 166, 164], 'cat': [148, 127, 242], 'dog': [203, 193, 190], 'horse': [136, 187, 244], 'sheep': [48, 6, 130], 'cow': [63, 207, 84], 'elephant': [108, 196, 99], 'bear': [29, 108, 228], 'zebra': [49, 26, 250], 'giraffe': [217, 151, 37], 'backpack': [20, 78, 160], 'umbrella': [46, 197, 146], 'handbag': [99, 56, 84], 'tie': [64, 207, 216], 'suitcase': [173, 181, 55], 'frisbee': [52, 198, 194], 'skis': [200, 93, 31], 'snowboard': [53, 32, 113], 'sports ball': [217, 172, 79], 'kite': [28, 250, 94], 'baseball bat': [13, 129, 243], 'baseball glove': [208, 41, 94], 'skateboard': [182, 159, 145], 'surfboard': [145, 147, 146], 'tennis racket': [247, 19, 216], 'bottle': [67, 147, 24], 'wine glass': [204, 1, 10], 'cup': [180, 134, 160], 'fork': [143, 245, 239], 'knife': [211, 50, 203], 'spoon': [68, 144, 100], 'bowl': [115, 21, 56], 'banana': [116, 201, 201], 'apple': [190, 38, 0], 'sandwich': [182, 235, 184], 'orange': [142, 228, 197], 'broccoli': [59, 2, 58], 'carrot': [74, 30, 78], 'hot dog': [103, 228, 29], 'pizza': [180, 206, 145], 'donut': [230, 118, 235], 'cake': [233, 199, 59], 'chair': [101, 146, 72], 'sofa': [226, 125, 59], 'pottedplant': [226, 77, 166], 'bed': [93, 120, 221], 'diningtable': [189, 101, 51], 'toilet': [205, 112, 180], 'tvmonitor': [12, 63, 201], 'laptop': [198, 132, 183], 'mouse': [70, 143, 48], 'remote': [98, 53, 193], 'keyboard': [183, 169, 121], 'cell phone': [239, 77, 20], 'microwave': [240, 216, 215], 'oven': [125, 45, 234], 'toaster': [181, 146, 42], 'sink': [121, 201, 221], 'refrigerator': [149, 4, 120], 'book': [189, 94, 26], 'clock': [213, 69, 130], 'vase': [205, 57, 236], 'scissors': [157, 251, 126], 'teddy bear': [13, 102, 62], 'hair drier': [85, 42, 42], 'toothbrush': [112, 70, 191]}

    #Code to create dictionary of colors
    """
    for class_name in classes:
        if class_name not in colors:
            colors[class_name] = [randint(0, 255), randint(0, 255), randint(0, 255)]
    print(colors)
    """

    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        cv2.putText(frame, class_name,(x,y - 10), cv2.FONT_HERSHEY_PLAIN, 2, color=colors[class_name], thickness=2)
        cv2.putText(frame, str(int(score*100)) + "%", ((x-10) + w, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, color=colors[class_name], thickness=2)
        cv2.rectangle(frame, (x,y),(x + w, y + h), color=colors[class_name], thickness=3)

    cv2.imshow('Computer Vision', frame)
    key = cv2.waitKey(1)
    if key == 32:
        break

cap.release()
cv2.destroyAllWindows()