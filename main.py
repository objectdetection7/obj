import cv2
import numpy as np

#Load Yolo
net = cv2.dnn.readNet("C:\\Users\\student\\Downloads\\project\\yolov3-tiny.weights","C:\\Users\\student\\Downloads\\project\\yolov3-tiny.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


#Loading video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    

    height, width, channels = frame.shape

    #Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    #Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                #Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                #Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    object_count = 0

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])     
            confidence = confidences[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 2)  
            cv2.putText(frame, label , (x, y + 30),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255), 3)
            object_count += 1

            #Display object count
    cv2.putText(frame,f"Objects : {object_count}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 2)
    


    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



        