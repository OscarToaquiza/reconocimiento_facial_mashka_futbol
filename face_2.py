import face_recognition
import cv2
import os
import numpy as np
config = "model/yolov3.cfg"
# Weights
weights = "model/yolov3.weights"
# Labels
LABELS = open("model/coco.names").read().split("\n")
#print(LABELS, len(LABELS))
#Visualizamos distintos colores para cada clase---
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
#print("colors.shape:", colors.shape)
# recargamos el modelo o utilizamos el modelo----
net = cv2.dnn.readNetFromDarknet(config, weights)


# Hacer dinamico
imageFacesPath = "./Fotos"
facesNames = []
rostrosEncoding = []

for file_name in os.listdir(imageFacesPath):
    known_image = face_recognition.load_image_file(imageFacesPath + "/" + file_name)
    encoding = face_recognition.face_encodings(known_image)[0]
    rostrosEncoding.append(encoding)
    facesNames.append(file_name.split(".")[0])


unknown_image = face_recognition.load_image_file('./Fotos/Anthony.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#print(unknown_encoding)
height, width, _ = known_image.shape
# Creamos el  blob - expesificaremos la imagen de entrada 
blob = cv2.dnn.blobFromImage(unknown_image, 1 / 255.0, (416,416),
                              swapRB=True, crop=False)
print("blob.shape:", blob.shape)



x= range(3)
n = 0;
for faceName in facesNames:
    print(faceName)
    #print(rostrosEncoding[n])
    results  = face_recognition.compare_faces([rostrosEncoding[n]], unknown_encoding)
    print(results)
    if(results[0]):
        break
    n=n+1
#print(results)
ln = net.getLayerNames()
#print("ln:", ln)
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()] 
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
#print("ln:", ln)
net.setInput(blob)
outputs = net.forward(ln)
#print("outputs:", outputs)
boxes = []
confidences = []
classIDs = []
for output in outputs:
     for detection in output:
          #print("detection:", detection)
          scores = detection[5:]
          classID = np.argmax(scores)
          confidence = scores[classID]
          if confidence > 0.5:
               #print("detection:", detection)
               #print("classID:", classID)
               box = detection[:4] * np.array([width, height, width, height])
               #print("box:", box)
               (x_center, y_center, w, h) = box.astype("int")
               #print((x_center, y_center, w, h))
               x = int(x_center - (w / 2))
               y = int(y_center - (h / 2))
               #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
               boxes.append([x, y, w, h])
               confidences.append(float(confidence))
               classIDs.append(classID)
idx = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
print("idx:", idx)
if len(idx) > 0:
     for i in idx:
          (x, y) = (boxes[i][0], boxes[i][1])
          (w, h) = (boxes[i][2], boxes[i][3])
          color = colors[classIDs[i]].tolist()
          text = "{}: {:.3f}".format(LABELS[classIDs[i]], confidences[i])
          cv2.rectangle(unknown_image, (x, y), (x + w, y + h), color, 2)
          cv2.putText(unknown_image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, color, 1)
cv2.imshow("Fotos", unknown_image)
cv2.waitKey(0)
cv2.destroyAllWindows()