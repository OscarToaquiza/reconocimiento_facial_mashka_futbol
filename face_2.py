import face_recognition
import cv2
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
known_image_1  = face_recognition.load_image_file('./Fotos/Dylan.jpg')
known_image_2  = face_recognition.load_image_file('./Fotos/Anthony.jpg')
known_image_3  = face_recognition.load_image_file('./Fotos/Mirian.jpg')

dylan_encoding = face_recognition.face_encodings(known_image_1)[0]
antony_encoding = face_recognition.face_encodings(known_image_2)[0]
mirian_encoding = face_recognition.face_encodings(known_image_3)[0]

rostrosEncoding = [ dylan_encoding, antony_encoding, mirian_encoding]
# Hacer dinamico


unknown_image = face_recognition.load_image_file('./Fotos/Dylan.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#print(unknown_encoding)
height, width, _ = known_image_1.shape
# Creamos el  blob - expesificaremos la imagen de entrada 
blob = cv2.dnn.blobFromImage(known_image_1, 1 / 255.0, (316,316),
                              swapRB=True, crop=False)
print("blob.shape:", blob.shape)



x= range(3)

for n in x:
    #print(n)
    #print(rostrosEncoding[n])
    results  = face_recognition.compare_faces([rostrosEncoding[n]], unknown_encoding)
    print(results)

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
          cv2.rectangle(known_image_1, (x, y), (x + w, y + h), color, 2)
          cv2.putText(known_image_1, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                         0.5, color, 2)
cv2.imshow("Image", known_image_1)
cv2.waitKey(0)
cv2.destroyAllWindows()