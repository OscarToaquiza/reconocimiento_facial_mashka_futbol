import face_recognition
import os
import json
import cv2

#Iterar imagenes de rostros
imageFacesPath =cv2.imread ("./base_imgs_pry")
data = {}
face_loc = face_recognition.face_locations(imageFacesPath)[0]

face_image_encodings = face_recognition.face_encodings(imageFacesPath, known_face_locations=[face_loc])[0]
print("face_image_encodings:", face_image_encodings)

for file_name in os.listdir(imageFacesPath):
    known_image = face_recognition.load_image_file(imageFacesPath + "/" + file_name)
    encoding = face_recognition.face_encodings(known_image)[0]
    data[file_name.split(".")[0]] = encoding.tolist()

#Generar ficheros db
with open('./base_datos_encoding.json', 'w') as outfile:
    archivo = json.dump(data,outfile)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while True:
     ret, frame = cap.read()
     if ret == False: break
     frame = cv2.flip(frame, 1)
     face_locations = face_recognition.face_locations(frame, model="cnn")
     if face_locations != []:
          for face_location in face_locations:
               face_frame_encodings = face_recognition.face_encodings(frame, known_face_locations=[face_location])[0]
               result = face_recognition.compare_faces([face_image_encodings], face_frame_encodings)
               #print("Result:", result)
               if result[0] == True:
                    color = (125, 220, 0)
               else:
                    color = (50, 50, 255)
               cv2.rectangle(frame, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), color, -1)
               cv2.rectangle(frame, (face_location[3], face_location[0]), (face_location[1], face_location[2]), color, 2)
               cv2.putText(frame,  (face_location[3], face_location[2] + 20), 2, 0.7, (255, 255, 255), 1)
     cv2.imshow("Frame", frame)
     k = cv2.waitKey(1)
     if k == 27 & 0xFF:
          break
cap.release()
cv2.destroyAllWindows()
