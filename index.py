# Libreria face_reconigtion_ modificado para ingresar 1 foto y devolver foto dibujada el rostro
from unittest import result
import os
import cv2
import face_recognition

imageFacesPath = "./Fotos"
facesEncodings = []
facesNames = []
for file_name in os.listdir(imageFacesPath):
    image = cv2.imread(imageFacesPath + "/" + file_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    scale_percent = 50 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
  
    # resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    face_loc = face_recognition.face_locations(resized)[0]
    
    cv2.rectangle(resized, (face_loc[3], face_loc[2]), (face_loc[1], face_loc[2] + 30), (125,220,0), -1)
    cv2.rectangle(resized, (face_loc[3], face_loc[0]),(face_loc[1], face_loc[2]), (125,220,0), 2)


    face_image_encodings = face_recognition.face_encodings(resized, known_face_locations=[face_loc])[0]
    facesEncodings.append(face_image_encodings)
    facesNames.append(file_name.split(".")[0])

    cv2.imshow(file_name.split(".")[0], resized)
    cv2.waitKey(0)

print(facesEncodings)
print(facesNames)

frame = cv2.imread('./ImgTest/Dylan.jpg')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

scale_percent = 50 # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
resized2 = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


face_location = face_recognition.face_locations(resized2)[0]

cv2.rectangle(resized2, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), (125,220,0), -1)
cv2.rectangle(resized2, (face_location[3], face_location[0]),(face_location[1], face_location[2]), (125,220,0), 2)

cv2.imshow("IMG IN", resized2)
cv2.waitKey(0)

if face_location != []:
    face_frane_encodings = face_recognition.face_encodings(resized2, known_face_locations=[face_loc])[0]
    print(face_frane_encodings);
    result = face_recognition.compare_faces([face_frane_encodings],facesEncodings, tolerance=0.6)
    print("Result", result)
    if result[0] == True:
        text = "Rostros"
        color = (125, 220, 0)
    else:
        text = "Desconocido"
        color = (50, 50, 255)

    cv2.rectangle(frame, (face_location[3], face_location[2]), (
        face_location[1], face_location[2] + 30), color, -1)
    cv2.rectangle(frame, (face_location[3], face_location[0]),
                    (face_location[1], face_location[2]), color, 2)
    cv2.putText(
        frame, text, (face_location[3], face_location[2]+20), 2, 0.7, (255, 255, 255), 1)


#cv2.imshow("Frame", frame)


# scale_percent = 50 # percent of original size
# width = int(frame.shape[1] * scale_percent / 100)
# height = int(frame.shape[0] * scale_percent / 100)
# dim = (width, height)
  
# # resize image
# resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
 
# print('Resized Dimensions : ',resized.shape)
 
# cv2.imshow("Resized image", resized)


cv2.waitKey(0)
cv2.destroyAllWindows()
