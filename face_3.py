import face_recognition
import os
import cv2

#Iterar imagenes de rostros
imageFacesPath = "./Fotos"
facesNames = []
rostrosEncoding = []

for file_name in os.listdir(imageFacesPath):
    known_image = face_recognition.load_image_file(imageFacesPath + "/" + file_name)
    encoding = face_recognition.face_encodings(known_image)[0]
    rostrosEncoding.append(encoding)
    facesNames.append(file_name.split(".")[0])

# known_image_1  = face_recognition.load_image_file('./Fotos/Dylan.jpg')
# known_image_2  = face_recognition.load_image_file('./Fotos/Anthony.jpg')
# known_image_3  = face_recognition.load_image_file('./Fotos/Mirian.jpg')

# dylan_encoding = face_recognition.face_encodings(known_image_1)[0]
# antony_encoding = face_recognition.face_encodings(known_image_2)[0]
# mirian_encoding = face_recognition.face_encodings(known_image_3)[0]

# rostrosEncoding = [ dylan_encoding, antony_encoding, mirian_encoding]

unknown_image = face_recognition.load_image_file('./ImgTest/Anthony.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#print(unknown_encoding)

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