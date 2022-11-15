import face_recognition
import os
import cv2
import json

#Iterar imagenes de rostros
imageFacesPath = "./base_imgs_pry"
# facesNames = []
# rostrosEncoding = []
data = {}

for file_name in os.listdir(imageFacesPath):
    known_image = face_recognition.load_image_file(imageFacesPath + "/" + file_name)
    encoding = face_recognition.face_encodings(known_image)[0]
    data[file_name.split(".")[0]] = encoding.tolist()
    # rostrosEncoding.append(encoding)
    # facesNames.append(file_name.split(".")[0])

# print(data)

#Generar ficheros db
with open('./base_datos_encoding.json', 'w') as outfile:
    archivo = json.dump(data,outfile)

# with open('./data_2.json', 'w') as outfile:
#     archivo = json.dump(imageFacesPath,outfile)

# unknown_image = face_recognition.load_image_file('./RECLEiMG/Alomoto_Calo_Diego_Ivan_0504658402.jpg')
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
# #print(unknown_encoding)

# n = 0;
# for faceName in facesNames:
#     print(faceName)
#     #print(rostrosEncoding[n])
#     results  = face_recognition.compare_faces([rostrosEncoding[n]], unknown_encoding)
#     print(results)
#     if(results[0]):
#         break
#     n=n+1

#print(results)