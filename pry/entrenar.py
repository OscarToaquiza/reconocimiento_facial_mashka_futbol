import face_recognition
import os
import json

#Iterar imagenes de rostros
imageFacesPath = "./base_imgs_pry"
data = {}

for file_name in os.listdir(imageFacesPath):
    known_image = face_recognition.load_image_file(imageFacesPath + "/" + file_name)
    encoding = face_recognition.face_encodings(known_image)[0]
    data[file_name.split(".")[0]] = encoding.tolist()

#Generar ficheros db
with open('./base_datos_encoding.json', 'w') as outfile:
    archivo = json.dump(data,outfile)
