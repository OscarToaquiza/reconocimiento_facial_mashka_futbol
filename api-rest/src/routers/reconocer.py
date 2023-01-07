# Planos de flask
from flask import Blueprint, jsonify, request
import face_recognition
from datetime import datetime
import numpy as np
from PIL import Image
import base64
import os
import cv2
import json

from utils.cronometer import Cronometer

main = Blueprint('reconocer_blueprint', __name__)

# unknown_image = face_recognition.load_image_file('./base_imgs_pry')
# facesNames = []
# rostrosEncoding = []
# for file_name in os.listdir(unknown_image):
#     known_image = face_recognition.load_image_file(unknown_image + "/" + file_name)
#     encoding = face_recognition.face_encodings(known_image)[0]
#     rostrosEncoding.append(encoding)
#     facesNames.append(file_name.split(".")[0])
    


# unknown_image = face_recognition.load_image_file('./base_imgs_pry/alomoto_calo_diego_ivan_0504658402.jpg')
# unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
# n = 0;
# for faceName in facesNames:
#     print(faceName)
#     #print(rostrosEncoding[n])
#     results  = face_recognition.compare_faces([rostrosEncoding[n]], unknown_encoding)
#     print(results)
#     if(results[0]):
#         break
#     n=n+1
#     imageFacesPath = "./base_imgs_pry"
# facesEncodings = []
# facesNames = []
# for file_name in os.listdir(imageFacesPath):
#     image = cv2.imread(imageFacesPath + "/" + file_name)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     scale_percent = 50 # percent of original size
#     width = int(image.shape[1] * scale_percent / 100)
#     height = int(image.shape[0] * scale_percent / 100)
#     dim = (width, height)


  
#     # resize image
#     resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

#     face_loc = face_recognition.face_locations(resized)[0]
    
#     cv2.rectangle(resized, (face_loc[3], face_loc[2]), (face_loc[1], face_loc[2] + 30), (125,220,0), -1)
#     cv2.rectangle(resized, (face_loc[3], face_loc[0]),(face_loc[1], face_loc[2]), (125,220,0), 2)


#     face_image_encodings = face_recognition.face_encodings(resized, known_face_locations=[face_loc])[0]
#     facesEncodings.append(face_image_encodings)
#     facesNames.append(file_name.split(".")[0])

#     cv2.imshow(file_name.split(".")[0], resized)
#     cv2.waitKey(0)

# print(facesEncodings)
# print(facesNames)

# frame = cv2.imread('./base_imgs_pry/alomoto_calo_diego_ivan_0504658402.jpg')
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# scale_percent = 50 # percent of original size
# width = int(frame.shape[1] * scale_percent / 100)
# height = int(frame.shape[0] * scale_percent / 100)
# dim = (width, height)
  
# # resize image
# resized2 = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


# face_location = face_recognition.face_locations(resized2)[0]

# cv2.rectangle(resized2, (face_location[3], face_location[2]), (face_location[1], face_location[2] + 30), (125,220,0), -1)
# cv2.rectangle(resized2, (face_location[3], face_location[0]),(face_location[1], face_location[2]), (125,220,0), 2)

# cv2.imshow("Resultado", resized2)
# cv2.waitKey(0)

# if face_location != []:
#     face_frane_encodings = face_recognition.face_encodings(resized2, known_face_locations=[face_loc])[0]
#     print(face_frane_encodings);
#     result = face_recognition.compare_faces([face_frane_encodings],facesEncodings, tolerance=0.6)
#     print("Result", result)
#     if result[0] == True:
#         text = "Rostros"
#         color = (125, 220, 0)
#     else:
#         text = "Desconocido"
#         color = (50, 50, 255)

#     cv2.rectangle(frame, (face_location[3], face_location[2]), (
#         face_location[1], face_location[2] + 30), color, -1)
#     cv2.rectangle(frame, (face_location[3], face_location[0]),
#                     (face_location[1], face_location[2]), color, 2)
#     cv2.putText(
#         frame, text, (face_location[3], face_location[2]+20), 2, 0.7, (255, 255, 255), 1)



@main.route('/')
def train_data():
    hora_inicio = datetime.now()

    unknown_image = face_recognition.load_image_file(
        './utils/img_tests/vale.jpg')

    unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

    print("Reconociendo ...")
    with open('./utils/base_datos_encoding.json') as file:
        data = json.load(file)
        nombre = "Desconocido_0000000000"
        for d in data:
            encoding = np.array(data[d])
            results = face_recognition.compare_faces(
                [encoding], unknown_encoding, tolerance=0.5)
            if (results[0]):
                nombre = d
    tiempo = Cronometer.obtener_tiempo_transcurrido_formateado(hora_inicio)
    return jsonify({
        "msg": nombre,
        "time": str(tiempo)
    })


@main.route('/person', methods=['POST'])
def post_data():
    try:
        print("Reconociendo ...")

        hora_inicio = datetime.now()
        file = request.files['foto']
        img = Image.open(file.stream)

        data = file.stream.read()
        #data = base64.encodebytes(data)
        # print(data)
        data = base64.b64encode(data).decode()
        #print(data)

        unknown_image = face_recognition.load_image_file(file)
        unknown_encoding = face_recognition.face_encodings(unknown_image)

        if( len(unknown_encoding) == 0 ):
            print(unknown_encoding)
            return jsonify({
                'msg': 'error - no se encontro rostros en la img',
                'name': 'No data',
                'time': 0,
                'size': [img.width, img.height], 
                'format': img.format,
                #'img': data
           })

        unknown_encoding = unknown_encoding[0]

        with open('./utils/base_datos_encoding.json') as file:
            dataJson = json.load(file)
            nombre = "Desconocido_0000000000"
            for d in dataJson:
                encoding = np.array(dataJson[d])
                results = face_recognition.compare_faces(
                    [encoding], unknown_encoding, tolerance=0.5)
                if (results[0]):
                    nombre = d
        
        #print( unknown_encoding )

        #TODO
        # MANIPULAR IMAGEN, AGREGAR EL CUADRO VERDE DONDE ESTE EL ROSTRO Y ENVIAR A BSE 64 A DATA
        

        tiempo = Cronometer.obtener_tiempo_transcurrido_formateado(hora_inicio)

        return jsonify({
                'msg': 'success',
                'name': nombre,
                'time': tiempo,
                'size': [img.width, img.height], 
                'format': img.format,
                'img': data
           })

    except Exception as ex:
        print( str(ex) )
        return jsonify({'msg': str(ex)}), 500

cv2.waitKey(0)
cv2.destroyAllWindows()
