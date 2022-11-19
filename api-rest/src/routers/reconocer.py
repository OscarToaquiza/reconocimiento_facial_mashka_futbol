# Planos de flask
from flask import Blueprint, jsonify, request
import face_recognition
from datetime import datetime
import numpy as np
from PIL import Image
import base64
import json

from utils.cronometer import Cronometer

main = Blueprint('reconocer_blueprint', __name__)


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
        file = request.files['image']
        img = Image.open(file.stream)

        data = file.stream.read()
        #data = base64.encodebytes(data)
        # print(data)
        data = base64.b64encode(data).decode('utf-8')
        print(data)

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
                'img': data
           })

        unknown_encoding = unknown_encoding[0]

        with open('./utils/base_datos_encoding.json') as file:
            dataJson = json.load(file)
            nombre = "Desconocido_0000000000"
            for d in dataJson:
                encoding = np.array(data[d])
                results = face_recognition.compare_faces(
                    [encoding], unknown_encoding, tolerance=0.5)
                if (results[0]):
                    nombre = d
        
        print( unknown_encoding )
        

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
        return jsonify({'msg': str(ex)}), 500
