import face_recognition
import numpy as np
import os
import json

unknown_image = face_recognition.load_image_file('./img_tests/oscar.jpg')
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#print(unknown_encoding)

with open('./base_datos_encoding.json') as file:
    data = json.load(file)
    for d in data:
        # print(d)
        # print(data[d])
        print(d)
        encoding = np.array(data[d])
        results  = face_recognition.compare_faces([encoding], unknown_encoding, tolerance=0.5)
        print(results)
        # if(results[0]):
        #     print(encoding)
        #     break
        # print(encoding)



# for faceName in facesNames:
#     print(faceName)
#     #print(rostrosEncoding[n])
#     results  = face_recognition.compare_faces([rostrosEncoding[n]], unknown_encoding)
#     print(results)
#     if(results[0]):
#         break
#     n=n+1