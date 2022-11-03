import face_recognition
import cv2

known_image_1  = face_recognition.load_image_file('./Fotos/Dylan.jpg')
known_image_2  = face_recognition.load_image_file('./Fotos/Anthony.jpg')
known_image_3  = face_recognition.load_image_file('./Fotos/Mirian.jpg')

dylan_encoding = face_recognition.face_encodings(known_image_1)[0]
antony_encoding = face_recognition.face_encodings(known_image_2)[0]
mirian_encoding = face_recognition.face_encodings(known_image_3)[0]

rostrosEncoding = [ dylan_encoding, antony_encoding, mirian_encoding]

unknown_image = face_recognition.load_image_file('./ImgTest/oscar_toaquiza_0504121344.jpeg')
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
#print(unknown_encoding)

x= range(3)

for n in x:
    #print(n)
    #print(rostrosEncoding[n])
    results  = face_recognition.compare_faces([rostrosEncoding[n]], unknown_encoding)
    print(results)

#print(results)