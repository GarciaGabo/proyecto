import mediapipe as mp
import os
import cv2
import matplotlib.pyplot as plt
import pickle

mp_manos = mp.solutions.hands
mp_dibujo = mp.solutions.drawing_utils
mp_dibujo_estilos = mp.solutions.drawing_styles

manos = mp_manos.Hands(static_image_mode=True, min_detection_confidence=0.3)

DIRECTORIO = './data'

datos = []
etiquetas = []

for carpetas in os.listdir(DIRECTORIO):
    for directorio_img in os.listdir(os.path.join(DIRECTORIO,carpetas)):
        datos_extras = []
        imagen = cv2.imread(os.path.join(DIRECTORIO,carpetas,directorio_img))
        imagen_rgb = cv2.cvtColor(imagen,cv2.COLOR_BGR2RGB)

        resultados = manos.process(imagen_rgb)
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    datos_extras.append(x)
                    datos_extras.append(y)

            datos.append(datos_extras)
            etiquetas.append(carpetas)
f = open('data.pickle','wb')
pickle.dump({'data':datos,'labels': etiquetas}, f)
f.close()
