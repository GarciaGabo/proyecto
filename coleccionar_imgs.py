import os
import cv2

DIRECTORIO = './data'
if not os.path.exists(DIRECTORIO):
    os.makedirs(DIRECTORIO)

numero_clases = 1
dataset_size = 100

video_captura = cv2.VideoCapture(0,cv2.CAP_DSHOW)
for json in range(numero_clases):
    if not os.path.exists(os.path.join(DIRECTORIO, str(json))):
        os.makedirs(os.path.join(DIRECTORIO, str(json)))

    print('Coleccionar datos de las clases {}'.format(json))

    done = False
    while True:
        ret, frame = video_captura.read()
        cv2.putText(frame, 'Presiona "Q" para iniciar! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                    cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    contador = 0
    while contador < dataset_size:
        ret, frame = video_captura.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(DIRECTORIO, str(json), '{}.jpg'.format(contador)), frame)

        contador += 1

video_captura.release()
cv2.destroyAllWindows()