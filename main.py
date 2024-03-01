# Importa las clases necesarias Kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.label import Label

# Importa OpenCV para procesamiento de imágenes
import cv2

# Importa MediaPipe para el seguimiento de manos
import mediapipe as mp
import numpy as np
import pickle

# Clase de la aplicación Kivy
class MyApp(App):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        try:
            # Inicializa la captura de video desde la fuente especificada
            self.video_captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            # Inicializar el seguimiento de manos de MediaPipe
            self.mp_manos = mp.solutions.hands
            self.manos = self.mp_manos.Hands()

            # Cargar el modelo RandomForestClassifier desde el archivo pickle
            diccionario = pickle.load(open('model.p', 'rb'))
            self.modelo = diccionario['model']

            # Diccionario para mapear las etiquetas numéricas a los gestos
            self.etiquetas = {0: 'A',1:'E',2:'I',3:'O',4:'U'}

            self.detectar = ''
        except Exception as e:
            print("Error al iniciar la cámara de video:", str(e))

    # Método para la interfaz de la aplicación
    def build(self):
        # Crea un BoxLayout de caja vertical
        layout = BoxLayout(orientation='vertical', padding=10)

        # Etiqueta de título
        lbtitulo = Label(text='Detector de señas',
                         size_hint=(None, None),
                         pos_hint={'center_x': 0.5},
                         size=(200, 80))

        # Etiqueta para almacenar la letra detectada
        self.lbletra = Label(text='',
                             font_size='40sp',
                             size_hint=(None, None),
                             pos_hint={'center_x': 0.5})

        # Crea un widget de imagen para mostrar el video
        self.video_image = Image()

        # Agrega los widgets al boxlayout
        layout.add_widget(lbtitulo)
        layout.add_widget(self.video_image)
        layout.add_widget(self.lbletra)

        # Programa la actualización del video a 40 fotogramas por segundo
        Clock.schedule_interval(self.update, 1.0 / 40.0)

        # Devuelve el layout como la interfaz de la aplicación
        return layout

    # Método que se llama para actualizar el video
    def update(self, dt):
        # Lee un fotograma del video
        ret, frame = self.video_captura.read()

        # Si se capturó el fotograma correctamente, realiza el procedimiento correspondiente
        if ret:
            # Voltea el fotograma horizontalmente (para evitar el efecto espejo)
            frame = cv2.flip(frame, 1)
            
            # Convierte el fotograma de BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Procesa el fotograma para detectar manos utilizando MediaPipe
            res = self.manos.process(frame_rgb)

            # Si se detectaron manos en el fotograma, realiza lo siguiente
            if res.multi_hand_landmarks:
                # Dibuja círculos en los puntos de referencia de las manos
                for hand_landmarks in res.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        height, width, _ = frame.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), cv2.FILLED)

                # Predecir el gesto utilizando el modelo RandomForestClassifier
                prediccion = self.modelo.predict([np.asarray(self.extract_caracteristicas(res.multi_hand_landmarks[0]))])
                prediccion_letra = self.etiquetas[float(prediccion[0])]
                self.detectar = prediccion_letra

            # Rota el fotograma en 90 grados en sentido antihorario
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Convierte el fotograma procesado en una textura de Kivy
            textura = self.convert_frame(frame)

            # Asigna la textura al widget de imagen
            self.video_image.texture = textura
            self.lbletra.text = f'{self.detectar}'

    # Método para extraer características de la mano
    def extract_caracteristicas(self, hand_landmarks):
        datos = []
        for landmark in hand_landmarks.landmark:
            x = landmark.x
            y = landmark.y
            datos.append(x)
            datos.append(y)
        return datos

    # Método para convertir un fotograma de OpenCV en una textura de Kivy
    def convert_frame(self, frame):
        textura = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        textura.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
        return textura

    # Método para detener la aplicación y liberar los recursos
    def stop(self, instance):
        self.video_captura.release()
        cv2.destroyAllWindows()

# Verifica si el script se está ejecutando directamente
if __name__ == '__main__':
    # Inicia la aplicación Kivy
    MyApp().run()